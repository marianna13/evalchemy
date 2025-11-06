from tree_sitter import Language, Parser
import tree_sitter_java
import json
import re
import ast

import xml.etree.ElementTree as ET



JAVA_LANGUAGE = Language(tree_sitter_java.language(), "java")

parser = Parser()
parser.set_language(JAVA_LANGUAGE)


def extract_function_calls(source_code: str) -> list[dict]:
    """
    Extract function calls from the given source code string.

    Args:
        source_code (str): The source code containing function calls.

    Returns:
        list[dict]: A list of extracted function call dictionaries.
    """
    regex = r"```json(.*?)```"
    matches = re.findall(regex, source_code, re.DOTALL)
    extracted_calls = []
    for match in matches:
        if match.strip():
            if "name" in match and "arguments" in match:
                extracted_calls.append(match.strip())
    return extracted_calls


def convert_value_by_type(raw_value, type_str):
    if type_str == "string":
        return raw_value
    elif type_str == "integer":
        return int(raw_value)
    elif type_str == "float":
        return float(raw_value)
    elif type_str == "boolean":
        return raw_value.lower() == "true"
    elif type_str == "null":
        return None
    elif (
        type_str == "array"
        or type_str == "tuple"
        or type_str == "object"
        or type_str == "dict"
    ):
        return ast.literal_eval(raw_value)
    else:
        return raw_value  # fallback to raw string


def parse_verbose_xml_function_call(input_str):
    root = ET.fromstring(input_str)
    results = []

    for func in root.findall("function"):
        func_name = func.attrib["name"]
        param_dict = {}

        params_container = func.find("params")
        param_elements = (
            params_container.findall("param")
            if params_container is not None
            else func.findall("param")
        )

        for param in param_elements:
            name = param.attrib.get("name")
            raw_value = param.attrib.get("value", "")
            type_str = param.attrib.get("type", "string")
            parsed_value = convert_value_by_type(raw_value, type_str)
            param_dict[name] = parsed_value

        results.append({func_name: param_dict})

    return results


def parse_concise_xml_function_call(input_str):
    root = ET.fromstring(input_str)
    results = []

    for func in root.findall("function"):
        func_name = func.attrib["name"]
        param_dict = {}

        for param in func.findall("param"):
            name = param.attrib["name"]
            raw_value = param.text or ""
            type_str = param.attrib.get("type", "string")
            parsed_value = convert_value_by_type(raw_value.strip(), type_str)
            param_dict[name] = parsed_value

        results.append({func_name: param_dict})

    return results




def parse_json_function_call(source_code):
    json_match = re.search(r"\[\s*{.*?}\s*(?:,\s*{.*?}\s*)*\]", source_code, re.DOTALL)
    if json_match:
        source_code = json_match.group(0)

    try:
        json_dict = json.loads(source_code)
    except json.JSONDecodeError as e:
        return []

    function_calls = []
    for function_call in json_dict:
        if isinstance(function_call, dict):
            function_name = function_call["function"]
            arguments = function_call["parameters"]
            function_calls.append({function_name: arguments})
    return function_calls


def parse_java_function_call(source_code):
    tree = parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node
    sexp_result = root_node.sexp()

    if "ERROR" in sexp_result:
        raise SyntaxError("Error parsing java the source code.")

    def get_text(node):
        """Returns the text represented by the node."""
        return source_code[node.start_byte : node.end_byte]

    def traverse_node(node, nested=False):
        if node.type == "string_literal":
            if nested:
                return get_text(node)
            # Strip surrounding quotes from string literals
            return get_text(node)[1:-1]
        elif node.type == "character_literal":
            if nested:
                return get_text(node)
            # Strip surrounding single quotes from character literals
            return get_text(node)[1:-1]
        """Traverse the node to collect texts for complex structures."""
        if node.type in [
            "identifier",
            "class_literal",
            "type_identifier",
            "method_invocation",
        ]:
            return get_text(node)
        elif node.type == "array_creation_expression":
            # Handle array creation expression specifically
            type_node = node.child_by_field_name("type")
            value_node = node.child_by_field_name("value")
            type_text = traverse_node(type_node, True)
            value_text = traverse_node(value_node, True)
            return f"new {type_text}[]{value_text}"
        elif node.type == "object_creation_expression":
            # Handle object creation expression specifically
            type_node = node.child_by_field_name("type")
            arguments_node = node.child_by_field_name("arguments")
            type_text = traverse_node(type_node, True)
            if arguments_node:
                # Process each argument carefully, avoiding unnecessary punctuation
                argument_texts = []
                for child in arguments_node.children:
                    if child.type not in [
                        ",",
                        "(",
                        ")",
                    ]:  # Exclude commas and parentheses
                        argument_text = traverse_node(child, True)
                        argument_texts.append(argument_text)
                arguments_text = ", ".join(argument_texts)
                return f"new {type_text}({arguments_text})"
            else:
                return f"new {type_text}()"
        elif node.type == "set":
            # Handling sets specifically
            items = [
                traverse_node(n, True)
                for n in node.children
                if n.type not in [",", "set"]
            ]
            return "{" + ", ".join(items) + "}"

        elif node.child_count > 0:
            return "".join(traverse_node(child, True) for child in node.children)
        else:
            return get_text(node)

    def extract_arguments(args_node):
        arguments = {}
        for child in args_node.children:
            if child.type == "assignment_expression":
                # For named parameters
                name_node, value_node = child.children[0], child.children[2]
                name = get_text(name_node)
                value = traverse_node(value_node)
                if name in arguments:
                    if not isinstance(arguments[name], list):
                        arguments[name] = [arguments[name]]
                    arguments[name].append(value)
                else:
                    arguments[name] = value
                # arguments.append({'name': name, 'value': value})
            elif child.type in ["identifier", "class_literal", "set"]:
                # For unnamed parameters and handling sets
                value = traverse_node(child)
                if None in arguments:
                    if not isinstance(arguments[None], list):
                        arguments[None] = [arguments[None]]
                    arguments[None].append(value)
                else:
                    arguments[None] = value
        return arguments

    def traverse(node):
        if node.type == "method_invocation":
            # Extract the function name and its arguments
            method_name = get_text(node.child_by_field_name("name"))
            class_name_node = node.child_by_field_name("object")
            if class_name_node:
                class_name = get_text(class_name_node)
                function_name = f"{class_name}.{method_name}"
            else:
                function_name = method_name
            arguments_node = node.child_by_field_name("arguments")
            if arguments_node:
                arguments = extract_arguments(arguments_node)
                for key, value in arguments.items():
                    if isinstance(value, list):
                        raise Exception(
                            "Error: Multiple arguments with the same name are not supported."
                        )
                return [{function_name: arguments}]

        else:
            for child in node.children:
                result = traverse(child)
                if result:
                    return result

    result = traverse(root_node)
    return result if result else {}


