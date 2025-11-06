import re
import ast
from constants import ReturnFormat
from parsers import (
    parse_java_function_call, 
    parse_json_function_call, 
    parse_verbose_xml_function_call, 
    parse_concise_xml_function_call
    )

def resolve_ast_call(call_node: ast.Call) -> dict:
    func_name = ""
    if isinstance(call_node.func, ast.Name):
        func_name = call_node.func.id
    elif isinstance(call_node.func, ast.Attribute):
        func_name = call_node.func.attr
    else:
        raise ValueError(f"Unsupported function node type: {type(call_node.func)}")

    param_dict = {}
    for keyword in call_node.keywords:
        key = keyword.arg
        value = ast.literal_eval(keyword.value)
        param_dict[key] = value

    return {func_name: param_dict}

def ast_parse(
    input_str: str,
    language: ReturnFormat = ReturnFormat.PYTHON,
    has_tool_call_tag: bool = False,
) -> list[dict]:
    if has_tool_call_tag:
        match = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", input_str, re.DOTALL)
        if match:
            input_str = match.group(1).strip()
        else:
            raise ValueError(f"No tool call tag found in input string: {input_str}")

    if language == ReturnFormat.PYTHON:
        # We only want to remove wrapping quotes that could have been added by the model.
        cleaned_input = input_str.strip().strip("'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                print("elem:", ast.dump(elem))
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return extracted

    elif language == ReturnFormat.JAVA:
        # Remove the [ and ] from the string
        # Note: This is due to legacy reasons, we should fix this in the future.
        return parse_java_function_call(input_str[1:-1])

    elif language == ReturnFormat.VERBOSE_XML:
        # Remove ```xml and anything before/after XML
        match = re.search(r"<functions>(.*?)</functions>", input_str, re.DOTALL)
        if not match:
            raise ValueError(
                f"No XML function call found in input string: {input_str}. Missing <functions> tag."
            )
        return parse_verbose_xml_function_call(match.group(0))

    elif language == ReturnFormat.CONCISE_XML:
        # Remove anything before/after <functions> and </functions>
        match = re.search(r"<functions>(.*?)</functions>", input_str, re.DOTALL)
        if not match:
            raise ValueError(
                f"No XML function call found in input string: {input_str}. Missing <functions> tag."
            )
        return parse_concise_xml_function_call(match.group(0))

    elif language == ReturnFormat.JSON:
        json_match = re.search(r"\[.*\]", input_str, re.DOTALL)
        if json_match:
            input_str = json_match.group(0)
        return parse_json_function_call(input_str)

    else:
        print(language, type(language), type(ReturnFormat.PYTHON), language == ReturnFormat.PYTHON)
        raise NotImplementedError(f"Unsupported language: {language}")


def default_decode_ast_prompting(
    result: str,
    language: ReturnFormat = ReturnFormat.PYTHON,
    has_tool_call_tag: bool = False,
) -> list[dict]:
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    decoded_output = ast_parse(result, language, has_tool_call_tag)
    return decoded_output