from typing import Dict, List, Any, Optional
import logging
import torch
import datasets
from tqdm import tqdm
import pandas as pd
import os
import ast
import traceback
import re


from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark
import json
import copy
from datasets import Dataset
from .ast_eval.ast_checker import ast_checker
from .utils import default_decode_ast_prompting, ast_parse
from .constants import Language, ReturnFormat
from .parsers import extract_function_calls


def decode_ast(result, language, has_tool_call_tag):
    return default_decode_ast_prompting(result, language, has_tool_call_tag)

def is_empty_output(decoded_result: list[dict]) -> bool:
    """Check if the decoded result is empty (no function calls)."""
    if not decoded_result:
        return True

    for func_call in decoded_result:
        if func_call.get("name") or func_call.get("arguments"):
            return False

    return True

def is_function_calling_format_output(decoded_output):
    """
    Ensure the output is a list of dictionaries of the form:
    `[{func1: {param1: val1, param2: val2, ...}}, {func2: {param1: val1, param2: val2, ...}}, ...]`
    Sometimes the model handler's `decode_ast` method will return successfully, but the output is not in the correct format, and that will mess up the downstream evaluation that expects this format.
    This is especially the case when the model doesn't predict any function calls, and the output is an human-readable string.
    Note: Empty list `[]` is considered the correct format in this check.
    """
    if type(decoded_output) != list:
        return False
    for item in decoded_output:
        if type(item) != dict:
            return False
        # Check for `{func1: {param1: val1, param2: val2, ...}}`, should only have one key-value pair
        if len(item) != 1:
            return False
        # Check for `{param1: val1, param2: val2, ...}`; the parameter-value pairs should be a dictionary
        if type(list(item.values())[0]) != dict:
            return False
    return True


def extract_model_result_items(response_text: str) -> str:
    # Try to find JSON code blocks (between ```json ... ```)
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", response_text, flags=re.DOTALL)
    if json_blocks:
        json_str = json_blocks[0]
    else:
        # fallback: try to find the first [ ... ] block
        match = re.search(r"\[.*\]", response_text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in response text.\n Response:\n" + response_text)
        json_str = match.group(0)

    # Now parse JSON
    try:
        model_result_items = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}\nExtracted:\n{json_str}")

    model_result_items = json.dumps(model_result_items)
    return model_result_items



def _evaluate_single_ast_entry(
    index,
    model_result_item,
    possible_answer_item,
    prompt_entry,
    model_name,
    test_category,
    language: Language,
    return_format: ReturnFormat,
    has_tool_call_tag=False,
):
    """Helper method to process a single AST entry."""
    prompt_function = prompt_entry["function"]


    try:
        model_result_item_raw = model_result_item
        model_result_item = extract_model_result_items(model_result_item)
        model_result_item = json_to_func_call(model_result_item)
        # model_result_item = decode_execute(model_result_item, language, has_tool_call_tag)
        model_result_item = decode_ast(
            model_result_item, return_format, has_tool_call_tag
        )
    except Exception as e:
        traceback.print_exc()
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": [f"Invalid syntax. Failed to decode AST. {str(e)}"],
            "error_type": "ast_decoder:decoder_failed",
            "prompt": prompt_entry,
            "model_result_raw": model_result_item_raw,
            "possible_answer": possible_answer_item,
        }

    decoder_output_valid = is_function_calling_format_output(model_result_item)
    if not decoder_output_valid:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": [
                "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
            ],
            "error_type": "ast_decoder:decoder_wrong_output_format",
            "prompt": prompt_entry,
            "model_result_raw": str(model_result_item_raw),
            "model_result_decoded": str(model_result_item),
            "possible_answer": possible_answer_item,
        }

    checker_result = ast_checker(
        [prompt_function],
        [model_result_item],
        possible_answer_item,
        language,
        # format sensitivity has parallel, multiple cases which is encoded in index
        test_category if test_category != 'format_sensitivity' else index.split(':')[-1],
        model_name,
    )

    if not checker_result["valid"]:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": checker_result["valid"],
            "error": checker_result["error"],
            "error_type": checker_result["error_type"],
            "prompt": prompt_entry,
            "model_result_raw": model_result_item_raw,
            "model_result_decoded": model_result_item,
            "possible_answer": possible_answer_item,
        }
    return {"valid": True}


def load_file(file_path: str):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))
    return result

def dotted_name_to_ast(name: str) -> ast.expr:
    """
    Convert a dotted name like 'obj.method' or 'pkg.mod.func'
    into an AST node: ast.Attribute(...ast.Name(...), 'method')
    """
    parts = name.split('.')
    node = ast.Name(id=parts[0], ctx=ast.Load())
    for attr in parts[1:]:
        node = ast.Attribute(value=node, attr=attr, ctx=ast.Load())
    return node


def json_to_func_call(json_str: str) -> str:
    """
    Parse a JSON-like string into a function call string using ast.
    
    Example:
        Input:  '{"name": "func1", "arguments": {"param1": "val1", "param2": "val2"}}'
        Output: 'func1(param1=val1, param2=val2)'
    """
    # Safely parse JSON-like string into a Python dict
    try:
        data = json.loads(json_str)
    except (SyntaxError, ValueError) as e:
        print("Error parsing JSON:", e, json_str)
        return ""

    data = data if isinstance(data, dict) else data[0]  # Handle list of dicts

    # Extract function name and arguments
    print("Parsed data:", data)
    func_name = data["tool_call_name"] if "tool_call_name" in data else data.get("name", "")
    args = data.get("arguments", {})

    if "." in func_name:
        func_node = dotted_name_to_ast(func_name)
    else:
        func_node = ast.Name(id=func_name, ctx=ast.Load())

    # Build the AST node for the function call
    call_node = ast.Call(
        func=func_node,
        args=[],
        keywords=[ast.keyword(arg=k, value=ast.Constant(v)) for k, v in args.items()]
    )
    if "." in func_name:
        print("elem:", ast.dump(call_node))

    # Convert AST back to a string
    call_str = ast.unparse(call_node)
    
    return call_str


def decode_execute(result, language, has_tool_call_tag) -> list[str]:
    """Decode the model result and convert to executable function calls.
    input: model result string
    output: list of executable function call strings
    Input example: '{"name": "func1", "arguments": {"param1": "val1", "param2": "val2"}}, {"name": "func2", "arguments": {"param1": "val1"}}'
    Example output: ["func1(param1=val1, param2=val2)", "func2(param1=val1)"]
    """
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    decoded_output = []
    func = result

    decoded_output = ast_parse(func, language, has_tool_call_tag)
    execution_list = []
    for function_call in decoded_output:
        for key, value in function_call.items():
            execution_list.append(
                f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
            )
    return execution_list


def load_json_dataset(test_entries: List[Dict[str, Any]]):
    data = {"id": [], "question": [], "function": []}
    test_entries_copy = copy.deepcopy(test_entries)

    for item in test_entries_copy:
        data["id"].append(item["id"])
        data["question"].append(item["question"])

        for func in item["function"]:
            func["parameters"]["properties"] = json.dumps(
                func["parameters"]["properties"]
            )
        data["function"].append(func)
    return Dataset.from_dict(data)


def _load_dataset(file_path: str) -> Dataset:
    this_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_script_dir, "data")
    file_path = os.path.join(data_dir, file_path + ".json")
    test_entries = load_file(file_path)
    dataset = load_json_dataset(test_entries)
    return dataset

def _load_possible_answers(file_path: str) -> Dict[str, Any]:
    this_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_script_dir, "data/possible_answer")
    file_path = os.path.join(data_dir, file_path + ".json")
    possible_answers = load_file(file_path)
    possible_answers_dict = {item["id"]: item["ground_truth"] for item in possible_answers}
    return possible_answers_dict


def _format_prompt(messages, function):
        formatted_prompt = ""

        system_message = "You are a helpful assistant that can use tools. You are developed by Salesforce xLAM team."
        remaining_messages = messages
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"].strip()
            remaining_messages = messages[1:]

        # Format system message with tool instructions
        formatted_prompt += "<|im_start|>system\n"
        formatted_prompt += system_message + "\n"
        formatted_prompt += "You have access to a set of tools. When using tools, make calls in a single JSON array: \n\n"
        formatted_prompt += '[{"name": "tool_call_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ... (additional parallel tool calls as needed)]\n\n'
        formatted_prompt += "If no tool is suitable, state that explicitly. If the user's input lacks required parameters, ask for clarification. "
        formatted_prompt += "Do not interpret or respond until tool results are returned. Once they are available, process them or make additional calls if needed. "
        formatted_prompt += "If at any point you need to use a tool, include all necessary parameters in the tool call. "
        formatted_prompt += "Ensure all tool calls are valid according to the provided tool definitions. Use the following format for tool calls:\n\n"
        formatted_prompt += '```json\n[{"name": "tool_call_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ...]\n```\n\n'
        formatted_prompt += "For tasks that don't require tools, such as casual conversation or general advice, respond directly in plain text. The available tools are:\n\n"

        for func in function:
            formatted_prompt += json.dumps(func, indent=4) + "\n\n"
        formatted_prompt += "<|im_end|>"

        # Format conversation messages
        for message in remaining_messages:
            if message["role"] == "tool":
                formatted_prompt += "<|im_start|>tool\n"
                if isinstance(message["content"], (dict, list)):
                    formatted_prompt += json.dumps(message["content"])
                else:
                    formatted_prompt += message["content"]
                formatted_prompt += "<|im_end|>"
            elif "tool_calls" in message and message["tool_calls"]:
                formatted_prompt += "<|im_start|>assistant\n"
                tool_calls = []
                for tool_call in message["tool_calls"]:
                    tool_calls.append(
                        {
                            "name": tool_call["function"]["name"],
                            "arguments": json.loads(tool_call["function"]["arguments"]),
                        }
                    )
                formatted_prompt += json.dumps(tool_calls) + "<|im_end|>"
            else:
                formatted_prompt += (
                    f"<|im_start|>{message['role']}\n{message['content'].strip()}<|im_end|>"
                )

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt



class BFCLBenchmark(BaseBenchmark):
    """
    A BFCL benchmark for evaluating language model responses on function calling instructions.
    """


    def __init__(
        self,
        dataset_name: str = "BFCL_v4_live_simple",
        subset: str = "alpaca_eval",
        split: str = "eval",
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.5,
        do_sample: bool = True,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize BFCL benchmark.

        Args:
            dataset_name: HuggingFace dataset name
            subset: Dataset subset name
            split: Dataset split to use
            max_tokens: Maximum number of tokens for generation
            temperature: Sampling temperature
            do_sample: Whether to use sampling for generation
            debug: debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.max_tokens = max_tokens if max_tokens is not None else 1024
        self.temperature = temperature
        self.do_sample = do_sample
        self.debug = debug

    def load_dataset(self) -> datasets.Dataset:
        """Load the evaluation dataset."""
        try:
            dataset = _load_dataset(self.dataset_name)

            if self.debug:
                dataset = dataset.select(range(2))
                self.logger.info(f"Debug mode: using 2 examples")

            self.logger.info(f"Loaded {len(dataset)} examples for evaluation")
            return dataset

        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate completions for instructions using the provided model.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing model outputs and identifier
        """
        try:
            eval_set = self.load_dataset()
            all_instances = []
            for idx, example in enumerate(eval_set):
                try:
                    formatted_instruction = _format_prompt(
                        messages=example["question"][0],
                        function=[example["function"]],
                    )

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                formatted_instruction,
                                {
                                    "max_new_tokens": self.max_tokens,
                                    "do_sample": self.do_sample,
                                    "temperature": self.temperature,
                                },
                            ),
                            idx,
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error preparing instance {idx}: {str(e)}")
                    traceback.print_exc()
                    continue

            with torch.no_grad():
                self.logger.info("Generating responses for Alpaca Eval...")
                outputs = self.compute(model, all_instances)

            if model.rank != 0:
                return None

            model_outputs = []
            for idx, (example, output) in enumerate(zip(eval_set, outputs)):
                try:
                    instance = {
                        "instruction": example["question"],
                        "id": example["id"],
                        "generator": model.model_identifier,
                        "output": output,
                    }
                    model_outputs.append(instance)
                except Exception as e:
                    self.logger.error(f"Error processing output {idx}: {str(e)}")
                    continue

            self.logger.info(f"Generated {len(model_outputs)} responses")

            return {"model_outputs": model_outputs, "model_identifier": model.model_identifier}

        except Exception as e:
            self.logger.error(f"Error in generate_responses: {str(e)}")
            raise

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated responses using Alpaca evaluation metrics.

        Args:
            results: Dictionary containing model outputs and identifier

        Returns:
            Dictionary containing evaluation metrics
        """
        if results is None:
            return None
        model_outputs = results["model_outputs"]
        model_name = results["model_identifier"]
        possible_answers_dict = _load_possible_answers(self.dataset_name)

        evaluation_results = []
        for item in tqdm(model_outputs, desc="Evaluating responses"):
            try:
                index = item["id"]
                model_result_item = item["output"]
                prompt_entry = {
                    "id": item["id"],
                    "question": item["instruction"],
                    "function": _load_dataset(self.dataset_name).filter(lambda x: x["id"] == item["id"])[0]["function"],
                }
                possible_answer_item = possible_answers_dict.get(index, [])

                eval_result = _evaluate_single_ast_entry(
                    index,
                    model_result_item,
                    possible_answer_item,
                    prompt_entry,
                    model_name,
                    test_category=self.dataset_name.replace("BFCL_v4_", ""),
                    language=Language.PYTHON,
                    return_format=ReturnFormat.PYTHON,
                    has_tool_call_tag="<TOOLCALL>" in model_result_item,
                )
                print(f"Eval result for id {item['id']}: {eval_result}")
                evaluation_results.append(eval_result)
            except Exception as e:
                self.logger.error(f"Error evaluating response for id {item['id']}: {str(e)}")
                traceback.print_exc()
                continue


        total = len(evaluation_results)
        correct = sum(1 for res in evaluation_results if res["valid"])
        accuracy = correct / total if total > 0 else 0.0
        print(f"Evaluation completed: {correct}/{total} correct. Accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy}
        
        

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete Alpaca benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        self.logger.info("Starting Alpaca benchmark evaluation")
        try:
            generation_results = self.generate_responses(model)

            # If not rank 0, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)
            evaluation_results.update(
                {"benchmark_version": "alpaca_eval", "temperature": self.temperature, "max_tokens": self.max_tokens}
            )
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
