import ast
import csv
import gzip
from typing import Iterable, Dict, Any
from openai import OpenAI
from tqdm import tqdm
import configparser
import json
from typing import Optional, Callable, Dict
import contextlib
import os
import signal
import io
import re
import sys
import math
import cmath
import heapq as hq
from collections import Counter, defaultdict, ChainMap as ct
from math import tan, pi
import regex
from itertools import tee
from FeidaChat import FeidaChat

#=================================================Read File Content============================================================
def get_config():
    """
    Get configuration file
    :return: config configuration
    """
    config = configparser.ConfigParser()
    config.read("./generate/config.ini")
    return config


def get_client(config):
    """
    Get OpenAI client
    :param config: Configuration options
    :return: OpenAI client instance
    """
    base_url = config.get("LLM", "base_url")
    api_key = config.get("LLM", "api_key")
    return OpenAI(base_url=base_url, api_key=api_key)


def read_md_file(file_path):
    """
    Read content from a Markdown file and return the result.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        str: String content of the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            result = file.read()
        return result
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
#=================================================Read File Content End============================================================


#=================================================Content Output===============================================================
def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    # Expand user path (e.g., ~/documents -> /home/user/documents)
    filename = os.path.expanduser(filename)

    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory:  # Only create directory if path is not in current directory
        os.makedirs(directory, exist_ok=True)

    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))


#=================================================Content Output End===============================================================



#=================================================Format Conversion===============================================================
def read_csv_to_dict(file_path):
    """
    Reads a CSV file and dynamically processes it into a set, dictionary, or nested dictionary
    based on the number of columns.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        - set: If the CSV has one column, returns a set of values.
        - dict: If the CSV has two columns, returns a dictionary where the first column is the key and the second column is the value.
        - dict of dict: If the CSV has more than two columns, returns a dictionary where the first column is the key
                        and the value is a dictionary of the remaining columns.
    """
    csv.field_size_limit(500 * 1024 * 1024)
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames  # Get the column headers

        if not headers:
            raise ValueError("CSV file is empty or has no headers.")

        if len(headers) == 1:
            # Single column: Return a set of values
            return {row[headers[0]] for row in reader}

        elif len(headers) == 2:
            # Two columns: Return a key-value dictionary
            return {row[headers[0]]: row[headers[1]] for row in reader}

        else:
            # More than two columns: Return a nested dictionary
            return {
                row[headers[0]]: {key: row[key] for key in headers[1:]} for row in reader
            }


def read_json_to_dict(file_path):
    """
    Read a JSON file that contains a list of dictionaries and convert it into
    a dictionary with a unique field (e.g., task_id) as the key.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary representation where the key is a unique field like task_id.
    """
    with open(file_path, 'r') as f:
        json_data = json.load(f)

    return json_data


def transform_mbpp_2_humaneval(obj1):
    """
    Transform MBPP object to HumanEval format.

    :param obj1: dict - Input object in MBPP format
    :return: dict - Transformed object in HumanEval format
    """
    import re

    # Extract function definition from code, including function name and parameters
    def extract_function_signature(code):
        # Match patterns like 'def function_name(param1, param2):'
        match = re.search(r'def\s+(\w+)\s*\((.*?)\)\s*:', code)
        if match:
            func_name = match.group(1)
            func_params = match.group(2)
            return func_name, func_params
        else:
            return None, None

    # Build test function in HumanEval format
    def build_test_function(test_list, entry_point):
        test_code = f"def check(candidate):\n\n"
        for test in test_list:
            # Replace all occurrences of function name with 'candidate' in test cases
            test_replaced = test.replace(entry_point, 'candidate')
            test_code += f"    {test_replaced}\n"
        return test_code

    # Format prompt information
    def format_prompt(text, test_list, entry_point, func_params):
        # Append test cases to problem description
        prompt = f"def {entry_point}({func_params}):\n\"\"\"\n{text}\n"
        if test_list[0] != "test test cases failed" and len(test_list[0]) < 100:
            prompt += "Your code should satisfy these tests:\n"
            for test in test_list:
                prompt += (f"{test.replace(entry_point, entry_point).replace('assert', '')}\n")
        prompt += "\"\"\"\n"
        return prompt

    # Extract function name and parameters
    func_name, func_params = extract_function_signature(obj1["code"])
    if not func_name or not func_params:
        raise ValueError("Cannot extract function definition from code, please check input format.")

    # Transform object
    obj2 = {
        "task_id": str(obj1["task_id"]),
        "prompt": format_prompt(obj1["text"], obj1["public_test"], func_name, func_params),
        "canonical_solution": obj1["code"],
        "entry_point": func_name,
        "test": build_test_function(obj1["test_list"], func_name)
    }

    return obj2


def merge_dict_values(dict1, dict2):
    """
    Merge values of two dictionaries. If keys are the same, expand values to the same level,
    with later values taking precedence for the same key.
    :param dict1: First dictionary.
    :param dict2: Second dictionary.
    :return: Merged dictionary.
    """
    merged_dict = {}

    # Iterate through all keys of both dictionaries to ensure merged dictionary contains all keys
    all_keys = set(dict1.keys()).union(set(dict2.keys()))

    for key in all_keys:
        # Merge values: Get values from dict1 and dict2, expand to one level
        merged_value = {}
        if key in dict1:
            merged_value.update(dict1[key])  # Add values from dict1
        if key in dict2:
            merged_value.update(dict2[key])  # Add values from dict2, overwriting same keys

        merged_dict[key] = merged_value

    return merged_dict


def load_result_csv_2_dict(result_path):
    """
    Load or initialize CSV based on path.
    :param result_path: Path to CSV file.
    :return: Dictionary.
    """
    if not os.path.exists(result_path):
        return defaultdict(dict)
    else:
        return read_csv_to_dict(result_path)


def transform_json_file_to_dict(file_path):
    """
    Reads a JSON file with multiple JSON objects (separated by newlines)
    and maps 'task_id' to the corresponding object.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary with 'task_id' as the key and the object as the value.
    """
    result = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    obj = json.loads(line)  # Parse each line as JSON
                    if "task_id" in obj:
                        result[str(obj["task_id"])] = obj
                    else:
                        print("Skipping object without 'task_id':", obj)
        return result
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")
        return None

def convert_mbpp_2_humaneval_dict(file_path, result_file_path):
    """
    Read MBPP data and generate problem dictionary.
    :param file_path: Path to JSON file.
    :param result_file_path: Path to result CSV file.
    :return: Two dictionaries: mbpp_data and problems.
    """
    result_dict = load_result_csv_2_dict(result_file_path)
    mbpp_data = transform_json_file_to_dict(file_path)

    problems = {}
    for task_id,item_dict in mbpp_data.items():
        task_id = str(task_id)
        item_dict["task_id"] = task_id
        if result_dict[task_id]["generate_test_cases"]:
            test_cases_list = json.loads(result_dict[task_id]["generate_test_cases"])
        else:
            test_cases_list = ["test test cases failed"]
        item_dict["public_test"] = test_cases_list
        problems[task_id] = transform_mbpp_2_humaneval(item_dict)
        result_dict[task_id]["generate_test_cases"] = test_cases_list
    problems = merge_dict_values(result_dict, problems)
    return problems


def dict_to_csv(dict_list: dict, csv_output_path: str):
    """
    Converts a dictionary to a CSV file, where each entry in the dictionary has
    a unique identifier (like HumanEval/0) and corresponding values.

    Parameters:
        dict_list (dict): A dictionary where keys are unique IDs, and values are dictionaries
                          with keys like 'plantUML' and 'Prompt'.
        csv_output_path (str): Path to the output CSV file.

    Returns:
        None
    """
    # Step 1: Prepare data rows by adding the main key as 'Task_id'
    data_rows = []
    for task_id, content in dict_list.items():
        # Create a shallow copy of the content to avoid modifying the original
        row = {"task_id": task_id}
        row.update(content)  # Merge the inner dictionary into the row
        if "generate_test_cases" in content:
            # Use json.dumps for the CSV value but do not modify the original dictionary
            row["generate_test_cases"] = json.dumps(content["generate_test_cases"])
        data_rows.append(row)

    # Step 2: Collect all unique headers from the rows, ensuring 'Task_id' is first
    headers = ['task_id']  # Always keep 'Task_id' as the first column
    for row in data_rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)

    # Step 3: Write the data to a CSV file
    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()  # Write the header row
        for row in data_rows:
            # Ensure missing keys are filled with empty strings
            row_with_defaults = {header: row.get(header, '') for header in headers}
            writer.writerow(row_with_defaults)


def construct_prompt(user_prompt):
    """
    Construct prompt format
    :param user_prompt: Prompt content
    :return: Constructed chat template
    """
    messages = [
        {
            "role": "system",
            "content": f"You are a professional python programmer."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    return messages

#=================================================Format Conversion End===============================================================


#=================================================Model Content Generation===============================================================
def generate_completion(client, problem, uml_type, model_name, generate_type, additional_text=""):
    """
    Process Completion and extract results (function code or PlantUML).

    :param problem: Requirement object.
    :param generate_type: Type of content to generate.
    :param model_name: Model to use for generation.
    :param additional_text: Additional text to append to user_prompt.
    :return: Dictionary of extracted results (only when mode is "function").
    """
    config = get_config()
    prompt_file_path = config.get("prompt", generate_type).format(uml_type=uml_type)

    if generate_type == "generate_uml":
        user_prompt = read_md_file(prompt_file_path).format(
            dataset_prompt=problem["prompt"],
        )
    elif generate_type == "generate_function_code":
        user_prompt = read_md_file(prompt_file_path).format(
            dataset_prompt=problem["prompt"],
            uml_type="dot" if uml_type == "graphviz" else uml_type,
            uml_content=problem[uml_type]
    )
    else:
        raise ValueError("Invalid generate_type. Must be 'generate_function' or 'generate_plantUML'.")

    user_prompt += f"\n{additional_text}"
    prompt = construct_prompt(user_prompt)
    result = generate_one_completion(client, prompt, config, model_name)

    return result


#=================================================Model Content Generation===============================================================


#=================================================Content Processing============================================================
def generate_one_completion(client, messages, config, model_name):
    """
    Generate one completion
    :param client: OpenAI client for making requests
    :param messages: Constructed prompt
    :param config: Configuration settings
    :param model_name: Model name
    :return: Large model output
    """

    try:
        if model_name.startswith("gpt"):
            # Initialize client
            api_key = config.get("LLM", "api_key")
            base_url = config.get("LLM", "base_url")
            client = FeidaChat(api_key, base_url)

            response = client.generate(messages)
            if response["success"]:
                return response["text"]
            else:
                return response["error"]
        else:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=config.getfloat("LLM", "temperature"),
                top_p=config.getfloat("LLM", "top_p"),
                max_tokens=config.getint("LLM", "max_tokens"),
            )
            return completion.choices[0].message.content
    except Exception as e:
        # Log error (import logging module if needed)
        print(f"Error generating completion: {e}")
        # Return None or a default value based on requirements
        return f"Error generating completion: {e}"


def extract_function_with_python(text):
    """
    Extract import statements and function definition parts from text (including docstrings, but excluding test code).

    Parameters:
        text (str): Text containing complete code.

    Returns:
        str: Extracted import statements and function definition parts.
    """
    # Split text into lines for line-by-line processing
    lines = text.splitlines()
    result = []
    inside_function = False  # Flag indicating if we're in function definition part

    for line in lines:
        # If it's an import statement, keep it directly
        if line.strip().startswith("import") or line.strip().startswith("from"):
            result.append(line)
        # If it's a function definition, start function extraction mode
        elif line.strip().startswith("def "):
            result.append(line)
            inside_function = True
        # If in function part, keep current line
        elif inside_function:
            # Determine function end by indentation
            if line.strip() == "" or line.startswith(" "):  # Keep empty lines and indented lines
                result.append(line)
            else:
                break  # End function part when encountering non-indented line

    return "\n".join(result)


def extract_function_with_java(text):
    """
    Extract import statements and function definition parts from text (excluding test code).
    Supports Python, JavaScript, Java, and C++.

    Parameters:
        text (str): Text containing complete code.

    Returns:
        str: Extracted import statements and function definition parts.
    """
    lines = text.splitlines()
    result = []
    inside_function = False
    brace_count = 0  # 跟踪花括号配对

    for line in lines:
        stripped_line = line.strip()

        # 处理导入语句
        if (stripped_line.startswith("import") or
                stripped_line.startswith("from") or
                stripped_line.startswith("#include") or
                stripped_line.startswith("using")):
            result.append(line)
            continue

        # 检测函数开始
        if (stripped_line.startswith("def ") or  # Python
                stripped_line.startswith("function ") or  # JavaScript
                "function" in stripped_line or  # JavaScript 箭头函数和函数表达式
                stripped_line.startswith("const ") or  # JavaScript const 声明
                stripped_line.startswith("let ") or  # JavaScript let 声明
                stripped_line.startswith("var ") or  # JavaScript var 声明
                (("public " in stripped_line or "private " in stripped_line or
                  "protected " in stripped_line or stripped_line.startswith("class ")) and
                 "(" in stripped_line)  # Java/C++ 方法
        ):
            result.append(line)
            inside_function = True
            brace_count += stripped_line.count("{")
            brace_count -= stripped_line.count("}")
            continue

        if inside_function:
            result.append(line)
            brace_count += stripped_line.count("{")
            brace_count -= stripped_line.count("}")

            # 检查函数是否结束
            if brace_count == 0:
                if (stripped_line.endswith("}") or  # JavaScript/Java/C++
                        not line.startswith(" ")):  # Python
                    inside_function = False

            # 检测测试用例开始
            if ("console.log(" in stripped_line or  # JavaScript
                    "System.out.println" in stripped_line or  # Java
                    "cout <<" in stripped_line or  # C++
                    stripped_line.startswith("print(")  # Python
            ):
                break

    # 移除末尾空行
    while result and not result[-1].strip():
        result.pop()

    return "\n".join(result)


def extract_function_with_cpp(text):
    """
    Extract import statements and function definition parts from a string of code.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    if not text or text.isspace():
        return ""

    # Clean up input
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Handle escaped characters
    text = text.replace('\\n', '\n')
    text = text.replace('\\"', '"')
    text = text.replace('""', '"')

    lines = text.splitlines()
    result = []
    inside_function = False
    brace_count = 0
    current_includes = []
    current_function = []

    cpp_types = [
        "bool", "void", "int", "float", "double", "string",
        "char", "vector<", "auto", "long", "unsigned", "short",
        "const", "static", "class", "struct", "template",
        "std::", "boost::"
    ]

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            if inside_function:
                current_function.append(line)
            elif current_includes:
                current_includes.append(line)
            continue

        # Skip test code
        if ("main(" in stripped_line or
                "cout" in stripped_line or
                "test" in stripped_line.lower() or
                "assert" in stripped_line):
            continue

        # Handle includes and using statements
        if (stripped_line.startswith("#include") or
                stripped_line.startswith("using")):
            current_includes.append(line)
            continue

        # Detect function start
        is_function_start = False
        if not inside_function:
            words = stripped_line.split()
            if words:
                is_function_start = (
                        any(first_word.startswith(type_name) for type_name in cpp_types for first_word in words[:2]) and
                        "(" in stripped_line and
                        ";" not in stripped_line and  # Avoid function declarations
                        "{" in stripped_line  # Must have opening brace
                )

        if is_function_start:
            if current_includes:
                result.extend(current_includes)
                result.append("")
                current_includes = []

            current_function.append(line)
            inside_function = True
            brace_count = stripped_line.count("{")
            brace_count -= stripped_line.count("}")
            continue

        # Inside function handling
        if inside_function:
            current_function.append(line)
            brace_count += stripped_line.count("{")
            brace_count -= stripped_line.count("}")

            if brace_count == 0:
                inside_function = False
                result.extend(current_function)
                current_function = []
                result.append("")

    # Add any remaining functions
    if current_function:
        result.extend(current_function)

    # Remove trailing empty lines
    while result and not result[-1].strip():
        result.pop()

    return "\n".join(result)


def add_missing_imports(code: str) -> str:
    """
    Checks for missing import statements in the given code and adds them if not present.

    Parameters:
        code (str): The Python code to check and update.

    Returns:
        str: The updated Python code with missing imports added.
    """
    # Define the default list of required imports
    required_imports = ["import math", "from typing import List"]

    # Split the code into lines
    lines = code.splitlines()

    # Collect existing imports in the code
    existing_imports = {line.strip() for line in lines if line.strip().startswith("import") or line.strip().startswith("from")}

    # Identify missing imports
    missing_imports = [imp for imp in required_imports if imp not in existing_imports]

    # Add missing imports to the top of the code
    if missing_imports:
        # Find the first non-empty line to insert imports after (handle shebangs or docstrings)
        insertion_index = 0
        for i, line in enumerate(lines):
            if line.strip():  # Non-empty line
                if line.startswith("#!") or line.startswith('"""') or line.startswith("'''"):
                    # Skip shebang or multi-line docstring
                    continue
                insertion_index = i
                break

        # Insert the missing imports
        updated_code = lines[:insertion_index] + missing_imports + [""] + lines[insertion_index:]
        return "\n".join(updated_code)

    # Return original code if no imports were missing
    return code


def extract_function_code(problem: dict, text: str, prompt_language: str) -> str:
    """
    Extract Python function definition part (including function comments and code) from given text.

    Parameters:
        text (str): Text containing complete content.
        prompt_language (str): Language of the prompt.

    Returns:
        str: Extracted function code part, returns error message if failed.
    """
    # Regular expression match for function definition part (including comments)
    function_pattern = rf"```{prompt_language}\s+([\s\S]*?)```"

    # Find all code blocks
    code_blocks = re.findall(function_pattern, text, re.MULTILINE)
    problem["response"] = code_blocks

    if not code_blocks:
        return "# extract function code failed"

    # 创建函数映射字典
    function_map = {
        'java': extract_function_with_java,
        'javascript': extract_function_with_java,
        'cpp': extract_function_with_cpp,
        'python': extract_function_with_python
    }

    # Extract first code block as function code
    function_code = function_map[prompt_language](code_blocks[0])
    if not function_code:
        return "# extract function code failed"

    # Clean function code
    return function_code


def extract_function(problem, completion, prompt_language="python"):
    """
    Generate and extract function code
    :param completion: Model output
    :param prompt_language: Language of the prompt
    :return: (Extracted code or failure message, extracted test cases, status)
    """

    # **Step 1: Extract function code and test cases**
    extracted_function_code = extract_function_code(problem, completion, prompt_language)

    if extracted_function_code == "# extract function code failed":
        # Extraction failed, return failure status and message
        return extracted_function_code, False
    # Code extraction success is sufficient, test cases are just a bonus

    return extracted_function_code, True


def extract_uml(response, uml_type):
    """
    Extract PlantUML code from a string.

    :param response: The input string containing PlantUML code.
    :return: A tuple (extracted_plantUML, success).
             - extracted_plantUML: Extracted PlantUML code or an error message.
             - success: Boolean indicating whether extraction was successful.
    """
    if uml_type == "graphviz":
        uml_type_extract = "dot"
    else:
        uml_type_extract = uml_type
    if not isinstance(response, str):
        return "Input must be a string", False

    try:
        matches = re.findall(rf"```{uml_type_extract}(.*?)```", response, re.DOTALL)
        if matches:
            return matches[0].strip(), True
        else:
            return f"extract {uml_type} failed", False
    except Exception as e:
        return f"Error during extraction: {str(e)}", False


#=================================================Content Processing============================================================

def rename_2_entry_point(code: str, new_name: str) -> str:
    """
    Renames the first function that does not call any other function, or if none are found,
    renames the function with the most lines of code. If no functions exist, return the original code.

    Parameters:
        code (str): The source code as a string.
        new_name (str): The new name to assign to the selected function.

    Returns:
        str: The modified source code with the selected function renamed, or the original code if no functions exist.
    """
    # Split the code into lines
    lines = code.strip().split('\n')

    # Extract all function definitions and their content
    functions = []
    current_func = None

    for line in lines:
        line = line.strip()
        if line.startswith("def ") and line.endswith(":"):
            # Start a new function definition
            func_name = line.split("(")[0][4:]
            if current_func:
                # Append the previous function to the list
                functions.append(current_func)
            current_func = {"name": func_name, "body": [], "start_line": line}
        elif current_func is not None:
            # Append lines to the current function's body
            current_func["body"].append(line)

    # Append the last function
    if current_func:
        functions.append(current_func)

    # If no functions are found, return the original code with a message
    if not functions:
        print(f"No functions found in the provided code. new_name is {new_name}")
        return code

    # Find the function that does not call any other function
    function_names = {func["name"] for func in functions}
    for func in functions:
        # Check if the function calls other functions
        if not any(other_func in "\n".join(func["body"]) for other_func in function_names):
            # Rename this function
            renamed_code = code.replace(f"def {func['name']}(", f"def {new_name}(")
            return renamed_code

    # If no unused function is found, rename the function with the most lines of code
    largest_func = max(functions, key=lambda func: len(func["body"]))
    renamed_code = code.replace(f"def {largest_func['name']}(", f"def {new_name}(")
    return renamed_code


def log_tqdm_progress_to_file(tqdm_bar, file_path="progress_log.txt"):
    """
    Logs the progress of a tqdm object to a specified text file.

    Parameters:
        tqdm_bar (tqdm): The tqdm progress bar object.
        file_path (str): The path of the file where the progress will be logged.
    """
    with open(file_path, "a") as file:
        file.write(f"Progress: {tqdm_bar.n}/{tqdm_bar.total}\n")

# =====================================Limit Code Execution Time, Prevent Infinite Loops Start====================================================

class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


# =====================================Limit Code Execution Time, Prevent Infinite Loops End====================================================


def extract_testcases_from_prompt(prompt):
    """
    Extract test cases from prompt text
    Extract text in the form of "function_call(...) -> expected_output",
    and convert it to the format "function_call(...) == expected_output"
    """
    patterns = [
        # Pattern 1: "function_call(...) -> expected_output"
        re.compile(r"(?P<func_call>\w+\([^)]*\))\s*->\s*(?P<expected_output>[^.\n]+)"),
        
        # Pattern 2: ">>> function_call(...)\nexpected_output"
        re.compile(r">>>\s*(?P<func_call>\w+\([^)]*\))\s*\n\s*(?P<expected_output>[^.\n]+)")
    ]
    
    all_matches = []
    
    for pattern in patterns:
        for match in pattern.finditer(prompt):
            # Matched function call
            func_call = match.group("func_call").strip()
            # Matched expected output
            expected_out = match.group("expected_output").strip()
            # Full matched text
            full_text = match.group(0).strip()

            # -- Additional filtering: Skip if this line looks like a function definition --
            # 1) Starts with "def "
            # 2) Or ends with ":" (common in Python function definitions with type annotations: `def foo(...) -> int:`)
            if full_text.startswith("def ") or full_text.endswith(":"):
                continue

            all_matches.append((func_call, expected_out))

    # Remove duplicates (optional)
    unique_pairs = list(dict.fromkeys(all_matches))

    # Convert all captured pairs (func_call, expected_out) to "func_call(...) == expected_output"
    results = [f"{fc} == {eo}" for (fc, eo) in unique_pairs]
    return results

def get_return_type(function_str: str) -> type:
    """
    Analyze function string to get return type
    """
    try:
        tree = ast.parse(function_str)
        return_nodes = []

        class ReturnVisitor(ast.NodeVisitor):
            def visit_Return(self, node):
                if node.value is not None:
                    return_nodes.append(node.value)

        ReturnVisitor().visit(tree)

        if return_nodes:
            last_return = return_nodes[-1]

            # String type
            if isinstance(last_return, ast.Str):
                return str
            # Numeric type
            elif isinstance(last_return, (ast.Num, ast.Constant)):
                return type(last_return.n) if hasattr(last_return, 'n') else type(last_return.value)
            # List
            elif isinstance(last_return, ast.List):
                return list
            # Dictionary
            elif isinstance(last_return, ast.Dict):
                return dict
            # Set
            elif isinstance(last_return, ast.Set):
                return set
            # Tuple
            elif isinstance(last_return, ast.Tuple):
                return tuple
            # Boolean
            elif isinstance(last_return, ast.NameConstant):
                if last_return.value in (True, False):
                    return bool
                elif last_return.value is None:
                    return type(None)
            # Formatted string
            elif isinstance(last_return, (ast.BinOp, ast.JoinedStr, ast.FormattedValue)):
                return str

        return Any  # Default return type is Any

    except Exception as e:
        print(f"Error analyzing return type: {e}")
        return Any


def format_value(value: Any, return_type: type = None) -> str:
    """
    Format a value based on its return type
    """
    try:
        # Step 1: Try to safely evaluate the value
        evaluated_value = safe_eval(value)

        # Step 2: Process according to return_type
        if return_type is not None:
            # If already the target type, use directly
            if isinstance(evaluated_value, return_type):
                value = evaluated_value
            # Otherwise try to convert
            else:
                value = convert_to_type(evaluated_value, return_type)
        else:
            # No specified return type, use evaluated value
            value = evaluated_value

        # Step 3: Format output
        if isinstance(value, str):
            # If it's a string but not a numeric string, add quotes
            if not is_numeric_string(value):
                return f"'{value}'"
            return value
        elif isinstance(value, (int, float)):
            # Numeric type, directly convert to string
            return str(value)
        elif isinstance(value, bool):
            # Boolean, convert to lowercase string
            return str(value).lower()
        elif isinstance(value, type(None)):
            # None value
            return 'None'
        elif isinstance(value, (list, tuple, set, dict)):
            # Container type, use str function directly
            return str(value)
        else:
            # Other types, add quotes
            return f"'{str(value)}'"

    except Exception as e:
        # Handle exception
        print(f"Error in format_value: {str(e)}")
        try:
            # Try the most basic string conversion
            return str(value)
        except:
            # If even basic string conversion fails, return empty string
            return ''


def create_global_scope():
    """
    Create global scope containing commonly used modules
    """
    global_scope = {
        # Basic modules
        're': re,
        'sys': sys,
        'io': io,

        # Math related
        'math': math,
        'cmath': cmath,
        'pi': pi,
        'tan': tan,

        # Data structures related
        'heapq': hq,
        'hq': hq,  # Alias support
        'Counter': Counter,
        'defaultdict': defaultdict,
        'ChainMap': ct,
        'ct': ct,  # Alias support

        # Other utility modules
        'regex': regex,
        'tee': tee
    }

    # Add built-in functions
    global_scope.update({
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'sorted': sorted,
        'reversed': reversed,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'all': all,
        'any': any,
        'round': round
    })

    return global_scope


def update_generate_test_cases(item_dict, function_str, function_name):
    """
    Modify test cases in item_dict, replacing incorrect expected outputs with actual calculated outputs
    Args:
        item_dict: Dictionary or list containing test cases
        function_str: String representation of the function
        function_name: Name of the function
    Returns:
        list: Updated list of test cases
    """
    # Create global scope with necessary modules
    global_scope = create_global_scope()
    local_scope = {}

    try:
        # Get function return type
        return_type = get_return_type(function_str)
        # Execute function definition
        exec(function_str, global_scope, local_scope)
        target_function = local_scope[function_name]
        # Add function to global scope
        global_scope[function_name] = target_function

    except Exception as e:
        return [f"Error during function definition or loading: {str(e)}"]

    updated_test_cases = []
    test_cases = item_dict if isinstance(item_dict, list) else []

    for test_case in test_cases:
        if 'assert' in test_case:
            last_assert_index = test_case.rfind('assert')
            test_case = test_case[last_assert_index + len('assert'):].strip()

        if " == " not in test_case:
            continue

        left_part, expected_output = test_case.split(" == ", 1)
        left_part = left_part.strip()

        try:
            # Capture function output
            stdout_backup = sys.stdout
            sys.stdout = io.StringIO()

            # Execute test case in global scope
            exec(f"print({left_part})", global_scope)
            actual_output = sys.stdout.getvalue().strip()

            # Restore standard output
            sys.stdout = stdout_backup

            # Use format_value to process output
            formatted_output = format_value(actual_output, return_type)
            new_test_case = f'{left_part} == {formatted_output}'
            updated_test_cases.append(new_test_case)

        except Exception as e:
            sys.stdout = stdout_backup
            print(f"Error processing test case '{test_case}': {e}")
            # If processing fails, keep original test case
            updated_test_cases.append(test_case)

    return updated_test_cases if updated_test_cases else ["No valid test cases found"]


def save_results(problems, result_file_path):
    """
    Save results to CSV file
    """
    dict_to_csv(problems, result_file_path)
    print("Results have been saved to file\n")


def convert_data_format(data_list, uml_type):
    """
    Convert dictionary elements in the given format data list to target format dictionary
    
    Parameters:
    data_list (list): List containing dictionaries in original format, each dictionary has multiple key-value pairs
    
    Returns:
    list: List containing dictionaries in target format
    """
    result = []
    # .items can separate key and value, otherwise we only get a complete object
    for task_id, data in data_list.items():
        new_data = {
            "task_id": task_id,
            "completion": data.get("completion"),
            f"{uml_type}": data.get(uml_type)
        }
        result.append(new_data)
    return result


def concatenate_test_results(test_result):
    """
    Concatenate elements from the test result list into a complete English description.

    :param test_result: List of test results, where each element is a description of a test case in the format "Expected xxx, but got xxx".
    :return: Complete English description string after concatenation.
    """
    if not isinstance(test_result, list):
        return ""
    result_str = "### Error-prone situations:\n"
    for idx, item in enumerate(test_result):
        error_msg = item
        result_str += f"{idx + 1}. {error_msg}\n"
    return result_str


def filter_problems(problems, config, condition_key, condition_value):
    """
    Filter the problem set, keeping only items that meet specific conditions and have task_id less than the maximum task ID.

    :param problems: Problem set dictionary.
    :param config: Configuration object.
    :param condition_key: Key to match.
    :param condition_value: Value to match.
    :return: Filtered problem set dictionary.
    """
    max_task_id = config.getint("LLM", "max_task_id")

    filtered = {
        task_id: details for task_id, details in problems.items()
        if (
            condition_key not in details or  # key does not exist
            not details.get(condition_key) or  # key exists but value is empty
            "".join(details.get(condition_key)).strip() == condition_value
        ) and int(re.search(r'\d+', str(task_id)).group()) < max_task_id
    }
    return filtered


def safe_eval(value):
    """
    Safely evaluate value, avoiding execution of dangerous code
    """
    try:
        return ast.literal_eval(str(value))
    except:
        return value


def is_numeric_string(s):
    """
    Check if string is numeric
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def convert_to_type(value, target_type):
    """
    Convert value to target type
    """
    try:
        if target_type == bool:
            return bool(value)
        elif target_type == int:
            return int(float(value))
        elif target_type == float:
            return float(value)
        elif target_type == str:
            return str(value)
        elif target_type == list:
            return list(value)
        elif target_type == tuple:
            return tuple(value)
        elif target_type == set:
            return set(value)
        elif target_type == dict:
            return dict(value)
        else:
            return value
    except:
        return value


def extract_uml_code(text: str) -> str:
    """
    Extract PlantUML code from given text.

    Parameters:
        text (str): Text containing complete content.

    Returns:
        str: Extracted PlantUML code, returns error message if failed.
    """
    # Regular expression match for PlantUML code block
    uml_pattern = r"```plantuml\s+([\s\S]*?)```"

    # Find all code blocks
    code_blocks = re.findall(uml_pattern, text, re.MULTILINE)

    if not code_blocks:
        return "# extract uml code failed"

    # Extract first code block as PlantUML code
    uml_code = code_blocks[0].strip()

    return uml_code
