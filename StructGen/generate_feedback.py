import argparse
import copy
import threading
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

from human_eval.evaluation import evaluate_functional_correctness
from utils import *
import ast
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def get_configuration_parameters(model_name, uml_type, dataset_name):
    """
    Retrieve parameters such as model name, dataset name, output file path, and PlantUML file path from the configuration file.

    Returns:
        tuple: A tuple containing the following elements:
            - model_name (str): The name of the model.
            - dataset_name (str): The name of the dataset.
            - output_file (str): The path to the output JSONL file.
            - result_file_path (str): The path to the final CSV result file.
            - mode (str): The running mode, which can be either "function" or "class".
    """
    config = get_config()

    output_file = config.get("basic", "output_file_jsonl").format(
        dataset_name=dataset_name,
        model_name=model_name,
        uml_type=uml_type,
        design_repair_num=config.get("feedback", "max_plantUML_attempts"),
        code_repair_num=int(config.get("feedback", "max_function_code_attempts"))
    )
    result_file_path = config.get("UML", "uml_csv_file").format(uml_type=uml_type,
                                                                dataset_name=dataset_name)
    return output_file, result_file_path


def generate_uml(client, problem, uml_type, model_name, config, additional_text=False):
    """
    Generate PlantUML diagram.
    Set maximum number of attempts in advance.
    Verify if generation is successful from problem["plantUML"].

    :param problem: Problem dictionary.
    :param model_name: Name of the designer model.
    :param config: PlantUML generation configuration.
    :param additional_text: Whether to include error messages in prompt.
    """
    plantUML_attempts = 0
    max_plantuml_attempts = config.getint("feedback", "max_plantUML_attempts")  # maximum number of UML generation attempts

    while plantUML_attempts < max_plantuml_attempts:
        plantUML_attempts += 1
        error_additional_text = problem["error_additional_text"] if additional_text and problem[
            "error_additional_text"] else ""
        result_uml = generate_completion(client, problem, uml_type, model_name, generate_type="generate_uml",
                                         additional_text=error_additional_text)
        result_uml, extract_uml_status = extract_uml(result_uml, uml_type)
        if extract_uml_status:  # success
            problem[uml_type] = result_uml  # update problem's PlantUML
            break
        else:
            problem[uml_type] = problem[uml_type] if additional_text else f"extract {uml_type} failed"


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


def test_generated_test_cases(test_cases_input, code_str):
    """
    Use assert statements to test multiple test cases for the similar_elements function and output detailed error messages upon failure.

    Parameters:
    - test_cases_input (str or list of str): A list containing test cases, each typically in the form of 'expr1 == expr2' or other evaluable expressions.
    - code_str (str): A string defining the similar_elements function.

    Returns:
    - "passed" if all test cases pass.
    - failed_tests (list of tuples): A list of tuples containing failed test cases and their corresponding error messages.
    """
    # Parse test_cases_input into a list
    if isinstance(test_cases_input, str):
        try:
            # Safely parse string to list using ast.literal_eval
            test_cases = ast.literal_eval(test_cases_input)
            if not isinstance(test_cases, list):
                raise ValueError("test_cases_input The string must represent a list.")
        except Exception as e:
            return "failed"
    elif isinstance(test_cases_input, list):
        test_cases = test_cases_input
    else:
        return "failed"

    failed_tests = []
    namespace = {}

    try:
        # Execute the code string at once, defining all necessary functions and variables.
        exec(code_str, namespace)
    except Exception as e:
        return f"failed: An error occurred while executing code_str: {e}"

    for i in range(len(test_cases) - 1, -1, -1):
        test_case = test_cases[i].strip()
        # Check if 'test_case' contains 'assert'.
        if 'assert' in test_case:
            # Find the position of the last 'assert' and retain the subsequent part.
            last_assert_index = test_case.rfind('assert')
            test_case = test_case[last_assert_index + len('assert'):].strip()
            test_cases[i] = test_case

        # Prepend an 'assert' to it.
        full_statement = f"{code_str}\nassert {test_case}"

        try:
            # TODO Multi-threading has default timeout setting
            # with swallow_io():
            #     with time_limit(float(3.0)):
                    exec(full_statement, namespace)
        except TimeoutException:
            failed_tests.append(f"Test {test_case} Failed: timed out")
        except BaseException as e:
            failed_tests.append(f"Test {test_case} Failed: {e}")

    return "passed" if not failed_tests else failed_tests



def generate_and_validate_function_code(client, problem, uml_type, model_name, config, additional_text=False):
    """
    Generate function code.
    Set maximum number of attempts in advance.
    Verify if generation is successful from problem["plantUML"].

    :param problem: Problem dictionary.
    :param model_name: Name of the designer model.
    :param config: Function code generation configuration.
    :param additional_text: Prompt error message.
    """
    function_code_attempts = 0
    max_function_code_attempts = config.getint("feedback", "max_function_code_attempts")  # maximum number of function code generation attempts
    if "error_additional_text" not in problem:
        problem["error_additional_text"] = None
    if "passed" not in problem:
        problem["passed"] = "False"

    while function_code_attempts <= max_function_code_attempts:
        function_code_attempts += 1
        error_additional_text = problem["error_additional_text"] if additional_text and problem[
            "error_additional_text"] else ""
        result_function_code = generate_completion(client, problem, uml_type,  model_name, generate_type="generate_function_code",
                                                   additional_text=error_additional_text)
        result_function_code, extract_function_code_status = extract_function(problem, result_function_code)
        result_function_code = add_missing_imports(result_function_code)
        test_cases_copy = copy.deepcopy(problem["generate_test_cases"])
        test_result = test_generated_test_cases(test_cases_copy, result_function_code)
        problem["completion"] = result_function_code

        if test_result == "passed":  # success
            problem["passed"] = "True"
            break
        else:
            additional_text = True
            problem["error_additional_text"] = concatenate_test_results(test_result)
            problem["passed"] = "False"


def filter_problems(problems, condition_key, condition_value):
    """
    Filter problem set, keeping only items that meet specific conditions and have task_id less than maximum task ID.

    :param problems: Problem set dictionary.
    :param condition_key: Key to match.
    :param condition_value: Value to match.
    :return: Filtered problem set dictionary.
    """
    config = get_config()
    max_task_id = config.getint("test", "temp_num")
    filtered = {
        task_id: details for task_id, details in problems.items()
        if (
                   condition_key not in details or  # key does not exist
                   not details.get(condition_key) or  # key exists but value is empty
                   "".join(details.get(condition_key)).strip() == condition_value

           ) and int(re.search(r'\d+', str(task_id)).group()) < max_task_id
    }
    return filtered


def process_single_thread(error_problems, handler_func, client, designer_model, config, additional_text,
                          result_file_path, problems, condition_key):
    """
    Single-thread task processing

    Args:
        error_problems: Dictionary of error problems to be processed
        handler_func: Handler function
        client: Client instance
        designer_model: Design model
        config: Configuration object
        additional_text: Additional text information
        result_file_path: Path to save results
        problems: Original problem dictionary
        condition_key: Condition key name
    """
    total_tasks = len(error_problems)
    processed_tasks = 0
    save_threshold = total_tasks // 5  # Save every 20% of tasks

    for task_id in tqdm(error_problems, desc=f"Processing {condition_key} related errors"):
        try:
            handler_func(client, error_problems[task_id], designer_model, config, additional_text)
            processed_tasks += 1

            # Save results when progress reaches threshold
            if processed_tasks >= save_threshold:
                save_results(merge_dict_values(problems, error_problems), result_file_path)
                processed_tasks = 0  # Reset counter
        except Exception as e:
            print(f"Error occurred while processing task {task_id}: {str(e)}")
    # Ensure final results are saved
    final_result = merge_dict_values(problems, error_problems)
    problems = final_result
    save_results(problems, result_file_path)


def timeout_decorator(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError('Function call timed out')]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(f'Function {func.__name__} timed out after {seconds} seconds')
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]

        return wrapper

    return decorator


def process_multi_thread(error_problems, handler_func, uml_type, client, designer_model, additional_text,
                         result_file_path, problems, condition_key):
    config = get_config()
    total_tasks = len(error_problems)
    task_ids = list(error_problems.keys())
    save_threshold = total_tasks // 10  # Save every 10% of tasks
    completed_tasks = 0
    multi_thread_timeout = config.getint("LLM", "multi_thread_timeout")
    multi_total_timeout = min(total_tasks * multi_thread_timeout, 3600)  # 1 hour timeout

    @timeout_decorator(multi_thread_timeout)
    def safe_handler(task_id):
        try:
            handler_func(client, error_problems[task_id], uml_type, designer_model, config, additional_text)
            return task_id, True
        except Exception as e:
            print(f"Error occurred while processing task {task_id}: {str(e)}")
            return task_id, False

    max_workers = config.getint("LLM", "max_workers")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(safe_handler, task_id): task_id
            for task_id in task_ids
        }

        # Process completed futures with timeout
        with tqdm(total=total_tasks, desc=f"Processing {condition_key} related errors") as progress_bar:
            try:
                for future in as_completed(futures, timeout=multi_total_timeout):
                    try:
                        task_id = futures[future]
                        success = False

                        try:
                            _, success = future.result(timeout=multi_thread_timeout)
                        except TimeoutError:
                            print(f"Task {task_id} result retrieval timed out")
                        except Exception as e:
                            print(f"Task {task_id} execution failed: {str(e)}")

                        if success:
                            completed_tasks += 1

                        progress_bar.update(1)

                        # Save intermediate results
                        if completed_tasks >= save_threshold:
                            save_results(merge_dict_values(problems, error_problems), result_file_path)
                            completed_tasks = 0

                    except Exception as e:
                        print(f"Error occurred while processing future: {str(e)}")

            except TimeoutError:
                print("Overall processing timed out, saving completed results...")

            # Cancel any remaining futures
            for future in futures:
                future.cancel()

    # Save final results
    final_result = merge_dict_values(problems, error_problems)
    problems = final_result
    save_results(problems, result_file_path)


def process_tasks_with_fallback(args):
    error_problems, handler_func, uml_type, client, model_name, additional_text, result_file_path, problems, condition_key = args
    try:
        print("Processing tasks using multi-threading mode...")
        process_multi_thread(error_problems, handler_func, uml_type, client, model_name,
                             additional_text, result_file_path, problems, condition_key)
    except Exception as e:
        print(f"Multi-threading processing failed (Error: {str(e)}), switching to single-thread mode...")
        process_single_thread(error_problems, handler_func, uml_type, client, model_name,
                              additional_text, result_file_path, problems, condition_key)


def handle_error_cases(client, problems, condition_key, condition_value,
                       handler_func, handler_args, additional_text=False,
                       rerun=False):
    """
    Handle specific types of error cases, execute the specified handler function for filtered problems, and update the problem dictionary.

    :param problems: Original problem dictionary
    :param condition_key: Condition key used for filtering error cases
    :param condition_value: Error value matching the condition key
    :param handler_func: Function to handle specific problems
    :param additional_text: Whether to include error messages
    :param rerun: Whether to regenerate everything
    :return: Updated problem dictionary
    """
    model_name, uml_type, dataset_name, output_file, result_file_path = handler_args

    if condition_key not in next(iter(problems.values()), {}):
        rerun = True

    error_problems = problems if rerun else filter_problems(problems, condition_key, condition_value)

    args = (error_problems, handler_func, uml_type, client, model_name, additional_text, result_file_path, problems, condition_key)
    process_tasks_with_fallback(args)

    return problems


def process_eval(problems, handler_args):
    """
    Loop to ensure each prompt has public tests and plantuml
    Function code generation starts, if function code fails three times (extraction failure and test failure), then modify plantUML

    :param args: args
    :param problems: Problem set.
    """
    config = get_config()
    model_name, uml_type, dataset_name, output_file, result_file_path = handler_args
    client = get_client(config)

    while len(filter_problems(problems, uml_type, f"extract {uml_type} failed")) > 1:
        problems = handle_error_cases(client=client, 
                                    problems=problems, 
                                    condition_key=uml_type, 
                                    condition_value=f"extract {uml_type} failed",
                                    handler_func=generate_uml, 
                                    handler_args=handler_args)

    # Function code generation process, needs to add error feedback to modify function code and UML
    feedback_uml_attempts = 0
    max_feedback_uml_attempts = config.getint("feedback", "max_plantUML_attempts")
    while feedback_uml_attempts <= max_feedback_uml_attempts:
        print("Generating and testing code...")
        problems = handle_error_cases(client=client,
                                    problems=problems,
                                    condition_key="passed",
                                    condition_value="False",
                                    handler_func=generate_and_validate_function_code,
                                    handler_args=handler_args,
                                    additional_text=True)
        # Save results
        sample = convert_data_format(problems, uml_type)
        write_jsonl(output_file, sample)

        evaluate_functional_correctness(sample_file=output_file, n_workers=10,
                                        problem_file=config.get("basic", "problem_file").format(dataset_name=dataset_name))
        evaluated_dict = transform_json_file_to_dict(f"{output_file}_results.jsonl")
        error_num = 0
        for task_id, item_dict in evaluated_dict.items():
            if not item_dict["passed"]:
                if problems[task_id]["passed"] == "True":
                    problems[task_id]["error_additional_text"] = f"Please improve code robustness by considering multiple scenarios"
                problems[task_id]["passed"] = "False"
                error_num += 1
            else:
                problems[task_id]["passed"] = "True"
        print(f"====pass@1:{(len(problems) - error_num) / len(problems):.7f}====\n")

        if feedback_uml_attempts == max_feedback_uml_attempts:
            save_results(problems, result_file_path)
            break

        print("Starting plantUML repair...")
        problems = handle_error_cases(client=client, 
                                    problems=problems, 
                                    condition_key="passed", 
                                    condition_value="False",
                                    handler_func=generate_uml, 
                                    handler_args=handler_args, 
                                    additional_text=True)

        feedback_uml_attempts += 1

    print("Successfully save the result to file")


def extract_function_function_name_params(code):
    # Match patterns like 'def function_name(param1, param2):'
    match = re.search(r'def\s+(\w+)\s*\((.*?)\)\s*:', code)
    if match:
        func_name = match.group(1)
        func_params = match.group(2)
        return func_name, func_params
    else:
        return None, None


def parse_arguments():
    """
    Parse command-line arguments and set default values for each argument.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process some inputs.")

    # Add command-line arguments with default values
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106",
                        help="Name of the model (default: gpt-3.5-turbo-1106)")
    parser.add_argument("--uml_type", type=str, default="plantuml", help="Type of UML (default: plantuml)")
    parser.add_argument("--dataset_name", type=str, default="human-eval", help="Name of the dataset (default: human-eval)")

    # Parse the command-line arguments
    args = parser.parse_args()
    return args


def main():
    # 解析命令行参数
    args = parse_arguments()

    # 使用解析后的参数
    model_name = args.model_name
    uml_type = args.uml_type
    dataset_name = args.dataset_name

    output_file, result_file_path = get_configuration_parameters(model_name, uml_type, dataset_name)
    result_dict = load_result_csv_2_dict(result_file_path)

    for task_id, item_dict in result_dict.items():
        if "generate_test_cases" in item_dict and item_dict["generate_test_cases"]:
            item_dict["generate_test_cases"] = json.loads(item_dict["generate_test_cases"])
    dataset_dict = transform_json_file_to_dict(get_config().get("basic", "problem_file").format(dataset_name=dataset_name))
    problems = merge_dict_values(dataset_dict, result_dict)

    print(f"\n===============================\n"
          f"model_name:{model_name}\n"
          f"output_file:{output_file}\n"
          f"===============================")
    args = (model_name, uml_type, dataset_name, output_file, result_file_path)
    process_eval(problems, args)


if __name__ == '__main__':
    main()
