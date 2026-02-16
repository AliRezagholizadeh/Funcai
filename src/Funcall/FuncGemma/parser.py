import re
# from Functions import *
import os
import sys
import importlib.util
import logging
import inspect

def extract_tool_calls(text):
    def cast(v):
        try: return int(v)
        except:
            try: return float(v)
            except: return {'true': True, 'false': False}.get(v.lower(), v.strip("'\""))

    return [{
        "name": name,
        "arguments": {
            k: cast((v1 or v2).strip())
            for k, v1, v2 in re.findall(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args)
        }
    } for name, args in re.findall(r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", text, re.DOTALL)]

def get_module_functions(module_name:str, module_abs_dir:str, logger: logging=None):

    # module_abs_dir = "/path/to/your/Functions.py"
    # module_name = "custom_module"
    functions = {}
    # get spec
    spec = importlib.util.spec_from_file_location(module_name, module_abs_dir)
    if spec is None:
        print(f"Could not find spec for {script_abs_path}")
        if logger:
            logger.error(f"import_script: Could not find spec for {script_abs_path}")
    else:
        if logger:
            logger.info(f"import_script: spec: {spec}")
        # get module
        module = importlib.util.module_from_spec(spec)
        if logger:
            logger.info(f"import_script: module: {module}")
        sys.modules[module_name] = module  # Important for caching

        if logger:
            logger.info(f"import_script: module added to sys modules.")
        try:
            spec.loader.exec_module(module)
            if logger:
                logger.info(f"import_script: module: {module}")

            # Access all functions defined in the module

            for name, obj in inspect.getmembers(module, inspect.isdatadescriptor()):
                if logger:
                    logger.info(f" * import_script: to get functions - name: {name} , obj: {obj}")
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # Ensure the function is actually defined in this module and not an imported one
                if logger:
                    logger.info(f"import_script: to get functions - name: {name} , obj: {obj}")
                if obj.__module__ == module_name:
                    functions[name] = obj

            if logger:
                logger.info(f"import_script: all functions recognized: {functions}")
        except Exception as e:
            del sys.modules[module_name]
            logger.error(f"Failed to execute module {module_name} from {file_path}: {e}")
            # raise ImportError(f"Failed to execute module {module_name} from {file_path}: {e}")



    return functions



def call_func(output: str = "", functions_script_name:str = "", module_abs_dir:str = "", logger: logging = None):
    functions = {}
    try:
        functions = get_module_functions(functions_script_name, module_abs_dir, logger)
    except Exception as e:
        logger.error(f"FuncGemma - call_func: Error in importing Function ({functions_script_name}) from {module_abs_dir}")

    calls = None
    try:
        calls = extract_tool_calls(output)
    except Exception as e:
        logger.error("FuncGemma - call_func: Error in extracting tool calls.")

    message = []
    if calls:
        message.append({
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": call} for call in calls]
        })
        # print("call_func:", message[-1])
        # if logger:
        #     logger.info(f"call_func: {message[-1]}")
        # Call the function and get the result
        #####################################
        # WARNING: This is a demonstration. #
        #####################################
        # Using globals() to call functions dynamically can be dangerous in
        # production. In a real application, you should implement a secure way to
        # map function names to actual function calls, such as a predefined
        # dictionary of allowed tools and their implementations.
        results = None
        try:
            results = [
                # {"name": c['name'], "response": globals()[c['name']](**c['arguments'])}
                {"name": c['name'], "response": functions[c['name']](**c['arguments'])}
                for c in calls
            ]
            logger.info(f"FuncGemma - call_func: results: {results}")
        except Exception as e:
            logger.error(f"FuncGemma - call_func: Error in calling a function: {e}.")

        message.append({
            "role": "tool",
            "content": results
        })
        # print("call_func: ", message[-1])
        # if logger:
        #     logger.info(f"call_func: {message[-1]}")

    return message