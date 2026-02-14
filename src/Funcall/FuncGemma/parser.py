import re
# from Functions import *
import os
import sys
import importlib.util
import logging

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

def import_script(script_name:str, script_abs_path:str):

    # file_path = "/path/to/your/custom_module.py"
    # module_name = "custom_module"

    spec = importlib.util.spec_from_file_location(script_name, script_abs_path)
    if spec is None:
        print(f"Could not find spec for {script_abs_path}")
    else:
        module = importlib.util.module_from_spec(spec)
        sys.modules[script_abs_path] = module  # Important for caching
        spec.loader.exec_module(module)
        # Now you can use module.your_function()



def call_func(output, function_script_name:str, script_path:str, logger: logging = None):
    try:
        import_script(function_script_name, script_path)
    except Exception as e:
        logger.error("FuncGemma - call_func: Error in importing Function.")

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
        try:
            results = [
                {"name": c['name'], "response": globals()[c['name']](**c['arguments'])}
                for c in calls
            ]
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