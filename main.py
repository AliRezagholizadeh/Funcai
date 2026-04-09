# import dotenv
import os.path
import time

from Funcai.FuncGemma.parser import get_module_functions
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
# import pytest
import logging
import sys

PROJECT_DIR = Path(os.path.dirname(__file__))
SRC_DIR_PATH = PROJECT_DIR / "src"
TEST_DIR_PATH = PROJECT_DIR / "tests"
print("PROJECT_DIR: ", PROJECT_DIR)
print("src_dir: ", SRC_DIR_PATH)
sys.path.append(str(SRC_DIR_PATH))
from Funcai.funcgemma import FGmodel_pipeline


func_dir = TEST_DIR_PATH / 'utils'
# print(f"func_dir: {func_dir}")
# Add the src directory to the system path
sys.path.append(str(func_dir))
from plyer_mobile_functions import *






# def test_funcgemma_(prompt:str, access_token:str, tools: list, function_name:str, function_module_dir:str, logger: logging):
def get_funcgemma_(model_name: str, access_token:str, tools: list, functions: dict, logger: logging = None):

    model_base_dir_path = TEST_DIR_PATH / f"model"

    assert access_token, "access_token is empty"

    input_ = {
        "model_name": model_name,
        "access_token": access_token,
        "model_base_dir": str(model_base_dir_path),
        "tools": tools,
        # "functions_script_name": function_name,
        # "functions_abs_path": function_module_dir,
        "functions": functions,
        "logger": logger

    }
    fg = FGmodel_pipeline(**input_)
    # prompt = 'Schedule a "team meeting" tomorrow at 4pm.'
    # messages = fg.infer()
    #
    # print(f"Test: messages: {messages}")

    return fg.infer



def main():
    # logger
    LOG_DIR = "logs"
    logdir = PROJECT_DIR / LOG_DIR
    zone = "America/Toronto"

    logdir.mkdir(parents=True, exist_ok=True)
    datetime_now = f"{datetime.now(ZoneInfo(zone)).strftime('%Y-%m-%dT%H:%M:%S %Z')}"
    logging.basicConfig(filename=f'{logdir}/main_oversample_log_{datetime_now}.log', level=logging.INFO, force=True,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    
    # Create a logger for the main module
    logger = logging.getLogger(__name__)
    
    print(f"log activated: {logdir}")

    # API access token
    env_path = TEST_DIR_PATH / ".env"
    print("env_path: ", env_path)
    load_dotenv(dotenv_path=env_path)
    access_token = os.environ.get("HUG_ACCESS_TOKEN")


    # get tools and function in the relevant function name
    tools = FG_tools_list
    function_name = "plyer_mobile_functions"
    function_module_dir = f"{func_dir}/{function_name}.py"


    try:
        functions = get_module_functions(function_name, function_module_dir, logger)
    except Exception as e:
        logger.error(
            f"FuncGemma - call_func: Error in importing module ({function_name}) from {function_module_dir}: {e}")
        raise e


    # model_name = "google/functiongemma-270m-it"
    # model_name = "litert-community/FunctionGemma_270M_Mobile_Actions"
    model_name = "AliRGHZ/functiongemma-270m-it-extended-mobile-actions"



    function_gemma_infer = get_funcgemma_(model_name = model_name, access_token = access_token, tools= tools, functions= functions, logger = logger)



    # prepare the environment to do the experiment
    border = f"{'-' * 40}\n"
    # test over the sample prompts
    sample_prompts_file = TEST_DIR_PATH / "data" / "plyer_func_sample_prompts.txt"
    print(f"Do experiment over existing sample prompts located in {sample_prompts_file}")
    with open(sample_prompts_file, "r") as file:
        for prompt_ in file:
            prompt = prompt_.strip()

            result = function_gemma_infer(prompt, call_function_active = False)

            logger.info(f"{border}>> prompt:\n{prompt}\nAgent response: \n{result[-1]}\n{border}")
            print(f"{border}>> prompt:\n{prompt}\nAgent response: \n{result[-1]}\n{border}")



    print(f"Do custom experiment. Write your prompt or 'quit' to terminate the run.")

    while True:
        user_input = input(f"{border}User prompt: \n> ")

        if(user_input == "quit"):
            break

        result = function_gemma_infer(user_input, call_function_active = False)
        time.sleep(1000)

        print(f"Agent response: \n{result[-1]}\n{border}")
        logger.info(f"{border}>> prompt:\n{user_input}\nAgent response: \n{result[-1]}\n{border}")


if __name__ == "__main__":
    main()