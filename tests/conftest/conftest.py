import os
import pytest
import sys
from pathlib import Path
import logging
SRC_DIR_PATH = Path(".").parent / "src"
print("src_dir: ", SRC_DIR_PATH)
sys.path.append(str(SRC_DIR_PATH))
from Funcai.FuncGemma.parser import get_module_functions


# Get the absolute path of the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"current_dir: {current_dir}")

# Construct the path to the parent directory
TEST_DIR_PATH = Path(current_dir).parent
print(f"test_dir: {TEST_DIR_PATH}")

func_dir = TEST_DIR_PATH / 'utils'
print(f"func_dir: {func_dir}")
# Add the src directory to the system path
sys.path.append(str(func_dir))

from Functions import *

@pytest.fixture(scope = "session")
def logger():
    LOG_DIR = "logs"
    logdir = TEST_DIR_PATH / LOG_DIR
    logdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=f'{logdir}/test_run.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    # Create a logger for the main module
    logger = logging.getLogger(__name__)

    return logger

@pytest.fixture()
def prompt():
    # prompt_ = 'Schedule a "team meeting" tomorrow at 4pm.'
    prompt_ = 'what is the temperature in Tehran?'
    return prompt_

@pytest.fixture(scope = "session")
def access_token():
     return os.environ.get("HUG_ACCESS_TOKEN")

@pytest.fixture(scope = "session")
def tools():
    return tools_list

@pytest.fixture(scope = "session")
def function_module_dir(function_name):
    # return f"{func_dir}/{function_name}"
    return f"{func_dir}/{function_name}.py"
    # return f"{func_dir}"


@pytest.fixture(scope = "session")
def function_name():
    return "Functions"

@pytest.fixture(scope = "session")
def functions(function_name, function_module_dir, logger):
    # take functions
    # functions = {}
    try:
        functions = get_module_functions(function_name, function_module_dir, logger)
    except Exception as e:
        logger.error(
            f"FuncGemma - call_func: Error in importing module ({function_name}) from {function_module_dir}: {e}")
        raise e
    return functions


