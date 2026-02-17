# import dotenv
from dotenv import load_dotenv
from pathlib import Path
import pytest
import logging
import sys

SRC_DIR_PATH = Path(".").parent / "src"
print("src_dir: ", SRC_DIR_PATH)
sys.path.append(str(SRC_DIR_PATH))
from Funcai.funcgemma import FGmodel_pipeline

TEST_DIR_PATH = Path(".").absolute()
sys.path.append(str(TEST_DIR_PATH))
from conftest import *

def test_funcgemma_(prompt:str, access_token:str, tools: list, function_name:str, function_module_dir:str, logger: logging):
    # API access token
    env_path = Path(".").absolute()/ "tests"/ ".env"
    print("env_path: ", env_path)
    load_dotenv(dotenv_path = env_path)

    # model_name = "google/functiongemma-270m-it"
    model_name = "litert-community/FunctionGemma_270M_Mobile_Actions"


    model_base_dir_path = TEST_DIR_PATH / f"model"
    input_ = {
        "model_name": model_name,
        "access_token": access_token,
        "model_base_dir": str(model_base_dir_path),
        "tools": tools,
        "functions_script_name": function_name,
        "functions_abs_path": function_module_dir,
        "logger": logger

    }
    fg = FGmodel_pipeline(**input_)
    # prompt = 'Schedule a "team meeting" tomorrow at 4pm.'
    messages = fg.infer(prompt)
    
    print(f"Test: messages: {messages}")