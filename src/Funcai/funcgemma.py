import copy
from datetime import datetime
from zoneinfo import ZoneInfo
# from Functions import tools
import dotenv
from transformers import pipeline
import logging
from typing import Callable
from pathlib import Path
import sys

curr_dir = Path(".").absolute() / "src" / "Funcai"
# print(f"curr_dir: {curr_dir}")
sys.path.append(str(curr_dir))
from Funcai.FuncGemma.model import get_model
from Funcai.FuncGemma.parser import call_func, get_module_functions
from Funcai.FuncGemma.utils import CHATSIDE, Role

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

class FGBasemodel:
    developer_content = f"""You are a model that can do function calling with the following functions. 
    Current date and time given in YYYY-MM-DDTHH:MM:SS format: {datetime.now(ZoneInfo("America/Toronto")).strftime('%Y-%m-%dT%H:%M:%S %Z')}.
    Day of week is {weekdays[datetime.now().weekday()]}.
    """

    message_ = [
                {
                    "role": Role.Developer.value,
                    "content": (developer_content),
                }
        ]
    tools = []

    # def __init__(self, model_name: str = "litert-community/FunctionGemma_270M_Mobile_Actions", access_token: str="", model_base_dir: str= "", tools:list= [], functions_script_name: str= "", functions_abs_path: str= "", logger: logging = None):
    def __init__(self, model_name: str, access_token: str="", model_base_dir: str= "", tools:list= [], functions: dict = {}, logger: logging = None):
        """

        Parameters:
        
        model_name: Function Gemma specific model name
        access_token: Hugging face API access key
        model_base_dir: base dirs from the current dir to store and load the model and its dependencies.
        tools: list of function schemas, like: tools = [weather_function_schema], which
            weather_function_schema = {
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "description": "Gets the current temperature for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name, e.g. San Francisco",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit, e.g. celsius",
                            }
                        },
                        "required": ["location"],
                    },
                }
            }
        # functions_script_name: name of the Python script to import containing the functions to call
        # functions_abs_path: abs path to the functions_script_name
        functions: a dictionary containing all functions mentioned in the tools list.
        logger: logging instance to store the logs.
        """
        assert model_name, "model_name is not defined."
        self.model_name = model_name
        self.access_token = access_token
        self.model_base_dir = model_base_dir
        # self.functions_script_name = functions_script_name
        # self.functions_abs_path = functions_abs_path
        self.functions = functions
        self.logger = logger
        # load determined model
        # if(self.access_token):
        self.load_model()
        # else:
        #     self.logger.error("access_token is not provided.")
        [self.tools.append(t) for t in tools]
        

    def load_model(self):
        self.model, self.processor, self.tokenizer, self.device = get_model(access_token=self.access_token, model_name=self.model_name, model_base_dir = self.model_base_dir, logger = self.logger)
        print("model loaded.")

    def add_message(self, message):
        self.message += message

    def add_tools(self, tools):
        self.tools += tools
    
    @classmethod
    def re_new_message_dec(cls, _):
        def wrap_1(func):
            # print("func: ", func)
            def wrapper(self, *args, **keyargs):
                # self = args[0]
                # print(f"cls type: {type(cls)} - cls: {cls} - is instance: {isinstance(cls, FGBasemodel)} - is string: {isinstance(cls, str)}")
                # print("clas: ", clas)
                # print("func: ", func)
                # print("args: ", [_ for _ in args])
                # print("keyargs: ", keyargs)

                # self.message = copy.deepcopy(cls(model_name=self.model_name, access_token=self.access_token, model_base_dir=self.model_base_dir, tools= self.tools, functions=self.functions, logger= self.logger).message_)
                self.message = copy.deepcopy(cls.message_)
                print("FGBasemodel.message_: ", FGBasemodel.message_)
                print("message: ", self.message)

                result = func(self, *args, **keyargs)
                return result
            return wrapper
        return wrap_1

class FGmodel(FGBasemodel):
    @FGBasemodel.re_new_message_dec
    def infer(self, prompt:str, call_function_active:bool = True, write_call_back:Callable = None):
        new_message = [
            {"role": Role.User.value,
             "content": prompt}
        ]
        self.message += new_message
        print(f"FGmodel-infer – new_message: {new_message}")
        if self.logger:
            self.logger.info(f"FGmodel-infer – new_message: {new_message}")

        # apply chat template
        inputs = self.processor.apply_chat_template(self.message, tools=self.tools, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt")

        out = self.model.generate(**inputs.to(self.device), pad_token_id=self.processor.eos_token_id, max_new_tokens=128)
        output = self.processor.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        if self.logger:
            self.logger.info(f"FGmodel-infer – output 1: {output}")
        else:
            print("*** there is no logger set.")
        # call right function inside the Functions.py
        # TODO: you might want to reform the call_func to be class with initialized variables and run func.
        print(f"FGmodel-infer – to call func with this output {output}")
        

        # message_res = call_func(output= output, functions_script_name=self.functions_script_name, module_abs_dir=self.functions_abs_path, logger=self.logger)
        message_res = call_func(output = output, functions = self.functions, call_function_active=call_function_active, logger= self.logger)


        self.message += message_res
        if self.logger:
            self.logger.info(f"FGmodel-infer – call out message: {message_res}")

        if(self.logger):
            self.logger.info(f"{('*' * 80)}")
            for mess in self.message:
                self.logger.info(f"FGmodel-infer – message after call: {mess }")
            self.logger.info(f"{('*'*80)}")

        # calling the model with function's response
        inputs = self.processor.apply_chat_template(self.message, tools=self.tools, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt")
        out = self.model.generate(**inputs.to(self.model.device), pad_token_id=self.processor.eos_token_id, max_new_tokens=128)
        if self.logger:
            self.logger.info(f"FGmodel-infer – output 2: {out}")
        # print("Infer: intermediate output: ", out)

        generated_tokens = out[0][len(inputs["input_ids"][0]):]
        # output = self.processor.decode(generated_tokens, skip_special_tokens=True)
        output = self.processor.decode(generated_tokens)
        self.message.append({"role": Role.Assistant.value, "content": output})
        # print("Infer: final output: ", output)
        if(self.logger):
            self.logger.info(f"FGmodel-infer – final output: {output}")


        if write_call_back:
            # put the response into the widget
            write_call_back(output, CHATSIDE.Agent.value)

        return self.message

class FGmodel_pipeline(FGBasemodel):
    def __init__(self, *args, **kwargs):
        print(f"args: {args} - kwargs: {kwargs}")
        super().__init__(*args, **kwargs)
        # self.tokenizer = self.processor.backend_tokenizer
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    @FGBasemodel.re_new_message_dec(FGBasemodel)
    def infer(self, prompt:str, call_function_active:bool = True, write_call_back:Callable = None):

        new_message = [
            {"role": Role.User.value,
             "content": prompt}
        ]
        self.message += new_message
        if self.logger:
            self.logger.info(f"FGmodel_pipeline-infer – new_message: {new_message}")

        # history = []
        thought_iterations = 1
        while (thought_iterations:= thought_iterations - 1) >=0:
            # Apply chat template
            # templated_message = message_template(self.message, history, self.tools)
            prompt = self.tokenizer.apply_chat_template(
                # templated_message,
                self.message,
                tools=self.tools,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Generate
            output = self.pipe(prompt, max_new_tokens=200)[0]["generated_text"][len(prompt):].strip()
            # print("Pipeline output 1:", output)
            if self.logger:
                self.logger.info(f"FGmodel_pipeline-infer – pipe output: {output}")
            else:
                print("*** there is no logger set.")

            if(output):
                # call right function inside the Functions.py
                # TODO: you might want to reform the call_func to be class with initialized variables and run func.
                try:

                    print(f"FGmodel-infer – to call func with this output {output}")
                    # message_res = call_func(output= output, functions_script_name=self.functions_script_name, module_abs_dir=self.functions_abs_path, logger=self.logger)
                    message_res = call_func(output= output, functions= self.functions, call_function_active= call_function_active, logger=self.logger)

                    self.message += message_res
                    # history.append(message_res)

                    if self.logger:
                        self.logger.info(f"FGmodel_pipeline-infer – call out message: {message_res}")
                except Exception as e:
                    self.logger.error(f"FGmodel_pipeline-infer: call_func raised an error: {e}")
                    raise e

                if write_call_back:
                    # put the response into the widget
                    if("answer" in output):
                        write_call_back(output["answer"], CHATSIDE.Agent.value)
                        break
                    else:
                        write_call_back(output, CHATSIDE.Agent.value)
            else:
                break

        if self.logger:
            self.logger.info(f"FGmodel_pipeline-infer – Final Message {('*' * 80)}")
            for mess in self.message:
                self.logger.info(f"FGmodel_pipeline-infer – message: {mess}")
            self.logger.info(f"{('*' * 80)}")


        return self.message




if __name__ == "__main__":
    import os



    # Get the absolute path of the directory containing the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"current_dir: {current_dir}")

    # Construct the path to the parent directory
    TEST_DIR_PATH = Path(current_dir).parent.parent / 'tests'
    print(f"test_dir: {TEST_DIR_PATH}")

    # API access token
    env_path = TEST_DIR_PATH / ".env"
    print("env_path: ", env_path)
    dotenv.load_dotenv(dotenv_path=env_path)

    func_dir = TEST_DIR_PATH / 'utils'
    print(f"func_dir: {func_dir}")
    # Add the src directory to the system path
    sys.path.append(str(func_dir))

    from Functions import *
    from dotenv import load_dotenv

    # ---------
    LOG_DIR = "logs"
    logdir = TEST_DIR_PATH / LOG_DIR
    logdir.mkdir(parents=True, exist_ok=True)
    datetime_now = f"{datetime.now(ZoneInfo('America/Toronto')).strftime('%Y-%m-%dT%H:%M:%S %Z')}"
    logging.basicConfig(filename=f'{logdir}/test_run_{datetime_now}.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    # Create a logger for the main module
    logger = logging.getLogger(__name__)

    access_token = os.environ.get("HUG_ACCESS_TOKEN")

    tools = tools_list

    function_name = "Functions"
    function_module_dir = f"{func_dir}/{function_name}.py"
    # ---------
    # API access token
    env_path = Path(".").absolute() / "tests" / ".env"
    print("env_path: ", env_path)
    load_dotenv(dotenv_path=env_path)

    # model_name = "google/functiongemma-270m-it"
    model_name = "litert-community/FunctionGemma_270M_Mobile_Actions"

    model_base_dir_path = TEST_DIR_PATH / f"model"

    # take functions
    functions = {}
    try:
        functions = get_module_functions(function_name, function_module_dir, logger)
    except Exception as e:
        logger.error(
            f"FuncGemma - call_func: Error in importing module ({function_name}) from {function_module_dir}: {e}")
        raise e

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

    # prompt = 'what is the temperature in Tehran?'
    prompt = 'Find the current temperature for Tehran'
    # prompt = 'Schedule a "team meeting" tomorrow at 4pm.'
    messages = fg.infer(prompt)

    print(f"Test: messages: {messages}")
    print(f"Test: messages: {messages[-1]['content']}")

    # prompt = 'Find the current temperature for Tehran'
    prompt = "put on my calendar my tomorrow's meeting at 10am with Nikolaj."
    messages = fg.infer(prompt)

    print(f"Test: messages: {messages}")
    print(f"Test: messages: {messages[-1]['content']}")