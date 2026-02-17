
from datetime import datetime
# from Functions import tools
from transformers import pipeline
from zoneinfo import ZoneInfo
import logging
from typing import Callable
from pathlib import Path
import sys

curr_dir = Path(".").absolute() / "src" / "Funcai"
print(f"curr_dir: {curr_dir}")
sys.path.append(str(curr_dir))
from FuncGemma.model import get_model
from FuncGemma.parser import call_func
from FuncGemma.utils import CHATSIDE, Role

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

    def __init__(self, model_name: str = "litert-community/FunctionGemma_270M_Mobile_Actions", access_token: str="", model_base_dir: str= "", tools:list= [], functions_script_name: str= "", functions_abs_path: str= "", logger: logging = None):
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
        functions_script_name: name of the Python script to import containing the functions to call
        functions_abs_path: abs path to the functions_script_name
        logger: logging instance to store the logs.
        """
        self.model_name = model_name
        self.access_token = access_token
        self.model_base_dir = model_base_dir
        self.functions_script_name = functions_script_name
        self.functions_abs_path = functions_abs_path
        self.logger = logger
        # load determined model
        self.load_model()
        [self.tools.append(t) for t in tools]
        

    def load_model(self):
        self.model, self.processor, self.tokenizer, self.device = get_model(self.access_token, self.model_name, self.model_base_dir, self.logger)
        print("model loaded.")

    def add_message(self, message):
        self.message += message

    def add_tools(self, tools):
        self.tools += tools
    
    @staticmethod    
    def re_new_message_dec(func):
        def wrapper(*args, **keyargs):
            self = args[0]
            print("func: ", func)
            print("args: ", args)
            print("keyargs: ", keyargs)
            self.message = FGBasemodel.message_
            result = func(*args, **keyargs)
            return result
        return wrapper

class FGmodel(FGBasemodel):
    @FGBasemodel.re_new_message_dec
    def infer(self, prompt:str, write_call_back:Callable = None):
        new_message = [
            {"role": Role.User.value,
             "content": prompt}
        ]
        self.message += new_message
        if self.logger:
            self.logger.info(f"FGmodel-infer – new_message: {new_message}")

        # apply chat template
        inputs = self.processor.apply_chat_template(self.message, tools=self.tools, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt")

        out = self.model.generate(**inputs.to(self.device), pad_token_id=self.processor.eos_token_id, max_new_tokens=128)
        output = self.processor.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        if self.logger:
            self.logger.info(f"FGmodel-infer – output 1: {output}")
        # call right function inside the Functions.py
        # TODO: you might want to reform the call_func to be class with initialized variables and run func.
        message_res = call_func(output = output, functions_script_name=self.functions_script_name, script_path= self.functions_abs_path, logger= self.logger)

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
        super().__init__(*args, **kwargs)
        # self.tokenizer = self.processor.backend_tokenizer
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    @FGBasemodel.re_new_message_dec
    def infer(self, prompt:str, write_call_back:Callable = None):

        new_message = [
            {"role": Role.User.value,
             "content": prompt}
        ]
        self.message += new_message
        if self.logger:
            self.logger.info(f"FGmodel_pipeline-infer – new_message: {new_message}")

        # history = []
        while True:
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

            if(output):
                # call right function inside the Functions.py
                # TODO: you might want to reform the call_func to be class with initialized variables and run func.
                try:
                    message_res = call_func(output= output, functions_script_name=self.functions_script_name,
                                            module_abs_dir=self.functions_abs_path, logger=self.logger)

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


