from model import get_model
from parser import call_func
from datetime import datetime
# from Functions import tools
from utils import CHATSIDE
from transformers import pipeline
from zoneinfo import ZoneInfo
import logging
from typing import Callable

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

class FGBasemodel:
    developer_content = f"""You are a model that can do function calling with the following functions. 
    Current date and time given in YYYY-MM-DDTHH:MM:SS format: {datetime.now(ZoneInfo("America/Toronto")).strftime('%Y-%m-%dT%H:%M:%S %Z')}.
    Day of week is {weekdays[datetime.now().weekday()]}.
    """

    message = [
                {
                    "role": "developer",
                    "content": (developer_content),
                }
        ]
    tools = []

    def __init__(self, model_name: str, access_token: str, model_base_dir: str, tools:list, function_script_name: str, function_abs_path: str, logger: logging = None):
        self.model_name = model_name
        self.access_token = access_token
        self.model_base_dir = model_base_dir
        self.function_script_name = function_script_name
        self.function_abs_path = function_abs_path
        self.logger = logger
        # load determined model
        self.load_model()
        [self.tools.append(t) for t in tools]

    def load_model(self):
        self.model, self.processor, self.tokenizer, self.device = get_model(self.access_token, self.model_name, self.model_base_dir)
        print("model loaded.")

    def add_message(self, message):
        self.message += message

    def add_tools(self, tools):
        self.tools += tools

class FGmodel(FGBasemodel):

    def infer(self, new_message, call_back):
        # tools = [weather_function_schema]
        self.message += new_message
        # print(f"self.message: {self.message}")

        # apply chat template
        inputs = self.processor.apply_chat_template(self.message, tools=self.tools, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt")

        out = self.model.generate(**inputs.to(self.device), pad_token_id=self.processor.eos_token_id, max_new_tokens=128)
        output = self.processor.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        # call right function inside the Functions.py
        # TODO: you might want to reform the call_func to be class with initialized variables and run func.
        message_res = call_func(output, function_script_name=self.function_script_name, script_path= self.function_abs_path, logger= self.logger)

        self.message += message_res
        # print(f"{('*'*80)}")
        self.logger.info(f"{('*' * 80)}")
        for mess in self.message:
            # print(f"Infer: message after call: {mess }")
            self.logger.info(f"FuncGemma - Infer: message after call: {mess }")
        # print(f"{('*'*80)}")
        self.logger.info(f"{('*'*80)}")

        # calling the model with function's response
        inputs = self.processor.apply_chat_template(self.message, tools=self.tools, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt")
        out = self.model.generate(**inputs.to(self.model.device), pad_token_id=self.processor.eos_token_id, max_new_tokens=128)

        # print("Infer: intermediate output: ", out)

        generated_tokens = out[0][len(inputs["input_ids"][0]):]
        # output = self.processor.decode(generated_tokens, skip_special_tokens=True)
        output = self.processor.decode(generated_tokens)
        self.message.append({"role": "assistant", "content": output})
        # print("Infer: final output: ", output)
        self.logger.info("Infer: final output: ", output)

        # put the response into the widget
        call_back(output, CHATSIDE.Agent)


class FGmodel_pipeline(FGBasemodel):
    def __init__(self, *args):
        super().__init__(*args)
        # self.tokenizer = self.processor.backend_tokenizer
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def infer(self, new_message, write_call_back:Callable = None):
        # tools = [weather_function_schema]
        self.message += new_message
        # print(f"self.message: {self.message}")
        # inputs = self.processor.apply_chat_template(self.message, tools=self.tools, add_generation_prompt=True, return_dict=True,
        #                                        return_tensors="pt")

        history = []
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
            print("Pipeline output 1:", output)

            if(output):
                # call right function inside the Functions.py
                # TODO: you might want to reform the call_func to be class with initialized variables and run func.
                message_res = call_func(output, function_script_name=self.function_script_name,
                                        script_path=self.function_abs_path, logger=self.logger)

                self.message += message_res
                history.append(message_res)

                # put the response into the widget
                if write_call_back:
                    if("answer" in output):
                        write_call_back(output["answer"], CHATSIDE.Agent)
                        break
                    else:
                        write_call_back(output, CHATSIDE.Agent)
            else:
                break



        print(f"{('*' * 80)}")
        for mess in self.message:
            print(f"Infer: message after call: {mess}")
        print(f"{('*' * 80)}")

        # # Apply chat template
        # prompt = self.tokenizer.apply_chat_template(
        #     self.message,
        #     tools=self.tools,
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )
        #
        # # Generate
        # output = self.pipe(prompt, max_new_tokens=200)[0]["generated_text"][len(prompt):].strip()
        # # output = self.pipe(prompt, max_new_tokens=200)[0]
        print("Pipeline output 2:", output)




def message_template(query, history, tools):
    query['']
    return f"""You are a ReAct (Reasoning and Acting) agent tasked with answering the following query:

    Query: {query}
    
    Your goal is to reason about the query and decide on the best course of action to answer it accurately.
    
    Previous reasoning steps and observations: {history}
    
    Available tools: {tools}
    
    Instructions:
    1. Analyze the query, previous reasoning steps, and observations.
    2. Decide on the next action: use a tool or provide a final answer.
    3. Respond in the following JSON format:
    
    If you need to use a tool:
    {{
        "thought": "Your detailed reasoning about what to do next",
        "action": {{
            "name": "Tool name (weather_function_schema, create_calendar_event_schema, send_email_schema, or none)",
            "reason": "Explanation of why you chose this tool",
            "input": "Specific input for the tool, if different from the original query"
        }}
    }}
    
    If you have enough information to answer the query:
    {{
        "thought": "Your final reasoning process",
        "answer": "Your comprehensive answer to the query"
    }}
    
    Remember:
    - Be thorough in your reasoning.
    - Use tools when you need more information.
    - Always base your reasoning on the actual observations from tool use.
    - If a tool returns no results or fails, acknowledge this and consider using a different tool or approach.
    - Provide a final answer only when you're confident you have sufficient information.
    - If you cannot find the necessary information after using available tools, admit that you don't have enough information to answer the query confidently."""