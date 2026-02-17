from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import torch 
from huggingface_hub import login
from pathlib import Path 
from dotenv import load_dotenv
import os
import logging

MODEL_BASE_DIR = "model/FunctionGemma327M_action"

def get_model_dir_path(model_name:str, base_dir:str= None, pre_trained:bool = True)->Path:
    train_status = "fine_tunned"
    if(pre_trained):
        train_status = "pre_trained"

    # model base dir
    if(base_dir):
        model_base_path = Path(".") / base_dir
    else:
        model_base_path = Path(".") / MODEL_BASE_DIR

    # model dir path
    model_dir_path = model_base_path / train_status / model_name

    # check/make 
    if not model_dir_path.is_dir():
        model_dir_path.mkdir(parents= True, exist_ok= True)
    

    return model_dir_path
    

def get_model(access_token:str, model_name:str , model_base_dir:str = None, logger: logging = None):
    """
    To get the model, processor, and tokenizer of specific Function Gemma model, as well as device.
    
    Parameters:
        access_token: Hugging face API access key
        model_name: Function Gemma specific model name
        model_base_dir: base dirs from the current dir to store and load the model and its dependencies.
        logger: logging instance to store the logs.
    Returns:
        model: Function Gemma model
        processor: Function Gemma processor
        tokenizer: Function Gemma tokenizer
        device: available device: cuda, mps, or cpu
    """
    # determine model name
    # model_name = "google/functiongemma-270m-it"
    # model_name = "litert-community/FunctionGemma_270M_Mobile_Actions"

    # get stored model path 
    model_path = get_model_dir_path(model_name, model_base_dir)
    
    # initialize
    processor = None
    model = None
    tokenizer = None

    # Load pre stored model
    try:
        processor = AutoProcessor.from_pretrained(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto", device_map="auto")

    except Exception as e:
        if logger:
            logger.info(f"get_model: >> Loading {model_name} from saved model failed. message: {e}")
            logger.info(">> To load from Hugging Face repository... <<")

    if(not model or not processor):
        # CONNECT TO HUGGING FACE
        # load env variables within env.sample
        # base_dir = Path(".")
        # sample_env_path = base_dir / "env.sample"
        # load_dotenv(dotenv_path = sample_env_path)
        #
        # # use Hugging Face Access Token and login
        # access_token = os.environ.get("HUG_ACCESS_TOKEN")

        # login Hugging Face
        login(access_token)
        if logger:
            logger.info(">> Logged in Hugging Face. <<")

        # DOWNLOAD functiongemma model and processor
        processor = AutoProcessor.from_pretrained(model_name, device_map="auto", fix_mistral_regex=True)
        processor.save_pretrained(model_path)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # tokenizer.save_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
        model.save_pretrained(model_path)
        
        if logger:
            logger.info(f">> Model and Processor ({model_name}) downloaded from Hugging Face and stored locally. <<")

    # set device
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    assert tokenizer!= None, "tokenizer not recognized"
    if logger:
        logger.info(f"model, processor, and device ({device}) found.")
    return model, processor, tokenizer, device

