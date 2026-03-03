# Funcai
This is a library to access and store Function Gemma model locally and deploy it in an application.
It just needs to provide relevant functions (which the model has trained on) as well as tools containing function schemas.


[//]: # (You can find an example of deplying the library in a simple Mobile application made by Kivy. In a separate repository, )
[//]: # (you will find its usage as a tool in a ReAct &#40;Reasoning Action&#41; agent.)

## Function Gemma
Function Gemma is a lightweight model trained by Google to call relevant functions it has been trained on. We used 
[litert-community/FunctionGemma_270M_Mobile_Actions](https://huggingface.co/litert-community/FunctionGemma_270M_Mobile_Actions) as a default.
It is a fine-tuned version which trained on [google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions). As mentioned there, the model is optimized to call these mobile action functions:

1- turn_on_flashlight() - Turns the device flashlight on

2- turn_off_flashlight() - Turns the device flashlight off

3- create_contact(first_name, last_name, phone_number?, email?) - Creates a new contact

4- send_email(to, subject, body?) - Sends an email to a recipient

5- show_map(query) - Displays a location on the map by name, business, or address

6- open_wifi_settings() - Opens the Wi-Fi settings screen

7- create_calendar_event(title, datetime) - Creates a calendar event (datetime in ISO format: YYYY-MM-DDTHH:MM:SS)


To adjust the functions, please follow ```tests/utils/Functions.py```. There, you need to:
- add schema and function
- update FG_tools_list


# setup
You need to provide API keys from Hugging Face (to load Function Gemma 270M Mobile Actions). To do so, you need to create an account on the [Hugging Face](https://huggingface.co/), log in to it and, navigate to the Access Tokens section in your settings.

For the test, you need to store it as an env variable:
```.env in ReAct folder
HUG_ACCESS_TOKEN = "Your Hugging Face API key"
```


## Environment Setup

It is highly recommended using a virtual environment to isolate project dependencies.

### Using `venv` (Recommended for most projects)
1. **Create the virtual environment** (named `.venv`):
   *   **Windows (Command Prompt):** 
   ```bash
    python3 -m venv .local_venv
    ```
    *Note: The directory name `.venv` is a common convention and is typically excluded from version control using a `.gitignore` file.*
    
    *    **macOS/Linux: (using pyenv):** 
    ```bash
    pyenv virtualenv python_version(like 3.10.4) pyenv_env_name
    ```
   If local environment is desired:
    ```bash
   cd project/dir
   virtualenv --python="/path/to/.pyenv/versions/{python_version}/envs/{pyenv_env_name}/bin/python" "local_venv"
    ```
3. **Activate the environment:**

    *   **macOS/Linux:**
        ```bash
        source .local_venv/bin/activate
        ```
    *   **Windows (Command Prompt):**
        ```bash
        .local_venv\Scripts\activate.bat
        ```

4. **Install dependencies:**
    Once the environment is activated, install the required packages using `pip` and the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```



### Test:
First, put your hugging face APIs access key to HUG_ACCESS_TOKEN variable environment within ```tests/.env``` file.
Activate a local python environment with minimum Python 3.10 and install requirements. Then, run the command to test the library:
```commandline
python3 -m pytest -o log_cli=true --log-cli-level=info
```


### Build library:
Ensure you have the build tools installed:
```commandline
pip install --upgrade build twine
```
From your project's root directory, run the build command:
```commandline
python3 -m build 
```
This command generates distribution files (like .whl and .tar.gz) in a newly created dist folder.

### Install:
You can install your package locally in editable mode to test it as if it were installed from PyPI:
```commandline
pip install -e .
```


## Sample code
You can then import it into a Python session or another project and use your own functions:

```Python
from Funcai.funcgemma import FGmodel_pipeline
from Funcai.FuncGemma.parser import get_module_functions
# import sys
# sys.path.append(abs path/to/the dir containing Function.py))
# from Function import FG_tools_list

# get access token
access_token = os.environ.get("HUG_ACCESS_TOKEN")

# model_name = "google/functiongemma-270m-it"
model_name = "litert-community/FunctionGemma_270M_Mobile_Actions"


module_abs_dir = Path("abs path/to/Function.py")
FG_module_name = "Function"

# get functions
try:
    functions = get_module_functions(FG_module_name, module_abs_dir, logger)
except Exception as e:
    raise Exception(f"FuncGemma: Error in importing module ({FG_module_name}) from {module_abs_dir}: {e}")

try:
    input_ = {
        "model_name": model_name,
        "access_token": access_token,
        "model_base_dir": str(model_base_dir),
        "tools": FG_tools_list,
        "functions": functions,
        # "logger": logger

    }
    fg = FGmodel_pipeline(**input_)

except Exception as e:
    if logger:
        logger.error(f"actions: FGmodel failed to be instantiated: {e}")


messages = fg.infer(prompt)

print(f"result: {messages[-1]['content']}")


```



