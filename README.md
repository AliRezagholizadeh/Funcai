# Funcall
This is a library to access and store Function Gemma model locally and deploy it in an application.
It just needs to provide relevant functions (which the model has trained on) as well as tools containing function schemas.


[//]: # (You can find an example of deplying the library in a simple Mobile application made by Kivy. In a separate repository, )
[//]: # (you will find its usage as a tool in a ReAct &#40;Reasoning Action&#41; agent.)


## Environment Setup

It is highly recommended to use a virtual environment to isolate project dependencies.

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
You can then import it into a Python session or another project and use your functions:

```Python
from Funcai.funcgemma import FGmodel_pipeline
```



