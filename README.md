# Funcall
This is a library to access and store Function Gemma model locally and deploy it in an application.

[//]: # (You can find an example of deplying the library in a simple Mobile application made by Kivy. In a separate repository, )
[//]: # (you will find its usage as a tool in a ReAct &#40;Reasoning Action&#41; agent.)

### Build:
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
You can install your package locally in editable mode to test it as if it were installed from
```commandline
from PyPI:pip install -e .
```
You can then import it into a Python session or another project and use your functions:

```Python
from Funcall.funcgemma import FGmodel_pipeline
```

