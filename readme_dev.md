#Packaging

Documentation about creating python packages can be found here:
https://packaging.python.org/en/latest/tutorials/packaging-projects/


Instructions to build psimage distro:

1. Install in a development mode (do not need to be rebuilt after each modification):
``` python -m pip install --editable . ```

2. To build the package distro first once setup and update build tool.
```python3 -m pip install --upgrade build```
After it's ok, build the package itself:
```python3 -m build```

P.S. Commands ```python setup.py develop``` and ```python setup.py bdist_wheel``` are deprecated.