# Large Language Models book

## Building

This book uses [Jupyter book](https://jupyterbook.org).

Install the dependencies:
```shell
pip install -r requirements.txt
```

Clone this repository and run the build from the root directory:
```shell
jupyter-book build .
```

This will generate the HTML in _build/html.

## Contribution

Execute the notebook(s) before committing (as most of them would be too expensive to automatically execute in a pipeline):

```shell
jupyter nbconvert --to notebook --inplace --execute notebook.ipynb
```
