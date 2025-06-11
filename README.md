# gemma-ollama-workshop-public
Public materials for a workshop "Creating a Local Agent with Gemma"

## Setup

This repo uses [uv](https://docs.astral.sh/uv/)

To set up the kernel for use in jupyter (if you have jupyter) use
``` sh
uv sync
uv run python -m ipykernel install --name gemma-ollama-workshop-public --user
```

Now you can run jupyter and use this kernel.

If you don't have Jupyter already you can also use

``` sh
uv add jupyter
uv run jupyter notebook
```
