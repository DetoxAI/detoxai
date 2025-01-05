# DetoxAI

desription here

# Quickstart

```python
import detoxai

model = ...
dataloader = ... # required for methods that fine-tune the model

corrected = detoxai.debias(model, dataloader)

metrics = corrected.get_all_metrics()
model = corrected.get_model()
```

# Installation

We recommend using `uv` to install DetoxAI. You can install `uv` by running the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or you can install it via PIP pip install uv
```

Once you have `uv` installed, you can set up local environment by running the following command:

```bash
# create a virtual environment with the required dependencies
uv venv 

# install the dependencies in the virtual environment
uv pip install -r pyproject.toml

# activate the virtual environment
source .venv/bin/activate

python main.py
```


To upgrade packages run
```bash
    uv lock --upgrade-package gdown
```

To build package run (recommended)
```bash
    uv build --no-sources
```

uv add --dev ipykernel
uv run ipython kernel install --user --name=detoxaikernel

uv build --no-sources --index-strategy unsafe-best-match

uv run --with detoxai --no-project -- python -c "import detoxai; print(detoxai.__version__)"

uv publish --token $UV_PUBLISH_TOKEN --index testpypi

### Be careful with the following command, upload all files from dist folder


pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ detoxai
