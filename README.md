# DetoxAI

[![Python tests](https://github.com/FairUnlearn/detoxai/actions/workflows/python-tests.yml/badge.svg?branch=main)](https://github.com/FairUnlearn/detoxai/actions/workflows/python-tests.yml)


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

# Acknowledgment
If you use this library in your work please cite as: 
```
@misc{detoxai,
  authors={Ignacy Stepka and Lukasz Sztukiewicz and Michal Wilinski},
  title={DetoxAI: a Python Package for Debiasing Neural Networks},
}
```


# License 
MIT License
