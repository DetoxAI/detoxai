# DetoxAI

[![Python tests](https://github.com/FairUnlearn/detoxai/actions/workflows/python-tests.yml/badge.svg?branch=main)](https://github.com/FairUnlearn/detoxai/actions/workflows/python-tests.yml)

DetoxAI is a Python package for debiasing neural networks. It provides a simple and efficient way to remove bias from your models while maintaining their performance. The package is designed to be easy to use and integrate into existing projects. We hosted a website with a demo and an overview of the package, which can be found at [https://detoxai.github.io](https://detoxai.github.io).  

## Installation

DetoxAI is available on PyPI, and can be installed by running the following command:
 ```bash
  pip install detoxai
  ```

## Quickstart

The snippet below shows the high-level API of DetoxAI and how to use it. 
```python
import detoxai

model = ...
dataloader = ... # has to output a tuple of three tensors: (x, y, protected attributes)

corrected = detoxai.debias(model, dataloader)

metrics = corrected["SAVANIAFT"].get_all_metrics() # Get metrics for the model debiased with SavaniAFT method
model = corrected["SAVANIAFT"].get_model()
```

A shortest snippet that would actually run and shows how to plug DetoxAI into your code is below. 
```python
import torch
import torch.nn as nn
import torchvision
import detoxai


model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Change last layer to binary output

X = torch.rand(128, 3, 224, 224)
Y = torch.randint(0, 2, size=(128,))
PA = torch.randint(0, 2, size=(128,))

dataloader = torch.utils.data.DataLoader(list(zip(X, Y, PA)), batch_size=32)

results: dict[str, detoxai.CorrectionResult] = detoxai.debias(model, dataloader)
``` 

Too see more examples of detoxai in use, see `examples/` folder.


## Development
### Install the environment using uv (recommended)
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
### Alternatively, install the environment using pip
  You can also install DetoxAI using pip. To do this, run the following command:
  ```bash
  pip install . # or pip install -e . for editable install
  ```

### Install pre-commit hooks
```bash
    pre-commit install
```

```bash
    pre-commit run --all-files
```


### Rebuild documentation
```bash
    cd docs
    make clean
    sphinx-apidoc -o . ../src/detoxai
    make html
```


## Acknowledgment
If you use this library in your work please cite as:
```
@misc{detoxai,
  authors={Ignacy Stepka and Lukasz Sztukiewicz and Michal Wilinski},
  title={DetoxAI: a Python Package for Debiasing Neural Networks},
}
```


## License
MIT License
