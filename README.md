# DetoxAI

desription here

# Quickstart

```python
import detoxai

model = ...
dataloader = ... # required for methods that fine-tune the model

corrected = detoxai.debias(model, dataloader, 'gender.male')

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
#uv init 
uv venv 

# activate the virtual environment
source .venv/bin/activate

python main.py
```
