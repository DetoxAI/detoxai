





| ![thumbnail](images/thumbnail_detoxai.png) | **DetoxAI** is a Python package for debiasing neural networks in image classification tasks. It transforms biased models into fair and balanced ones with minimal code changes.  | 
|:-------------------------:|:----------------------------:|
|  Website and demo: [https://detoxai.github.io](https://detoxai.github.io)|  Documentation: [https://detoxai.readthedocs.io](https://detoxai.readthedocs.io) |

[![Python tests](https://github.com/DetoxAI/detoxai/actions/workflows/python-tests.yml/badge.svg?branch=main)](https://github.com/DetoxAI/detoxai/actions/workflows/python-tests.yml)


## Installation

Install DetoxAI from PyPI:

```bash
pip install detoxai
```

## Quickstart

```python
import detoxai

model = ...
dataloader = ...  # should output (input, label, protected attributes)

corrected = detoxai.debias(model, dataloader)

metrics = corrected["SAVANIAFT"].get_all_metrics()
model = corrected["SAVANIAFT"].get_model()
```

Minimal runnable example:

```python
import torch
import torchvision
import detoxai

model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

X = torch.rand(128, 3, 224, 224)
Y = torch.randint(0, 2, size=(128,))
PA = torch.randint(0, 2, size=(128,))

dataloader = torch.utils.data.DataLoader(list(zip(X, Y, PA)), batch_size=32)

results: dict[str, detoxai.CorrectionResult] = detoxai.debias(model, dataloader)
```

More examples: see `examples/` folder.


# How DetoxAI Works

![Workflow](images/flow.png)

DetoxAI transforms biased neural networks into fair models with simple code integration.

# Key Features

## 🛠️ Multiple Debiasing Methods

- ClArC family ([paper](https://www.sciencedirect.com/science/article/pii/S1566253521001573))
- Zhang et al. ([paper](https://arxiv.org/abs/1801.07593))
- Savani et al. ([paper](https://arxiv.org/abs/2006.08564))
- Belrose et al. ([paper](https://arxiv.org/abs/2306.03819))

[Learn how to add your own method →](https://detoxai.readthedocs.io/en/latest/tutorials.adding_a_method.html)

## 📀 Dataset Integration

- CelebA ([dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))
- FairFace ([paper](https://arxiv.org/abs/1908.04913))
- CIFAR-10/100 ([dataset](https://www.cs.toronto.edu/~kriz/cifar.html))
- Caltech101 ([dataset](https://data.caltech.edu/records/mzrjq-6wc02))

[Learn how to use datasets →](https://detoxai.readthedocs.io/en/latest/tutorials.dataset.html)

## 📊 Visualization Tools

- Saliency maps with Layer-wise Relevance Propagation ([paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140))
- Side-by-side comparisons of original vs. debiased models
- Aggregate visualizations to track model focus shift

[Explore visualization tools →](https://detoxai.readthedocs.io/en/latest/detoxai.visualization.html)

| Before/After Saliency Map | Aggregate Bias Visualization |
|:-------------------------:|:----------------------------:|
| ![Side-by-side LRP](images/side-by-side.png) | ![Aggregate visualization](images/aggregate.png) |
| Saliency maps show model focus during classification. |Aggregate visualizations show focus shift after debiasing. |


## 💻 Simple API

- Works with existing PyTorch models
- Standard dataloaders
- Single function call for multiple debiasing methods

[See a complete example →](https://detoxai.readthedocs.io/en/latest/examples/example.html)





# Development

### Install the environment using `uv` (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or pip install uv

uv venv
uv pip install -r pyproject.toml
source .venv/bin/activate

python main.py
```

### Alternatively, install the environment using pip

```bash
pip install .
# or pip install -e . for editable install
```

### Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```

### Rebuild documentation

```bash
chmod u+x ./build_docs.sh
./build_docs.sh
```

---

# Acknowledgment

If you use DetoxAI, please cite:

```bibtex
@misc{detoxai2025,
  author={Ignacy Stepka and Lukasz Sztukiewicz and Michal Wilinski and Jerzy Stefanowski},
  title={DetoxAI: a Python Package for Debiasing Neural Networks},
  year={2025},
  url={https://github.com/DetoxAI/detoxai},
}
```

# License

MIT License