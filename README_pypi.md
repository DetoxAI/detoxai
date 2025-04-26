DetoxAI is a Python package for debiasing neural networks. It provides a simple and efficient way to remove bias from your models while maintaining their performance. The package is designed to be easy to use and integrate into existing projects. 
For more information about the package, see the website [https://detoxai.github.io](https://detoxai.github.io) and documentation [https://detoxai.readthedocs.io](https://detoxai.readthedocs.io)  

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
import torchvision
import detoxai

model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Make it binary classification

X = torch.rand(128, 3, 224, 224)
Y = torch.randint(0, 2, size=(128,))
PA = torch.randint(0, 2, size=(128,))

dataloader = torch.utils.data.DataLoader(list(zip(X, Y, PA)), batch_size=32)

results: dict[str, detoxai.CorrectionResult] = detoxai.debias(model, dataloader)
``` 

Too see more examples of detoxai in use, navigate to the github repo [https://github.com/DetoxAI/detoxai](https://github.com/DetoxAI/detoxai) and see `examples/` folder.



## Acknowledgment
If you use this library in your work please cite as:
```
@misc{detoxai2025,
  author={Ignacy St\k{e}pka and Lukasz Sztukiewicz and Micha\l{} Wili\'{n}ski and Jerzy Stefanowski},
  title={{DetoxAI}: a {Python} Package for Debiasing Neural Networks},
  year={2025},
  url={https://github.com/DetoxAI/detoxai},
}
```


## License
MIT License
