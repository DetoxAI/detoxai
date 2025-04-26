DetoxAI documentation
=====================

Welcome to the documentation for DetoxAI!

DetoxAI is a Python package for debiasing neural networks. 
It provides a simple and efficient way to remove bias from your models while maintaining their performance. 
The package is designed to be easy to use and integrate into existing projects. 

We hosted a website with a demo and an overview of the package, which can be found at `https://detoxai.github.io <https://detoxai.github.io>`_.
DetoxAI is also available on GitHub at `https://github.com/DetoxAI/detoxai <https://github.com/DetoxAI/detoxai>`_.


Getting Started
====================

DetoxAI is available on PyPI, and can be installed by running the following command:

.. code-block:: bash

   pip install detoxai


Then, import the library and debias your model using the api:

.. code-block:: python 

   import detoxai

   model = ... # your torch model
   dataloader = ... # your torch dataloader returning (image, label, prot. attr) tuples
   results = detoxai.debias(model, dataloader) 


To get started with DetoxAI, please refer to the following examples:

.. toctree::
   :maxdepth: 1
   :caption: Notebooks

   examples

We have also put together a bunch small tutorials to help you get started with DetoxAI in your own projects:

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   
   
   tutorials


**API Reference**

For detailed information on the library's functions and classes, see the API reference:

.. toctree::
   :maxdepth: 4
   :caption: API

   detoxai

**Contributing**

Interested in contributing to DetoxAI? Check out our contribution guidelines on GitHub.  

**License**

DetoxAI is licensed under the MIT License.

Index
==================
* :ref:`modindex`