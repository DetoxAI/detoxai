Adding a New Method to DetoxAI
=====================================

In this tutorial, we will show you how to add a new method to DetoxAI.

**Files to edit:**

* ``methods/your_method.py``
* ``methods/__init__.py``
* ``core/interface.py``

``methods/your_method.py``:
---------------------------

- Implement your new method:
    - Inherit from ``ModelCorrectionMethod``
    - Implement the abstract ``apply_model_correction`` method, make sure it has ``**kwargs`` in the constructor to facilitate flushing unused arguments (yes, we know this is not the best practice, but it is a compromise we made to make the API more flexible)

.. note::
   Technically, this is all you need to do, but we recommend checking out already implemented methods to see how they are structured and what ideas and workarounds we used to introduce hooks, logging, and all that fancy stuff.

``methods/__init__.py``:
------------------------

- Import your new method to make it visible in the package.

``core/interface.py``:
-----------------------

- Import your new method at the top to make it available in the interface script.
- Add to ``_method_mapping`` dictionary. This maps the method name to the class name.
- Add an entry in ``DEFAULT_METHODS_CONFIG`` and set a default config to your method, if applicable. There has to be an entry, but it can be empty if your method does not have any default config.