from setuptools import setup, find_packages

setup(
    name="debias",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "Pillow",
        "tqdm",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "pandas",
        "opencv-python",
        "wandb",
        "albumentations",
        "pytorch-lightning",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Specify the Python version compatibility
)
