Using internal tools for datasets
==================================

In this tutorial, we will show you how to use the internal tools for datasets in DetoxAI. 
Note, that you don't actually have to ever use them if you are using the library as a regular user.
The ``detoxai.debias`` interface does not require you to use internal tools for datasets.
You can just pass your regular torch dataloader to the ``detoxai.debias`` function and it will work just fine, 
as long as it returns batches of (image, label, protected_attribute) tuples.

However, if you are a developer who wants to do experiments or some other crazy stuff within the library,
you might want to use the internal tools for datasets.
They come in handy for experiments and for downloading datasets on the target machines (e.g., they were incredible useful for us 
in our ClearML-based experiments and running experiments on multiple machines).


Download supported datasets 
------------------------------

Run ``download.py``. 
It will discover all directories in ``datasets/catalog``, download them and properly structure them, given that they have a ``links.yaml`` file and a ``handler.py`` file.

At the moment, DetoxAI is shipped with the following datasets supported:

-   CelebA
-   FairFace
-   Cifar10
-   Cifar100
-   Caltech101


Add a new dataset 
-------------------

1.  Create a folder under ``datasets/catalog/<dataset_name>``

2.  In this folder add ``links.yml``. In there, you want to put all the download links that let you fetch 
the desired dataset. The general structure of this file is as follows:

.. code-block:: yaml    

    link1:
       url: [https://drive.google.com/uc?id=](https://drive.google.com/uc?id=)<file_id>
       output: data.zip
       type: google_drive
    link2:
       url: [https://drive.google.com/uc?id=](https://drive.google.com/uc?id=)<file_id>
       output: l_train.csv
       type: google_drive

An exception is made for datasets from torchvision where you just put ``torchvision`` in ``links.yaml`` as it has 
a pretty standard interface, so it is easy to handle it. 

Here are a few examples. 

For CelebA:

.. code-block:: yaml

    link1:
    url: https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset
    output: kaggle.zip
    type: curl

For Cifar10:

.. code-block:: yaml
    
    torchvision

For FairFace:

.. code-block:: yaml

    link1:
    url: https://drive.google.com/uc?id=1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86
    output: data.zip
    type: google_drive
    link2:
    url: https://drive.google.com/uc?id=1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH
    output: l_train.csv
    type: google_drive
    link3:
    url: https://drive.google.com/uc?id=1wOdja-ezstMEp81tX1a-EYkFebev4h7D
    output: l_val.csv
    type: google_drive


3.  You also need to implement a ``handler.py`` script, which will handle all the downloaded raw files 
and transform them into the format used by our system. The target format is as follows:

.. code-block:: text

    <dataset_name>/
       -> label_names.yaml
       -> labels.csv
       -> data/
          -> 0.jpg
          -> 1.jpg
          -> ...

``label_names.yaml``

.. code-block:: yaml

    attribute1:
       0: label1
       1: label2
       2: label3
       <...>
    attribute2:
       0: label1
       1: label2
       <...>


``labels.csv``

.. code-block:: text

    image_id, attribute1, attribute2, <...>
    0.jpg, 0, 1, <...>
    1.jpg, 1, 0, <...>
    <...>


There might be various ways to implement the ``handler.py`` script, and it depends on the dataset you are using.
We highly recommend to check out the ``handler.py`` scripts for the datasets we already support. 

In case you don't want to browse the repo for yourself, here is one of them for CelebA, we have the following implementation:

.. code-block:: python

    import os
    import shutil
    import zipfile
    import pandas as pd
    import yaml

    home = os.environ.get("DETOXAI_DATASET_PATH", os.path.expanduser("~"))
    directory = os.path.join(home, "celeba")

    tmp_directory = os.path.join(directory, "tmp")
    data_path = os.path.join(tmp_directory, "kaggle.zip")


    # Extract data to directory
    with zipfile.ZipFile(data_path, "r") as zip_ref:
        zip_ref.extractall(tmp_directory)
        print("Done")

    # Read csvs
    df = pd.read_csv(os.path.join(tmp_directory, "list_attr_celeba.csv"))

    # Transform all -1 to 0
    df = df.replace(-1, 0)

    # Create mapping for each attribute 1 - present, 0 - not present
    mapping = {column: {1: "present", 0: "not present"} for column in df.columns[1:]}


    mapping_path = os.path.join(directory, "labels_mapping.yaml")
    with open(mapping_path, "w") as f:
        yaml.dump(mapping, f)

    # Save concatenated csv
    df.to_csv(os.path.join(directory, "labels.csv"), index=False)


    # Move data from tmp to directory under data/
    # it has now valid and train subdirectories but we don't need them
    # so we will move the files to the data directory

    data_directory = os.path.join(directory, "data")
    os.makedirs(data_directory, exist_ok=True)

    # Move files
    img_dir = os.path.join(tmp_directory, "img_align_celeba")
    for subdir, dirs, files in os.walk(img_dir):
        for file in files:
            source, extension = os.path.splitext(file)
            target = os.path.join(data_directory, f"{source.zfill(6)}{extension}")
            shutil.move(os.path.join(subdir, file), os.path.join(data_directory, target))

    # Remove tmp directory
    shutil.rmtree(tmp_directory)