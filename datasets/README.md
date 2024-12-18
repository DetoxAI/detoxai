## To fetch the datasets
1. Run `download.py`. It will discover all directories in `datasets/catalog`, download them and properly structure, given that they have `links.yaml` file and `handler.py` file.

## To add a new dataset you need to
1. Create a folder under `datasets/catalog/<dataset_name>`
2. In this folder add `links.yml`, the general structure of this file is as follows:
    ```yaml
    link1:
        url: https://drive.google.com/uc?id=<file_id>
        output: data.zip
        type: google_drive
    link2:
        url: https://drive.google.com/uc?id=<file_id>
        output: l_train.csv
        type: google_drive
    ```
    An exception is made for datasets from torchvision where you just put `torchvision` in `links.yaml` and in `handler.py` you just do your stuff.

3. You also need to add `handler.py` which will handle the downloaded raw files and transform them into format used by our system. The format is as follows:
    ```
        <dataset_name>/
                        -> label_names.yaml
                        -> labels.csv
                        -> data/
                                -> 0.jpg
                                -> 1.jpg
                                -> ...
    ```

    `label_names.yaml`:
    ```yaml
    attribute1:
        0: label1
        1: label2
        2: label3
        <...>
    attribute2:
        0: label1
        1: label2
    <...>
    ```

    `labels.csv`:
    ```
    image_id, attribute1, attribute2, <...>
    0.jpg, 0, 1, <...>
    1.jpg, 1, 0, <...>
    <...>
    ```
