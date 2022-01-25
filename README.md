# Rock, paper, scissors

This repository was made for a school project in Artificial intelligence and is a proof of concept for image classification using convolutional neural networks (CNNs) with the help of the Keras machine learning framework.

## Setup

In order to run the scripts from this repository it is necessary to install the required dependencies. All of the dependencies are listed in `requirements.txt` and can be installed using `pip`. The easiest way to get started is making a [conda](https://docs.conda.io/en/latest/) environment:

- `conda create --name <env_name> python==3.8`
- `conda activate <env_name>`
- `pip install -r requirements.txt`

## Datasets

For training and testing purposes, we have used two publically available datasets, as well as our own dataset. All of them can be found and downloaded on our [Google Drive](https://drive.google.com/drive/folders/11EgWYgV9urQpSIzBbi4YyrE5dm7OEMmH). We recommend using `dataset2` for training purposes.

## Running

### Training

To run training, simply run the `train.py` script and pass the training and validation directory paths. The directory structure should be the same as the ones you can download from our [Google Drive](https://drive.google.com/drive/folders/11EgWYgV9urQpSIzBbi4YyrE5dm7OEMmH) - each class should have its own subdirectory. Running `python train.py -h` will display the help dialog, such as the one below, showing you all of the possible arguments you could pass to the training script.

```
python train.py -h
usage: train.py [-h] [-e EPOCHS] [-b BATCH_SIZE] [-m MODEL] [--cp CP] train_data_path val_data_path

This module is used for training the Rock, paper, scissors model.

positional arguments:
  train_data_path       The path to your training data directory, which has 3 subdirectories correspoding to the rock, paper and scissors class.
  val_data_path         The path to your validation data directory, which has 3 subdirectories correspoding to the rock, paper and scissors class.

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs for training the model, defaults to 25.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size used for training, defaults to 16.
  -m MODEL, --model MODEL
                        Name of the model used for saving the model in h5 format in the models directory after training is finished, defaults to 'model'.
  --cp CP               Name of the MobileNetv2 model already saved in h5 format in the models directory from which to continue training, defaults to   None, which means a randomly initialized model will be instantiated.
  ```


### Testing

Testing is available both from a data directory, as during training, or using your camera. Similar to the training script, you can either run `python testing.py -d <data_dir>` to test from a data directory or `python testing.py -c`. One of these parameters must be passed, while the camera one takes presedence. 

```
python testing.py -h                                                                     
usage: testing.py [-h] [-d DIR] [-c] [-m MODEL]

This module is used for testing the Rock, paper, scissors model either with pictures from disk or with camera.

optional arguments:
  -h, --help            show this help message and exit
  -d DIR, --dir DIR     The path to your validation data directory, which has 3 subdirectories correspoding to the rock, paper and scissors class. This argument must be passed   
                        if the -c flag is not set.
  -c, --camera          Whether to use the camera for testing.
  -m MODEL, --model MODEL
                        Name of the model in h5 format in the models directory to load for testing, defaults to 'model'.
```

### Running the game

To run the game, simply run `python RPS_game.py` and wait for the screen and the camera to load. The start the game press the `SPACE` key. Once you've had enough of rock, paper, scissors, press the `ESC` key. Optionally, you can pass the model name through the `-m` parameter.