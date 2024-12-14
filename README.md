# ACM TensorFlow Training

This training package contains the code and instructions for the ACM TensorFlow Training project.

## Table of Contents

Installation
Usage
Script Descriptions
Folder Descriptions
Additional Python File Descriptions
Acknowledgements

## Installation

To run the code in this repository, you'll need to have Anaconda installed. If you don't have Anaconda installed, you can download it from the official website: [Anaconda](https://www.anaconda.com/products/individual).

Once you have Anaconda installed, follow these steps to create a virtual environment for TensorFlow v2.15:

1. Open a terminal or command prompt.
2. Create a new virtual environment by running the following command:
    ```
    conda create -n tf_env python=3.11
    ```
3. Activate the virtual environment by running the following command:
    ```
    conda activate tf_env
    ```
4. Install TensorFlow v2.15 and the recommended libraries by running the following commands:
    ```
    pip install tensorflow==2.15 # Note: Installing a different version of TensorFlow may introduce compatibility issues.
    pip install numpy==1.24.4
    pip install matplotlib==3.9.1
    pip install scipy==1.14.0
    ```
    You can verify the installed versions by running `pip list` in the activated virtual environment.

## Usage

To use the code in this repository, follow these steps:

1. Navigate to the project directory.
2. Activate the virtual environment by running the following command:
    ```
    conda activate tf_env
    ```
3. Run the desired scripts.

## Script Descriptions

- `training_FCN.py`: This script is used for training the model. It uses a set of input data and corresponding labels to adjust the model parameters for better data fitting through an optimization algorithm. It likely uses a deep learning framework like TensorFlow or PyTorch to define and train the model.

- `convertToTFLiteModel.py`: This script performs quantization by converting the trained model into TensorFlow Lite format(.tflite) and applying full integer quantization to convert the weights into int8 format.

- `convertNNModel.py`: This script is used to interpret the tflite model and generate corresponding `.h` and `.asm` files for deploying the model to MCU firmware. It uses an interpreter to parse the structure and parameters of the tflite and prints them to header files for running the model on the MCU.

- `validate_FCN_plotPickle.py`: This script is used to validate the trained FCN (Fully Convolutional Network) model using the test data. It loads the trained model and the test data from pickle files and performs inference on the test data. It then plots the predicted labels and the ground truth labels for visual comparison.

## Folder Descriptions

- `config` folder: This folder is used to generate the necessary header files (.h) and assembly files (.asm) for the MCU firmware. The acm_nn_model.asm files are used to store the model in the flash memory in Vulcan at the physical address 0x10000.

- `data` folder: This folder contains the images of 30 subjects' finger/thumb/palm, each with a size of 13x13. The images are stored as pickle files for TensorFlow to use during training.

- `best_model` folder: After running `training_FCN.py`, this folder will be created and contains the trained model in the form of `.pb` file. The `.pb` file represents the TensorFlow SavedModel format, which includes the model architecture and the learned weights. This trained model can be used for inference or further analysis.

## Additional Python Descriptions 

- `models.py`: This file contains the implementation of various models for image classification using TensorFlow. It provides functions to create dense models for image classification tasks.

- `losses.py`: This file contains the implementation of the pairwise_cross_entropy_alpha loss function, which is used in the models.py file for image classification tasks. The loss function calculates the cross-entropy loss between the predicted labels and the true labels, with an additional alpha parameter to control the weight of positive samples in the loss calculation. It also includes the `CosineLearningRate` class, which is a callback used to implement a cosine learning rate schedule during training. The `CosineLearningRate` class adjusts the learning rate based on the current epoch and the total number of epochs. It gradually decreases the learning rate from the initial value to a minimum value using a cosine function.


## Acknowledgements

- This project was developed in collaboration with Karthik Shanmuga Vadivel from Synaptics AI team.

- Intellectual Property “SYSTEM AND METHOD FOR NEURAL NETWORK BASED TOUCH CLASSIFICATION IN A TOUCH SENSOR”, 230137US01, 2024“ 


For questions or feedback, please contact Carl Yang (carl.yang@synaptics.com).