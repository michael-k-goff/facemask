# Facemask Analysis

This project is based on Kaggle's [facemask dataset](https://www.kaggle.com/vijaykumar1799/face-mask-detection). The goal is to classify images of faces as to whether they have a fask on, don't have a mask on, or are wearing a mask incorrectly. The model is based on [this one](https://www.kaggle.com/brsdincer/mask-prediction-process-end-to-end).

See the Jupyter notebook `exploration.ipynb` to explore the data, or the python script `main.py` to run the compilation and testing of the model in one step.

Topics explored in the notebook include

- A neural network with convolutional layers and early stopping to perform image classification
- Training, test, and validation data sets
- Confusion matrices