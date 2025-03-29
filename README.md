Speech Emotion Recognition

This project implements a Speech Emotion Recognition (SER) system using TensorFlow and Keras with a custom Convolutional Neural Network (CNN) to classify emotions from speech audio data.

Project Overview

Speech Emotion Recognition identifies human emotions from speech signals. This model analyzes audio clips and classifies them into different emotion categories using deep learning techniques.

Key Features

Built a custom CNN model for emotion classification.

Used Librosa for audio preprocessing and feature extraction (MFCCs).

Classifies speech into multiple emotions like happy, sad, angry, and neutral.

Improved model accuracy through hyperparameter tuning.

Technologies Used

Python

TensorFlow and Keras

Librosa (for audio processing)

NumPy, Pandas (for data handling)

Matplotlib, Seaborn (for visualization)

Dataset

The model is trained on the RAVDESS dataset, which includes speech recordings labeled with various emotions.

Model Architecture

The custom CNN extracts temporal and spectral features from audio. The architecture includes:

Convolutional Layers for feature extraction

Batch Normalization for faster convergence

Dropout Layers to reduce overfitting

Dense Layers for emotion classification

How to Run the Project

Clone the repository:

git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition

Install required packages:

pip install -r requirements.txt

Run the model on Google Colab. Link: Open in Colab

Execute the script:

python train_model.py

Results

Achieved high accuracy through hyperparameter tuning.

Successfully classified emotions like happy, sad, angry, and neutral.

Future Improvements

Improve accuracy with advanced architectures (e.g., LSTM-CNN hybrid).

Add real-time emotion recognition.

Train on larger and diverse datasets.
