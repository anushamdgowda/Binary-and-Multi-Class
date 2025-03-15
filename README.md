# Malayalam Fake News Detection: Binary Classification


This repository contains code for detecting fake news in Malayalam text using deep learning models. The code is implemented in Python and utilizes libraries such as TensorFlow and Keras.

## Project Overview
The project aims to classify Malayalam news articles as either real or fake. It combines data preprocessing, feature engineering, and deep learning models to achieve this goal. The main steps involved are:

1. **Data Loading:** Loading the training, development, and test datasets.
2. **Preprocessing:** Cleaning the text data by removing stop words, punctuation, and other irrelevant characters.
3. **Feature Engineering:** Converting text into numerical representations using FastText embeddings.
4. **Model Training:** Training deep learning models, including Bi-RNN, DNN, GRU, and LSTM + RNN, for classification.
5. **Ensemble Learning:** Combining the predictions of multiple models to improve overall accuracy.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- Pandas
- Scikit-learn
- NLTK
- TextBlob

## Model Details

- Bi-RNN: Bidirectional Recurrent Neural Network.
- DNN: Deep Neural Network.
- GRU: Gated Recurrent Unit.
- LSTM + RNN: Long Short-Term Memory combined with a Recurrent Neural Network.
- Fusion Model: An ensemble of the above models.

## Results

The models were evaluated on a development set and achieved promising results in terms of accuracy, precision, recall, and F1-score. The fusion model, in particular, showed significant improvement over individual models.

## Future Work

- Explore other embedding techniques, such as Word2Vec and GloVe.
- Experiment with different deep learning architectures, such as CNN and Transformer models.
- Fine-tune hyperparameters to optimize model performance.
- Evaluate the models on a larger and more diverse dataset.
- Develop a web application to make the fake news detection system accessible to users.

**Multi Class Classification**
Description

This is a brief description of your project. Explain the purpose and goals of your project. You can also mention the technologies and tools used.

### Prerequisites

List any software or libraries that need to be installed before running the project.
* Python 3.x
* TensorFlow 2.x
* Pandas
* NumPy


## Model Details

- Bi-RNN: Bidirectional Recurrent Neural Network.
- DNN: Deep Neural Network.
- GRU: Gated Recurrent Unit.
- LSTM + RNN: Long Short-Term Memory combined with a Recurrent Neural Network.
- Fusion Model: An ensemble of the above models.

## Results

The models were evaluated on a development set and achieved promising results in terms of accuracy, precision, recall, and F1-score. The fusion model, in particular, showed significant improvement over individual models.

## Future Work

- Explore other embedding techniques, such as Word2Vec and GloVe.
- Experiment with different deep learning architectures, such as CNN and Transformer models.
- Fine-tune hyperparameters to optimize model performance.
- Evaluate the models on a larger and more diverse dataset.
- Develop a web application to make the fake news detection system accessible to users.



