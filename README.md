# SMS Spam Detection System

This repository contains a machine learning pipeline built for classifying SMS messages as either spam or ham (non-spam). The system applies natural language processing (NLP) and ensemble learning techniques to improve classification performance.

## Overview

The core of the project is a Jupyter Notebook (`model.ipynb`) that walks through the complete workflow, including:

- Data loading and cleaning
- Text preprocessing (tokenization, stemming, stop word removal, etc.)
- Feature extraction using TF-IDF
- Model training and evaluation
- Implementation of ensemble models: Voting Classifier and Stacked Generalization
- Visualization of results with Matplotlib and Seaborn

## Repository Contents

- `model.ipynb`: Jupyter Notebook containing the entire end-to-end implementation, from preprocessing to model evaluation.
- `model.pkl`, `vectorizer.pkl`, `voting_classifier.pkl`: Serialized files for quick model deployment without needing to retrain.
- `spam.csv`: Dataset used for training and testing.

## Libraries and Tools

The project uses the following Python libraries:

- `pandas`, `NumPy` — for data manipulation and analysis
- `NLTK` — for natural language processing
- `scikit-learn` — for model building and evaluation
- `Matplotlib`, `Seaborn` — for data visualization

## Models Used

Three base classifiers are trained individually and as part of ensemble methods:

- Multinomial Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

### Ensemble Methods

- **Voting Classifier**: Combines predictions from all base models using hard voting.
- **Stacking Classifier**: Uses base model predictions as input for a meta-model (Logistic Regression).

## Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Visual tools like confusion matrices and word clouds

Stacked Generalization showed improved performance compared to individual models and the Voting Classifier.

## Future Work

Potential improvements include:

- Graphical User Interface
- Integrating word embeddings (e.g., Word2Vec or BERT)
- Incorporating message metadata as features
- Hyperparameter optimization for each model
