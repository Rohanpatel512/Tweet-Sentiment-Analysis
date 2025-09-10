# Tweet-Sentiment-Analysis

## Description
This project implements **sentiment analysis on tweets** using a **logistic regression classifier built from scratch**
with **gradient descent**. The goal of this project is educational (understanding how sentiment analysis with logistic regression
works at a low level).

## Features
- Custom implementation of **Logistic Regression**:
  - Sigmoid activation
  - Binary Cross Entropy (BCE) loss
  - Gradient descent weight updates
- Preprocessing pipeline:
  - Lowercasing, tokenization, stemming
  - Positive/negative word frequency features

## Model Evaluation on Train Set
- Accuracy: 98.30%
- Precision: 100.00%
- Recall: 96.60%
- F1-Score: 98.27%

## Model Evaluation on Test Set
- Accuracy: 98.65%
- Precision: 100.00%
- Recall: 97.30%
- F1-Score: 98.63%

## ⚙️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/Rohanpatel512/Tweet-Sentiment-Analysis.git
cd Tweet-Sentiment-Analysis
pip install -r requirements.txt
