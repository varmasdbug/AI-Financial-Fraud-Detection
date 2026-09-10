
# 🔐 AI Financial Fraud Detection System

🚀 **Live Demo:**  
https://ai-financial-fraud-detection-nqscgt7lya3vprjj8nwpqd.streamlit.app

Machine Learning powered financial transaction fraud detection system using Logistic Regression and Streamlit.

This is a machine learning project where I worked on detecting fraudulent transactions.  
The dataset comes from Kaggle: [Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download).  
It’s a large dataset (over 6 million rows), and only a very small percentage of the transactions are fraud.  

---

## What I did
- Explored the data to see how fraud is distributed  
- Looked at transaction types (fraud mainly happens in `TRANSFER` and `CASH_OUT`)  
- Visualized transaction amounts, time steps, and balance changes  
- Trained a Logistic Regression model with scikit-learn  
- Used `class_weight="balanced"` to deal with the imbalance  
- Evaluated the model with precision, recall, F1-score, ROC-AUC, and a confusion matrix  
- Built a small Streamlit app to input transaction details and get predictions  

---

## How to run
1. Download the dataset from the Kaggle link above and put it in the project folder  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
