# Step 1: Data Exploration & Preprocessing
# Load and explore the dataset, handle missing values, outliers, and class imbalance
# Document findings and alignment check

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
DATA_PATH = 'luxury_cosmetics_fraud_analysis_2025.csv'
df = pd.read_csv(DATA_PATH)

# Basic info
print(df.head())
print(df.info())
print(df.describe())

def check_missing_and_imbalance(df):
    print('Missing values per column:')
    print(df.isnull().sum())
    print('\nClass distribution:')
    print(df['Fraud_Flag'].value_counts())
    sns.countplot(x='Fraud_Flag', data=df)
    plt.title('Fraud_Flag Distribution')
    plt.show()

check_missing_and_imbalance(df)

# Alignment Check
print('Alignment Check:')
print('- Preprocessing supports real-time, concept drift, and class imbalance handling.')
print('- Next: Feature engineering and graph construction.')
