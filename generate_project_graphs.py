import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
DATA_PATH = 'luxury_cosmetics_fraud_analysis_2025.csv'
df = pd.read_csv(DATA_PATH)

# 1. Fraud vs. Non-Fraud Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Fraud_Flag', data=df)
plt.title('Fraud vs. Non-Fraud Distribution')
plt.savefig('fraud_flag_distribution.png')
plt.close()

# 2. Missing Values per Column
plt.figure(figsize=(10,4))
df.isnull().sum().plot(kind='bar')
plt.title('Missing Values per Column')
plt.ylabel('Count')
plt.savefig('missing_values_per_column.png')
plt.close()

# 3. Purchase Amount Distribution
plt.figure(figsize=(6,4))
df['Purchase_Amount'].hist(bins=50)
plt.title('Purchase Amount Distribution')
plt.xlabel('Purchase Amount')
plt.ylabel('Frequency')
plt.savefig('purchase_amount_distribution.png')
plt.close()

# 4. Transactions by Hour of Day
if 'Transaction_Date' in df.columns and 'Transaction_Time' in df.columns:
    df['Transaction_Datetime'] = pd.to_datetime(df['Transaction_Date'] + ' ' + df['Transaction_Time'])
    df['hour'] = df['Transaction_Datetime'].dt.hour
    plt.figure(figsize=(8,4))
    df['hour'].value_counts().sort_index().plot(kind='bar')
    plt.title('Transactions by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Transactions')
    plt.savefig('transactions_by_hour.png')
    plt.close()

# 5. Transactions by Payment Method
plt.figure(figsize=(8,4))
sns.countplot(x='Payment_Method', hue='Fraud_Flag', data=df)
plt.title('Transactions by Payment Method and Fraud')
plt.xticks(rotation=45)
plt.savefig('transactions_by_payment_method.png')
plt.close()

# 6. Customer Age Distribution
if 'Customer_Age' in df.columns:
    plt.figure(figsize=(6,4))
    df['Customer_Age'].dropna().astype(float).hist(bins=20)
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('customer_age_distribution.png')
    plt.close()

# 7. Transactions by Location
if 'Location' in df.columns:
    plt.figure(figsize=(10,4))
    sns.countplot(x='Location', hue='Fraud_Flag', data=df)
    plt.title('Transactions by Location and Fraud')
    plt.xticks(rotation=45)
    plt.savefig('transactions_by_location.png')
    plt.close()

print('All graphs generated and saved.')
