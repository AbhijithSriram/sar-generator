import pandas as pd
import numpy as np

# Load IBM AML data
accounts = pd.read_csv('data/ibm_aml/accounts.csv')
transactions = pd.read_csv('data/ibm_aml/transactions.csv')
alerts = pd.read_csv('data/ibm_aml/alerts.csv')

print(f"Accounts: {len(accounts):,}")
print(f"Transactions: {len(transactions):,}")
print(f"Alerts: {len(alerts):,}")
print(f"\nAccount columns: {accounts.columns.tolist()}")
print(f"\nSample accounts:")
print(accounts.head())
print(f"\nTransaction columns: {transactions.columns.tolist()}")
print(f"\nSample transactions:")
print(transactions.head())
print(f"\nAlert columns: {alerts.columns.tolist()}")
print(f"\nSample alerts:")
print(alerts.head())