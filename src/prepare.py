import pandas as pd
from sklearn.model_selection import train_test_split
import os


df = pd.read_csv('data/raw/churn.csv')
df = df.fillna(df.median(numeric_only=True))
train, test = train_test_split(df, test_size=0.2, random_state=42)


os.makedirs('data/processed', exist_ok=True)
train.to_csv('data/processed/train.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)