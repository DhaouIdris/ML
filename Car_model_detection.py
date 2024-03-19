#Imports
import os
import pandas as pd



# Load datasets
train_df = pd.read_csv('train_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')

#The class in training and test were not really consistent, with negative values. changing them ranging from 0 to 196
train_df['class'] = train_df['class'].apply(lambda x: x - 1 if x > 0 else x + 255)
train_df['class'] = train_df['class'].astype(str)
test_df['class'] = test_df['class'].apply(lambda x: x - 1 if x > 0 else x + 255)
test_df['class'] = test_df['class'].astype(str)

# Save updated datasets
train_df.to_csv('updated_train.csv', index=False)
test_df.to_csv('updated_test.csv', index=False)


