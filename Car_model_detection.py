#Imports
import os
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

#DataGen

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col='relative_im_path',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col='relative_im_path',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
