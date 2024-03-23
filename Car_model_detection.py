#Imports
import os
import pandas as pd
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
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


#Classification

#Model config
IMAGE_SIZE = [224, 224]
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

#Transfer Learning

#freeze the base layers:
for layer in vgg.layers:
    layer.trainable = False

# Add new layers on top of the pretrained base
x = Flatten()(vgg.output)

prediction = Dense(196, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

print(model.summary())

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


#Training 

r = model.fit_generator(
  train_generator,
  validation_data=test_generator,
  epochs=30,
  steps_per_epoch=len(train_generator),
  validation_steps=len(test_generator)
)

#Results Analysis
# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


#Saving the model
model.save('model_vgg.h5')





