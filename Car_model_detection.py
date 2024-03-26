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


#Object detection 
#Model Loading
model1 = YOLO('yolov8n.pt')

#calculate the area for a box
def calculate_box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

#choosing only the biggest box given a list of boxes by size
def filter_largest_box(boxes):
    if not boxes:
        return None

    biggest_box = boxes[0]
    max_area = calculate_box_area(biggest_box)

    for box in boxes[1:]:
        area = calculate_box_area(box)
        if area > max_area:
            max_area = area
            biggest_box = box

    return biggest_box


#drawing the box on the jpg
def draw_bounding_box(image_path, box, title):
    image = plt.imread(image_path)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )


    ax.add_patch(rect)

    plt.title(title)
    plt.show()


#Loading the dictionnary for the cars models name
with open('class_names_dict.pkl', 'rb') as file:
    models = pickle.load(file)

print(list(models.keys())[list(models.values()).index(58)])


img = train_df['relative_im_path'][100]
results = model1(img, show = False)
boxes = results[0].boxes.xyxy.tolist()
largest_box = filter_largest_box(boxes)
print(boxes)
print(largest_box)

draw_bounding_box(img, largest_box,f"{list(models.keys())[list(models.values()).index(train_df['class'][100])]}")



def main(img1, img2):

  img1=image.load_img('img1',target_size=(224,224))
  img2=image.load_img('img2',target_size=(224,224))
  img1=image.img_to_array(img1)/255
  img2=image.img_to_array(img2)/255

  img1=np.expand_dims(img1,axis=0)
  img1_data=preprocess_input(img1)
  img1_data.shape
  img2=np.expand_dims(img2,axis=0)
  img2_data=preprocess_input(img2)
  img2_data.shape


  if np.argmax(model.predict(img1_data), axis=1)==np.argmax(model.predict(img2_data), axis=1):
    print(f'The cars are the same model: {list(models.keys())[list(models.values()).index(np.argmax(model.predict(img1_data), axis=1)+1)]}') #We ranges the labels from 0 to 195 but it is from 1 to 196 in the dictionnary
  else:
    print('The cars are different models')

  results1 = model1(img1, show = False)
  boxes = results[0].boxes.xyxy.tolist()
  largest_box = filter_largest_box(boxes)
  draw_bounding_box(img, largest_box, f'First car model :{list(models.keys())[list(models.values()).index(np.argmax(model.predict(img1_data), axis=1)+1)]}')
  print("Coordinates of the box of the first car:")
  print(f"x1: {largest_box[0]}, y1: {largest_box[1]}")
  print(f"x2: {largest_box[2]}, y2: {largest_box[3]}")

  results = model1(img1, show = False)
  boxes = results[0].boxes.xyxy.tolist()
  largest_box = filter_largest_box(boxes)
  draw_bounding_box(img, largest_box, f'Second car model :{list(models.keys())[list(models.values()).index(np.argmax(model.predict(img2_data), axis=1)+1)]}')
  print("Coordinates of the box of the second car:")
  print(f"x1: {largest_box[0]}, y1: {largest_box[1]}")
  print(f"x2: {largest_box[2]}, y2: {largest_box[3]}")
