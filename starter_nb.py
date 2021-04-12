path = '../input/plant-pathology-2021-fgvc8/'
train_dir = path + 'train_images/'
test_dir = path + 'test_images/'

import pandas as pd
df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv", dtype=str)
df.labels.value_counts()

df['labels'] = df['labels'].astype(str)

from keras.preprocessing.image import ImageDataGenerator

train_datagen  = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.3)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                 validation_split = 0.3)

train_generator = train_datagen.flow_from_dataframe(dataframe = df,
                                                   directory = train_dir,
                                                   target_size = (150,150),
                                                   x_col = 'image',
                                                   y_col = 'labels',
                                                   batch_size = 256,
                                                   color_mode = 'rgb',
                                                   class_mode = 'categorical',
                                                   subset = 'training')

test_generator = test_datagen.flow_from_dataframe(dataframe = df,
                                                 directory = train_dir,
                                                 target_size = (150,150),
                                                 x_col = 'image',
                                                 y_col = 'labels',
                                                 batch_size = 256,
                                                 color_mode = 'rgb',
                                                 class_mode = 'categorical',
                                                 subset = 'validation')
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()

model.add(layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation = 'relu', input_shape = (150,150,3)))
model.add(layers.MaxPooling2D((2,2)))


model.add(layers.Conv2D(256, (3,3), activation = 'relu', input_shape = (150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(12, activation = 'softmax'))

optimizer = optimizers.Adam(lr = 0.001)

model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
model.summary()

history = model.fit(train_generator, epochs = 10, validation_data = test_generator)

#Plotting Results

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['validation_accuracy']
loss = history.history['loss']
val_loss = history.history['validation_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
