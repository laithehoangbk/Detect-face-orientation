import numpy as np
from keras.datasets.mnist import load_data
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

num_classes = 3
image_height, image_width, num_channels = 160, 160, 3

'''
Build CNN architecture
'''
def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(image_height, image_width, num_channels)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2))) 
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = build_model()
print(model.summary())

# create data generator
datagen = ImageDataGenerator(rescale=1.0/255.0)

# prepare iterators
train_it = datagen.flow_from_directory('./data_training/train/',
	class_mode='categorical', batch_size=64, target_size=(160, 160))
test_it = datagen.flow_from_directory('./data_training/valid/',
	class_mode='categorical', batch_size=64, target_size=(160, 160))

# fit model
history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
	validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=1)

# evaluate model
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
print('> %.3f' % (acc * 100.0))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("weight.h5")
print("Saved model to disk")


# results = model.fit(train_data2, train_labels_cat2, 
#                     epochs=15, batch_size=64,
#                     validation_data=(val_data, val_labels_cat))

# test_loss, test_accuracy = model.evaluate(test_data, test_labels_cat, batch_size=64)
# print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))