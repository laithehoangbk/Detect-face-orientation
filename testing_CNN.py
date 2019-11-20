import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import load_model

if __name__ == "__main__":

    CLASSES = ['frontal', "leftside", 'rightside']
    # load json and create model
    with open('model/model.json', 'r') as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("model/weight.h5")

    img = cv2.imread('data_test/2.jpg')

    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    cv2.imshow('show', img)
    
    img = np.expand_dims(img, 0)
    print(img.shape)
    img = img / 255.0

    y_pred = model.predict(img)
    print(y_pred)
    labels = np.argmax(y_pred, axis = 1)[0]

    print(CLASSES[labels])

    cv2.waitKey(0)
pihfkdfkd
