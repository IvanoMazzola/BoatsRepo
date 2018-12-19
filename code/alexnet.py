import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import os
import sys
import cv2

np.random.seed(1000)
models_dir = './models'
TrainingSetPath = './BoatsRepo/ReducedDataSet/TrainingSet'
TestSetPath = './BoatsRepo/ReducedDataSet/TestSet'

def buildTrainingSet(TrainingSetPath):
    labels_name = next(os.walk(TrainingSetPath))[1]   # Lista delle directory in 'path'
    filename_list = list()
    label_list = list()
    classes = dict()
    class_id = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    index = -1
    for class_name in labels_name:
        current_id = class_id.copy()
        index += 1
        current_id[index] += 1
        classes[class_name] = current_id
        img_list = os.listdir(TrainingSetPath + "/" + class_name) # entro nella directory della classe corrente e listo tutte le immagini
        #remove hidden files
        for elem in img_list:
            if(elem.startswith('.')):
                img_list.remove(elem)
        for el in img_list:
            filename_list.append(TrainingSetPath + "/" + class_name + "/" + el)
            #label_list.append(current_id)
            label_list.append(index)
    return [filename_list, label_list, classes]

def buildTestSet(TestSetPath):
    labels_name = next(os.walk(TestSetPath))[1]   # Lista delle directory in 'path'
    filename_list = list()
    label_list = list()
    classes = dict()
    class_id = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    index = -1
    for class_name in labels_name:
        current_id = class_id.copy()
        index += 1
        current_id[index] += 1
        classes[class_name] = current_id
        img_list = os.listdir(TestSetPath + "/" + class_name) # entro nella directory della classe corrente e listo tutte le immagini
        #remove hidden files
        for elem in img_list:
            if(elem.startswith('.')):
                img_list.remove(elem)
        for el in img_list:
            filename_list.append(TestSetPath + "/" + class_name + "/" + el)
            #label_list.append(current_id)
            label_list.append(index)
    return [filename_list, label_list, classes]

def training(TrainingSetPath, num_classes):
    Xtrain = []
    Ytrain = []
    train_filename_list, train_label_list, train_classes = buildTrainingSet(TrainingSetPath)
    train_size = len(train_filename_list)
    num_classes = max(train_label_list)+1
    for i in range(train_size):
        img = cv2.imread(train_filename_list[i], cv2.IMREAD_COLOR)
        lab = train_label_list[i]
        img_resize = cv2.resize(img,(224,224))
        Xtrain.append(img_resize)
        Ytrain.append(lab)

    Xtrain = np.array(Xtrain)
    #Xtrain = Xtrain.astype('float32')
    #Xtrain /=255
    Ytrain = np.array(Ytrain)
    Ytrain = keras.utils.to_categorical(Ytrain, num_classes)
    return Xtrain,Ytrain

def test(TestSetPath,num_classes):
    Xtest = []
    Ytest = []
    test_filename_list, test_label_list, test_classes = buildTestSet(TestSetPath)
    test_size = len(test_filename_list)
    for i in range(test_size):
        img = cv2.imread(test_filename_list[i], cv2.IMREAD_COLOR)
        lab = test_label_list[i]
        img_resize = cv2.resize(img,(224,224))
        Xtest.append(img_resize)
        Ytest.append(lab)

    Xtest = np.array(Xtest)
    #Xtest = Xtest.astype('float32')
    #Xtest /=255
    Ytest = np.array(Ytest)
    Ytest = keras.utils.to_categorical(Ytest, num_classes)
    return Xtest,Ytest

def AlexNet(input_shape, num_classes):

    # (3) Create a sequential model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape = input_shape, kernel_size=(3,3),\
     strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()

    # (4) Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam',\
     metrics=['accuracy'])

    return model

def main():
    x_train, y_train = training(TrainingSetPath,3)
    x_test, y_test = test(TestSetPath,3)
    input_shape = (x_train.shape[1], x_train.shape[2], 3)
    model = AlexNet(input_shape,3)
# (5) Train
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1, \
    validation_split=0.2, shuffle=True)
    try:
        score = model.evaluate(x_test, y_test)
        print("Test loss: %f" %score[0])
        print("Test accuracy: %f" %score[1])
    except KeyboardInterrupt:
        pass
    #model.predict(x_test[:5])

main()
