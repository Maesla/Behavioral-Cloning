import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    #name = './IMG/'+batch_sample[0].split('/')[-1]
                    name = batch_sample[i].split('/')[-1]
                    #print(name)
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    correction_factor = 0.2
                    if(i == 0):#center
                        angle += 0.0*correction_factor
                    elif(i == 1): #left
                        angle += 1.0*correction_factor
                    else: #right
                        angle += -1.0*correction_factor

                    images.append(image)
                    angles.append(angle)


            # trim image to only see section with road

            augmented_images, augmented_angles = [],[]
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            #print("X_train_shape {}".format(X_train.shape))
            #print("Y_train_shape {}".format(y_train.shape))
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
ch, row, col = 3, 80, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5, subsample =(2,2),activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5, subsample =(2,2),activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5, subsample =(2,2),activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

print("samples_per_epoch  {}".format(len(train_samples)))
print("nb_val_samples {}".format(len(validation_samples)))



model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples)*3*2, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples)*3*2, nb_epoch=10)

### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
exit()




#First model
#model = Sequential()
#model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Flatten())
#model.add(Dense(1))

#Second Model (lenet)
##model = Sequential()
##model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
##model.add(Lambda(lambda x: x/255.0 - 0.5))
##model.add(Convolution2D(6,5,5,activation="relu"))
##model.add(MaxPooling2D())
##model.add(Convolution2D(6,5,5,activation="relu"))
##model.add(MaxPooling2D())
##model.add(Flatten())
##model.add(Dense(120))
##model.add(Dense(84))
##model.add(Dense(1))
