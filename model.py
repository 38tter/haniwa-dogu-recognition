from keras import layers, models, optimizers
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(2,activation="sigmoid"))
# number of category

model.summary()

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])

categories = ["埴輪", "土偶"]
nb_classes = len(categories)

X_train, X_test, Y_train, Y_test = np.load("./data/haniwa_dogu_data.npy")

X_train = X_train.astype("float") / 255
X_test = X_test.astype("float") / 255

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

model = model.fit(X_train,
                  Y_train,
                  epochs=20,
                  batch_size=6,
                  validation_data=(X_test,Y_test))

acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('pdf/acc.pdf')

plt.figure()

plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='valodation loss')
plt.title('training and validation loss')
plt.legend()
plt.savefig('pdf/loss.pdf')
