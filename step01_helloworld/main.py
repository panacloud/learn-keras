from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # unpacking tuple of four numpy arrays

network = Sequential()
network.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils.np_utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

print("**************** Test *********************")

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)








