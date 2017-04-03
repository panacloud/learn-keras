from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils.np_utils import to_categorical

csv_all_data = np.genfromtxt('data/cs-training.csv', delimiter=",")
csv_predict = np.genfromtxt('data/cs-test.csv', delimiter=",")
#print(csv[1])
csv_all_data = csv_all_data[1:-1,:] #remove first name row
csv_all_data = csv_all_data[:,1:] #remove first index column

testsize = 30000
all_training_data = csv_all_data[0:len(csv_all_data)-testsize,:]
all_test_data = csv_all_data[len(csv_all_data)-testsize:len(csv_all_data),:]

train_labels = all_training_data[:,0] #slice the first column which are the labels
train_data = all_training_data[:,np.arange(1,11)]
train_labels = to_categorical(train_labels)

test_labels = all_test_data[:,0] #slice the first column which are the labels
test_data = all_test_data[:,np.arange(1,11)]
test_labels = to_categorical(test_labels)


network = Sequential()
network.add(Dense(10, activation='relu', input_shape=(10,)))
network.add(Dense(1, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

network.fit(train_data, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_data, test_labels)

print('test_acc:', test_acc)
