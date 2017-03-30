from keras.datasets import mnist

# Loading the MNIST (Mixed National Institute of Standards and Technology) dataset in Keras
# https://en.wikipedia.org/wiki/MNIST_database

(train_images, train_labels), (test_images, test_images) = mnist.load_data() # unpacking tuple of four numpy arrays

# The images above are encoded as Numpy arrays, and the labels are simply an array of digits, ranging from 0 to 9.

print("The Training Data")
print(train_images.shape)
print(len(train_images))

print("The Testing Data")
print(test_images.shape)
print(len(test_images))




