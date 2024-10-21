from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Reshape and normalize the images
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255

# One-hot encode the labels
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Split training data into training and validation sets
validation_split = 0.1
num_val_samples = int(validation_split * train_images.shape[0])
val_images = train_images[:num_val_samples]
val_labels = train_labels[:num_val_samples]
partial_train_images = train_images[num_val_samples:]
partial_train_labels = train_labels[num_val_samples:]
