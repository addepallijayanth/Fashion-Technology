from google.colab import drive
import pandas as pd

# Connect to Google Drive
drive.mount('/content/drive')

# Load the training and test datasets
train_data = pd.read_csv('/content/drive/MyDrive/fashion-mnist_train[1].csv')
test_data = pd.read_csv('/content/drive/MyDrive/fashion-mnist_test[1].csv')

X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values

X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

# Normalize pixel values (from 0-255 to 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape data for input into the CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
