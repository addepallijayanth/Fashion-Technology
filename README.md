# Fashion MNIST Classification Using CNN

This project consists of only one main part:
1. **Fashion MNIST Classification using CNNs (Convolutional Neural Networks)** with various optimizers and batch size experiments.

---

## 1. Fashion MNIST Classification with CNNs

### Overview
The goal is to classify images from the Fashion MNIST dataset using Convolutional Neural Networks (CNNs). The dataset consists of 60,000 training images and 10,000 test images of fashion products, categorized into 10 classes. We explore multiple CNN architectures, experiment with different optimizers like Adam and SGD (Stochastic Gradient Descent), and change batch sizes to improve accuracy.

### Steps in the Code:
1. **Load Data from Google Drive**:
    - The training and test datasets are loaded from Google Drive into a Pandas DataFrame.
    - The images are normalized (pixel values scaled between 0 and 1) and reshaped to fit the CNN input format.

2. **Build a CNN Model**:
    - The model consists of two Conv2D layers with ReLU activations and MaxPooling layers.
    - The final layers include Dense (fully connected) layers with softmax for multi-class classification.

3. **Compile and Train the Model**:
    - The model is compiled using `Adam` or `SGD` optimizers.
    - Training occurs with different optimizers and batch sizes to assess their effect on performance.
    - Each model is validated on the test set.

4. **Results**:
    - The model's accuracy and loss are printed after training for each optimizer (Adam/SGD), with and without momentum, and different batch sizes.
    - A larger CNN model with doubled channels and dense units is also implemented and evaluated.

### Key Functions:
- **model.fit**: Trains the CNN model on the dataset.
- **model.evaluate**: Evaluates the trained model on the test dataset and prints accuracy and loss metrics.
- **SGD**: Stochastic Gradient Descent optimizer, used with and without momentum to compare training performance.
- **Adam**: An adaptive learning rate optimizer used for comparison.

### Plots:
- Training and validation loss/accuracy are printed after each epoch to visualize model performance over time.

## Conclusion
This project provides hands-on experience in:
- Building and training CNNs for image classification.
- Exploring optimizer and batch size effects on model performance.
