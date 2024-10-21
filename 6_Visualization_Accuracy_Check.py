import matplotlib.pyplot as plt

# Visualize one of the images in the training set
plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
plt.title(f'Label: {y_train[0]}')
plt.show()

# After training, print accuracy and loss for each epoch
for epoch in range(5):
    print(f"Epoch {epoch+1}")
    print(f"Training Loss: {history.history['loss'][epoch]}, Training Accuracy: {history.history['accuracy'][epoch]}")
    print(f"Validation Loss: {history.history['val_loss'][epoch]}, Validation Accuracy: {history.history['val_accuracy'][epoch]}")
