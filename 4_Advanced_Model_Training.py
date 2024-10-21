from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model with SGD optimizer
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.99),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train and evaluate the model
history = model.fit(partial_train_images, partial_train_labels, epochs=5, batch_size=32, validation_data=(val_images, val_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")
