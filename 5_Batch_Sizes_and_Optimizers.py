# Re-train with a batch size of 8
history_small_batch = model.fit(partial_train_images, partial_train_labels, epochs=5, batch_size=8, validation_data=(val_images, val_labels))

# Re-compile the model with SGD without momentum
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.0),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Re-train the model without momentum
history_no_momentum = model.fit(partial_train_images, partial_train_labels, epochs=5, batch_size=32, validation_data=(val_images, val_labels))

# Compile the model with Adam optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.99, beta_2=0.999),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using Adam optimizer
history_adam = model.fit(partial_train_images, partial_train_labels, epochs=5, batch_size=32, validation_data=(val_images, val_labels))
