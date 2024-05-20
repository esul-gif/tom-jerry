from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Adjust the number of units according to the number of classes
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Assuming you have your data in train_directory and validation_directory
train_generator = train_datagen.flow_from_directory(
        train_directory,  # This is the target directory
        target_size=(128, 128),  # All images will be resized to 150x150
        batch_size=32,
        class_mode='categorical')  # Since we use categorical_crossentropy loss, we need categorical labels

# Fit the model
history = model.fit(
      train_generator,
      steps_per_epoch=100,  # batches in the dataset
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)  # Total validation steps to walk through the validation data
