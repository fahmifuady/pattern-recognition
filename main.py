import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from models.data import DataPreprocessor, DataPaths
from models.cnn import CNNModel, myCallback

# Define paths and preprocess the data
print('Define paths and preprocess the data')
dataset_path = './dataset'
output_path = './dataset_split'
data_preprocessor = DataPreprocessor(dataset_path, output_path)
data_preprocessor.preprocess_data()
tr = os.path.join(output_path, 'train')
va = os.path.join(output_path, 'val')

# Define dataset paths
print('Define dataset paths')
main_dir = './dataset_split'
data_paths = DataPaths(main_dir)
train_paths = data_paths.get_train_paths()
val_paths = data_paths.get_val_paths()

# Image data generators
# Define your ImageDataGenerators here

# Define CNN model
input_shape = (256, 256, 3)  # Example input shape, adjust as needed
num_classes = 12  # Example number of classes, adjust as needed
cnn_model = CNNModel(input_shape, num_classes)
model = cnn_model.build_model()

# Train the model
# Use your ImageDataGenerators and train the model here
tr_idg = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    shear_range=0.2
    )
va_igd = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    fill_mode='nearest',
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

tr_generator = tr_idg.flow_from_directory(
    tr,
    # train_paths,
    target_size=(256, 256),
    batch_size=4,
    class_mode='categorical',
    shuffle=True
)

va_generator = va_igd.flow_from_directory(
    va,
    # val_paths,
    target_size=(256, 256),
    batch_size=4,
    class_mode='categorical',
    shuffle=True
)
# Plot training history
# Plot your training history here

callbacks = myCallback()

history = model.fit(
    tr_generator,
    steps_per_epoch=35,
    epochs=100,
    validation_data=va_generator,
    validation_steps=5,
    verbose=2,
    callbacks=[callbacks])

# liat grafik akurasi
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validasion Accuracy')
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

# Save and convert model to TFLite
# Save and convert your model to TFLite here
#liat versi tf
print(tf.__version__)

model.save('savemodel', include_optimizer=True)

# Convert the model
saved_model_dir = 'savemodel'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
