import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array


def predict_fruit(image_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    image = load_img(
        image_path, target_size=(256, 256)
    )  # Resize to the target size used during training
    img_array = img_to_array(image)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype(
        np.float32
    )  # Ensure the data type matches the model's input type

    # Set the input tensor
    interpreter.set_tensor(input_details[0]["index"], img_array)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # Post-process the output to get the predicted class
    predicted_class = np.argmax(output_data, axis=1)

    # Map the index to class names
    class_names = [
        "Apple_Bad",
        "Apple_Good",
        "Banana_Bad",
        "Banana_Good",
        "Guava_Bad",
        "Guava_Good",
        "Lime_Bad",
        "Lime_Good",
        "Orange_Bad",
        "Orange_Good",
        "Pomegranate_Bad",
        "Pomegranate_Good",
    ]

    return class_names[predicted_class[0]]


# list of image path
# ./dataset/Apple_Bad/IMG_20190910_172657.jpg
# ./dataset/Apple_Good/IMG_9625.jpg
# ./dataset/Banana_Bad/IMG_7695.jpg
# ./dataset/Banana_Good/IMG_0120.jpg
# ./dataset/Pomegranate_Good/20190820_143619.jpg
# ./dataset/Pomegranate_Bad/IMG_20190829_072303.jpg
# ./dataset/Orange_Good/IMG_1896.jpg
# ./dataset/Orange_Bad/IMG_2369.jpg
# ./dataset/Lime_Good/IMG_20190902_102418.jpg
# ./dataset/Lime_Bad/IMG_6950.jpg
# ./dataset/Guava_Good/20190813_130029.jpg
# ./dataset/Guava_Bad/IMG_20190822_080645.jpg

# loop for predict
while True:
    image_path = input("Enter the image path (or type 'exit()' to quit): ")
    if image_path.lower() == 'exit()':
        break
    try:
        predicted_fruit = predict_fruit(image_path)
        print(f"Predicted class: {predicted_fruit}")
    except Exception as e:
        print(f"Error: {e}")
