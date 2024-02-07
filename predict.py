from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def generate_random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())

def predict_coordinates(image_path, model_path):
    # Load the pre-trained model
    model = load_model(model_path)

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Unable to load the image from {image_path}")
        sys.exit(1)

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = cv2.circle(image, (80, 75), 5, generate_random_color(), -1)

    # Make a prediction
    predicted_labels = model.predict(np.expand_dims(image, axis=0))[0]

    # Print the predicted coordinates
    print("Predicted Coordinates:", predicted_labels)

    # Plot the image with the predicted coordinates
    plt.imshow(image)
    plt.scatter(predicted_labels[0], predicted_labels[1], c='green', label='Predicted')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_coordinates.py <image_path> <model_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = 'model.h5'
    predict_coordinates(image_path, model_path)
