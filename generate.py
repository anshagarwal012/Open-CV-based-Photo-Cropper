import os
import cv2
import numpy as np
from PIL import Image

def detect_face(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Load the pre-trained deep learning model
    prototxt_path = "deploy.prototxt"  # Path to deploy.prototxt
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"  # Path to res10_300x300_ssd_iter_140000.caffemodel
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Prepare the image for deep learning-based face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Initialize the list of faces detected
    faces = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the `confidence` is greater than a minimum threshold
        if confidence > 0.5:  # You can adjust the confidence threshold as needed
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Append the face coordinates to the list
            faces.append((startX, startY, endX - startX, endY - startY))

    # Ensure at least one face is found
    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")
    
    print(f"Faces detected: {len(faces)}")  # Debugging line

    # Return the coordinates of the first detected face
    return faces[0]

def crop_image_to_face(image_path, output_path, desired_width, desired_height):
    # Detect the face in the image
    try:
        x, y, w, h = detect_face(image_path)
    except ValueError as e:
        print(e)
        return

    # Load the image using Pillow
    image = Image.open(image_path)

    # Calculate the center of the face
    center_x, center_y = x + w // 2, y + h // 2

    # Calculate the desired crop box
    left = max(center_x - desired_width // 2, 0)
    upper = max(center_y - desired_height // 2, 0)
    right = min(center_x + desired_width // 2, image.width)
    lower = min(center_y + desired_height // 2, image.height)

    # Crop the image to the desired box
    cropped_image = image.crop((left, upper, right, lower))

    # Convert the image to 'RGB' mode if necessary
    if cropped_image.mode != 'RGB':
        cropped_image = cropped_image.convert('RGB')

    # Save the cropped image
    cropped_image.save(output_path)
    print(f"Cropped image saved to: {output_path}")  # Debugging line

def process_images(input_folder, output_folder, desired_width, desired_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            crop_image_to_face(input_image_path, output_image_path, desired_width, desired_height)

# Example usage
input_folder = 'photo'
output_folder = 'output'
desired_width = 200
desired_height = 300

process_images(input_folder, output_folder, desired_width, desired_height)
