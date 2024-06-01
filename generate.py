import os
import cv2
import numpy as np
from PIL import Image, ImageOps

def detect_faces(image_path):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            faces.append((startX, startY, endX - startX, endY - startY))

    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")

    return faces

def crop_image_to_face(image_path, output_path, desired_width, desired_height, reduce=True):
    try:
        faces = detect_faces(image_path)
    except ValueError as e:
        print(e)
        return

    image = Image.open(image_path)

    for i, (x, y, w, h) in enumerate(faces):
        center_x, center_y = x + w // 2, y + h // 2

        # Ensure a margin around the face
        margin_w = int(0.6 * w)  # 60% of face width as margin
        margin_h = int(0.6 * h)  # 60% of face height as margin

        left = max(center_x - desired_width // 2 - margin_w, 0)
        upper = max(center_y - desired_height // 2 - margin_h, 0)
        right = min(center_x + desired_width // 2 + margin_w, image.width)
        lower = min(center_y + desired_height // 2 + margin_h, image.height)

        cropped_image = image.crop((left, upper, right, lower))

        if cropped_image.mode != 'RGB':
            cropped_image = cropped_image.convert('RGB')

        if reduce:
            cropped_image = cropped_image.resize((desired_width, desired_height), Image.LANCZOS)

        # Adding white background if the cropped image is smaller than the desired size
        background = Image.new('RGB', (desired_width, desired_height), (255, 255, 255))
        offset = ((desired_width - cropped_image.width) // 2, (desired_height - cropped_image.height) // 2)
        background.paste(cropped_image, offset)

        face_output_path = os.path.splitext(output_path)[0] + f"_face_{i + 1}.jpg"
        background.save(face_output_path)
        print(f"Cropped face {i + 1} saved to: {face_output_path}")

def process_images(input_folder, output_folder, desired_width, desired_height, reduce=True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            crop_image_to_face(input_image_path, output_image_path, desired_width, desired_height, reduce)

# Example usage
input_folder = 'photo'
output_folder = 'output'
desired_width = 200
desired_height = 300

process_images(input_folder, output_folder, desired_width, desired_height, reduce=True)
