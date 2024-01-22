import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.src.utils.data_utils import get_file
import keras_vggface
from keras_vggface.vggface import VGGFace
import mtcnn
import keras_vggface.utils
import PIL
import os
import os.path
from sklearn.utils.class_weight import compute_class_weight

from functools import wraps
from deepface.commons import functions
import sys
import io

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np

# https://stackoverflow.com/questions/75231091/deepface-dont-print-logs-from-mtcnn-backend
def capture_output(func):
    """Wrapper to capture print output."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout

    return wrapper

def resize_image(image_array, target_size):
        image = Image.fromarray(image_array)
        resized_image = image.resize(target_size)
        return np.array(resized_image)

def preprocess_image(image_original, label, face_detector_model = mtcnn.MTCNN()):
    # Have to convert image to certain type for resize to work for some reason --> uint8
    # https://stackoverflow.com/questions/68429181/cv2-error-opencv4-5-2-c-users-modules-imgproc-src-resize-cpp3929-err
    # https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
    
    if not face_detector_model:
        face_detector_model = mtcnn.MTCNN()
    
    # Ensure the image has 3 channels (for RGB)
    if image_original.shape[-1] != 3:
        print("Invalid number of channels in the image.")
        return []

    image2 = np.array(image_original,dtype=np.uint8)

    w_detect_faces = capture_output(face_detector_model.detect_faces)
    face_roi = w_detect_faces(image2)

    if not face_roi:
         print("No face detected, moving on.")
         return image_original, label
    else:
        x1, y1, width, height = face_roi[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = image2[y1:y2, x1:x2]
        
    # Assuming face_array is your input image data
    face_array_resized = resize_image(face, target_size=(224, 224))

    # Expand dimensions to create a batch of size 1
    face_array_expanded = np.expand_dims(face_array_resized, axis=0)

    # Preprocess the input data for VGG16 (imported)
    face_array_preprocessed = preprocess_input(face_array_expanded)

    return face_array_preprocessed, label

def preprocess_image2(image, label):
    # Your preprocessing logic here
    # Modify the image data and label as needed
    
    # Example: Convert image to grayscale
    image = tf.image.rgb_to_grayscale(image)
    
    # Example: Normalize image pixel values
    image = image / 255.0
    
    # Example: Resize image
    image = tf.image.resize(image, (224, 224))
    
    return image, label

def preprocess_photo_1(photo_path = 'training/ravi/headshot.JPG', face_detector_model = mtcnn.MTCNN()):
    initial_photo = plt.imread(photo_path)
    print(initial_photo.shape)

    if not face_detector_model:
        face_detector_model = mtcnn.MTCNN()
    
    face_roi = face_detector_model.detect_faces(initial_photo)

    x1, y1, width, height = face_roi[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = initial_photo[y1:y2, x1:x2]
    print(face.shape)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5,3))
    axes[0].imshow(initial_photo)
    axes[1].imshow(face)
    plt.show()

    # Assuming face_array is your input image data
    face_array_resized = resize_image(face, target_size=(224, 224))

    # Expand dimensions to create a batch of size 1
    face_array_expanded = np.expand_dims(face_array_resized, axis=0)
    plt.imshow(face_array_expanded[0])

    # Preprocess the input data for VGG16
    face_array_preprocessed = preprocess_input(face_array_expanded)
    plt.imshow(face_array_preprocessed[0])
    plt.show()
    return face_array_preprocessed


# Define a function to load and preprocess each image
def load_and_preprocess_image(file_path = 'training/madonna/httpassetsrollingstonecomassetsimagesalbumreviewaffaceabdcccaeedjpg.jpg', label = 0):
    '''
    
    Takes in filepath to an image and preprocesses it such that it can be ran through VGG16 facial detection.

    Arguments:
        - file_path (str): full path to the target image
        - label (int): label for the image dataset

    Return:
        - face_array_preprocessed
        - label (int): label for the image dataset
    '''
    image = load_img(file_path)
    image = img_to_array(image, dtype=np.uint8)
    # print(image.shape)

    w_detect_faces = capture_output(mtcnn.MTCNN().detect_faces)
    face_roi = w_detect_faces(image)
    if not face_roi:
            print("No face detected, moving on.")
            # Not an actual face, var name used for simplicity
            face = resize_image(image, target_size=(224, 224))

    else:
        x1, y1, width, height = face_roi[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = image[y1:y2, x1:x2]
            
    # Assuming face_array is your input image data
    face_array_resized = resize_image(face, target_size=(224, 224))

    # Expand dimensions to create a batch of size 1
    face_array_expanded = np.expand_dims(face_array_resized, axis=0)
    
    # Preprocess the input data for VGG16 (imported)
    # For some reason, this function is darkening the image and/or making it appear blue
    # The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.
    face_array_preprocessed = preprocess_input(face_array_expanded)

    return face_array_preprocessed, label

def custom_image_dataset(dir_path = 'training', preprocessing_function = load_and_preprocess_image, batch_size=8, shuffle=True):
    file_list = []
    labels = []
    
    # Iterate through the directories in the main directory
    for label in os.listdir(dir_path):
        # print(label) # The names of the subdirectories (people)
        label_path = os.path.join(dir_path, label)
        
        # Iterate through the files in each subdirectory
        for file in os.listdir(label_path): #[0:2]:
            file_path = os.path.join(label_path, file)
            # print(file_path)
            # Append file path and corresponding label
            file_list.append(file_path)
            labels.append(label)
    
    # Apply the load_and_preprocess_image function to each element in the dataset
    preprocessed_data = [
        preprocessing_function(file_path, label)
        for file_path, label in zip(file_list, labels)
    ]

    # Unpack the preprocessed data
    preprocessed_images, preprocessed_labels = zip(*preprocessed_data)

    # Flatten the nested list
    preprocessed_images = [item for sublist in preprocessed_images for item in sublist]
    # Create a mapping from labels to integers
    label_to_index = {label: idx for idx, label in enumerate(set(preprocessed_labels))}
    # Output: {'cat': 0, 'dog': 1, 'bird': 2, 'elephant': 3}

    # Use the mapping to convert labels to integers
    post_processed_labels = [label_to_index[label] for label in preprocessed_labels]
    # Output: [0, 1, 1, 2, 3, 1, 0]

    # Create a dataset from the preprocessed data
    dataset = tf.data.Dataset.from_tensor_slices((preprocessed_images, post_processed_labels))

    # Shuffle the dataset if specified
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_list))

    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    return dataset, label_to_index


if __name__ == "__main__":
    testing_image, testing_label = load_and_preprocess_image()