import os
import time

import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Define labels
LABELS = [
    'barretts', 'bbps-0-1', 'bbps-2-3', 'dyed-lifted-polyps', 'dyed-resection-margins',
    'hemorrhoids', 'ileum', 'impacted-stool', 'normal-cecum', 'normal-pylorus', 'normal-z-line',
    'oesophagitis-a', 'oesophagitis-b-d', 'polyp', 'retroflex-rectum', 'retroflex-stomach', 'short-segment-barretts',
    'ulcerative-colitis-0-1', 'ulcerative-colitis-1-2', 'ulcerative-colitis-2-3', 'ulcerative-colitis-grade-1',
    'ulcerative-colitis-grade-2', 'ulcerative-colitis-grade-3'
]

# Set path to test dataset
TEST_DATASET_PATH = '/endotect'

# Load Keras model
model = load_model('/submission/model.h5')

# Loop over each image of the test dataset
for image_name in os.listdir(TEST_DATASET_PATH):

    # Load the test image
    image_path = os.path.join(TEST_DATASET_PATH, image_name)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Start timer
    start_time = time.time()

    # Perform prediction
    prediction = model.predict(image)[0]
    prediction_index = prediction.argmax(axis=-1)
    
    # End timer
    end_time = time.time() - start_time

    # Print results to stdout
    print('{}, {}, {}, {:.10f}'.format(image_name, LABELS[prediction_index], prediction[prediction_index], end_time))