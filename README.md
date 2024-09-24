# Automatic License Plate Detection

## Overview

This project involves detecting vehicle license plates using different techniques, including:

- **Haar Cascade Classifier** for classical object detection.
- **Single Shot Multibox Detector (SSD)** for deep learning-based object detection.
- **Feedforward Neural Network (FFNN)** for bounding box regression.

The project was implemented in Python, leveraging libraries like OpenCV, TensorFlow, and Matplotlib, and developed in Jupyter Notebook.

---


## Tools and Libraries

- **Python 3.8+**
- **OpenCV**: For image processing and classical detection (Haar Cascade).
- **TensorFlow**: For building SSD and FFNN models.
- **Matplotlib**: For visualizing detection results.
- **Pandas**: For handling data (CSV format).
- **XML.etree.ElementTree**: For parsing XML files containing annotations.

---

## Objectives

1. **License Plate Detection using Haar Cascade**: 
   - Implemented a classical object detection algorithm using OpenCV's Haar Cascade Classifier to detect license plates in images.
   - The Haar Cascade used was `haarcascade_russian_plate_number.xml`.
   
2. **Bounding Box Regression using FFNN**:
   - Built a Feedforward Neural Network (FFNN) to predict the bounding box coordinates of the license plate in an image.
   
3. **Object Detection using SSD**:
   - Built a Single Shot Multibox Detector (SSD) model using a pre-trained MobileNetV2 as the base and fine-tuned it for license plate detection.

---

## Methodology

### 1. **Haar Cascade Classifier**
   
   - The Haar Cascade is a machine learning-based approach for object detection. It uses pre-trained classifiers to detect objects like faces or license plates.
   - **Code Snippet:**
     ```python
     # Load the Haar Cascade for license plate detection
     plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

     # Detect license plates in the image
     plates = plate_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
     ```

### 2. **Feedforward Neural Network (FFNN)**
   
   - The FFNN is used for bounding box regression, where the model learns to predict the bounding box (center_x, center_y, width, height) given an input image.
   - **Code Snippet:**
     ```python
     model = models.Sequential([
         layers.Flatten(input_shape=(300, 300, 3)),
         layers.Dense(128, activation='relu'),
         layers.Dense(64, activation='relu'),
         layers.Dense(4)  # Output: bounding box coordinates
     ])
     ```

### 3. **Single Shot Multibox Detector (SSD)**
   
   - An SSD model was built using a pre-trained MobileNetV2 as the base model. SSD models are known for fast and efficient object detection.
   - **Code Snippet:**
     ```python
     base_model = tf.keras.applications.MobileNetV2(input_shape=(300, 300, 3), include_top=False)
     base_model.trainable = False
     
     inputs = layers.Input(shape=(300, 300, 3))
     x = base_model(inputs, training=False)
     x = layers.Conv2D(4, (1, 1), padding='same')(x)
     x = layers.Flatten()(x)
     outputs = layers.Dense(4)(x)  # Output: bounding box coordinates
     model = tf.keras.Model(inputs, outputs)
     ```

---

## Dataset

- **Input**: A folder containing vehicle images and their corresponding XML files, which provide bounding box annotations for the license plates.
- **Processing**:
   1. XML files are parsed using `xml.etree.ElementTree`.
   2. Bounding box coordinates are extracted and converted into a CSV file for further processing.
   3. Images are resized and normalized for input to the neural networks.

---

