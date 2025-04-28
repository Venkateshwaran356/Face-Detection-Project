



#---------------------------------------------------------------Step 1: Load Images and YOLO Annotations---------------------------------------------------------------------

import os
import cv2
import numpy as np

# Paths
image_folder = "C:/Data Science and AI/Project/Face Detection Project/images"
label_folder = "C:/Data Science and AI/Project/Face Detection Project/labels"

# To store loaded images and labels
images = []
bboxes = []

# Loop through all images
for image_name in os.listdir(image_folder):
    if image_name.endswith(('.jpg', '.png')):
        # Read the image
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))  # Resize to 224x224
        images.append(img)

        # Read the corresponding label
        label_name = image_name.rsplit('.', 1)[0] + ".txt"
        label_path = os.path.join(label_folder, label_name)

        with open(label_path, 'r') as f:
            line = f.readline().strip().split()
            # YOLO format: class_id x_center y_center width height
            x_center, y_center, width, height = map(float, line[1:])
            bboxes.append([x_center, y_center, width, height])

# Convert lists to numpy arrays
images = np.array(images)
bboxes = np.array(bboxes)

print("Images shape:", images.shape)
print("Bounding boxes shape:", bboxes.shape)

#----------------------------------------------------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------Step 2: Build a Tiny CNN Model---------------------------------------------------------

from tensorflow.keras import layers, models

# Create a simple CNN model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),           # Input layer

    layers.Conv2D(16, (3, 3), activation='relu'),  # Convolution 1
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation='relu'),  # Convolution 2
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),  # Convolution 3
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),                             # Flatten the feature maps
    layers.Dense(128, activation='relu'),          # Fully connected
    layers.Dense(4)                                # Output: 4 numbers (bounding box)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Print model summary
model.summary()

# Normalize images (0 to 1)
images = images / 255.0

# Train the model
history = model.fit(
    images,            # Input: images
    bboxes,            # Target: bounding boxes
    epochs=100,        # Number of times model sees the entire data
    batch_size=2,      # How many samples at once
    verbose=1          # Show training progress
)

#----------------------------------------------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------Predict and Draw Bounding Box---------------------------------------------------------

# Pick one image to test
test_image = images[3]  # First image
test_image_input = np.expand_dims(test_image, axis=0)  # Add batch dimension

# Predict bounding box
predicted_bbox = model.predict(test_image_input)[0]  # Predict and remove batch dimension

# Convert YOLO format back to pixel format
h, w, _ = test_image.shape
x_center, y_center, width, height = predicted_bbox

x_center *= w
y_center *= h
width *= w
height *= h

# Calculate box coordinates
x1 = int(x_center - width/2)
y1 = int(y_center - height/2)
x2 = int(x_center + width/2)
y2 = int(y_center + height/2)

# Draw rectangle
test_image_draw = (test_image * 255).astype(np.uint8)  # Convert back to 0-255
cv2.rectangle(test_image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image
import matplotlib.pyplot as plt

plt.imshow(cv2.cvtColor(test_image_draw, cv2.COLOR_BGR2RGB))
plt.title("Predicted Face Detection")
plt.axis('off')
plt.show()

# Save the entire model
model.save('face_detection_model.h5')

# Load the model again
# loaded_model = tf.keras.models.load_model('face_detection_model.h5')


#----------------------------------------------------------------------------------------------------------------------------------------------------
