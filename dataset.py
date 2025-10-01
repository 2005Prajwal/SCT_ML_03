import os
import cv2
import numpy as np

def load_images_from_folder(folder, label, image_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
        except:
            pass  # Skip broken images
    return images, labels

def load_dataset(train_path="dataset/train"):
    cats, cat_labels = load_images_from_folder(os.path.join(train_path, "cats"), 0)
    dogs, dog_labels = load_images_from_folder(os.path.join(train_path, "dogs"), 1)
    X = np.array(cats + dogs)
    y = np.array(cat_labels + dog_labels)
    X = X.reshape(len(X), -1)  # Flatten images for SVM
    return X, y
