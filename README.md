link:https://sctml0-7opstzjbttaaicxfxgb2iv.streamlit.app/

Cats vs Dogs Classifier using SVM
A web application built with Streamlit that predicts whether an uploaded image is a cat or a dog using a pre-trained Linear SVM model.

How it Works
Upload an image (JPG/PNG).
The image is resized to 64x64 and converted into a numerical array.
Features are fed into a pre-trained Linear SVM model saved as cat_dog_svm.pkl.
The app displays a prediction: Cat or Dog.

Dataset

Dependencies
pip install streamlit numpy scikit-learn joblib pillow
