import streamlit as st
from PIL import Image
import cv2 as cv
from keras.models import load_model
import numpy as np

model = load_model('artifacts\\training\model.h5')


def sleep_detection(image):
    image=Image.open(image)
    img=image.resize((224,224))
    y_pred = model.predict(img.reshape(1, 224, 224, 3))
    prediction=np.argmax(y_pred)
    if prediction==0:
        print('The Person is Awake')
    else:
        print('The Person is Sleeping')

file=st.file_uploader('Upload the Image',type=['jpg','png'])
if file is None:
    st.text("Please upload the image file")
else:
    st.image(Image.open(file),use_column_width=True)
    sleep_detection(file)


