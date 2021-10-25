import numpy as np
from streamlit.uploaded_file_manager import UploadedFile
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import streamlit as st


def app():
    page_names=["Model Prediction","Dataset"]
    page=st.sidebar.radio("Navigation",page_names)
    if page=="Model Prediction":
        classes = { 0:'Speed limit (20km/h)',
                    1:'Speed limit (30km/h)', 
                    2:'Speed limit (50km/h)', 
                    3:'Speed limit (60km/h)', 
                    4:'Speed limit (70km/h)', 
                    5:'Speed limit (80km/h)', 
                    6:'End of speed limit (80km/h)', 
                    7:'Speed limit (100km/h)', 
                    8:'Speed limit (120km/h)', 
                    9:'No passing', 
                    10:'No passing veh over 3.5 tons', 
                    11:'Right-of-way at intersection', 
                    12:'Priority road', 
                    13:'Yield', 
                    14:'Stop', 
                    15:'No vehicles', 
                    16:'Veh > 3.5 tons prohibited', 
                    17:'No entry', 
                    18:'General caution', 
                    19:'Dangerous curve left', 
                    20:'Dangerous curve right', 
                    21:'Double curve', 
                    22:'Bumpy road', 
                    23:'Slippery road', 
                    24:'Road narrows on the right', 
                    25:'Road work', 
                    26:'Traffic signals', 
                    27:'Pedestrians', 
                    28:'Children crossing', 
                    29:'Bicycles crossing', 
                    30:'Beware of ice/snow',
                    31:'Wild animals crossing', 
                    32:'End speed + passing limits', 
                    33:'Turn right ahead', 
                    34:'Turn left ahead', 
                    35:'Ahead only', 
                    36:'Go straight or right', 
                    37:'Go straight or left', 
                    38:'Keep right', 
                    39:'Keep left', 
                    40:'Roundabout mandatory', 
                    41:'End of no passing', 
                    42:'End no passing veh > 3.5 tons' }

        model_path = "model.h5"
        loaded_model = tf.keras.models.load_model(model_path)

        st.title("Traffic Signs Recognition")
        

        UploadedFile = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
        
        if UploadedFile is not None:
            image = Image.open(UploadedFile)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            image=np.asarray(image)
            image_fromarray = Image.fromarray(image, 'RGB')
            print(image_fromarray)
            resize_image = image_fromarray.resize((30, 30))
            expand_input = np.expand_dims(resize_image,axis=0)
            input_data = np.array(expand_input)
            input_data = input_data/255
            pred = loaded_model.predict(input_data)
            result = pred.argmax()
            print(classes[result])
            st.success(classes[result])

    elif page=="Dataset":
        #st.markdown("<h4 style='text-align: center; color: #ff4b4b;'>This model is created with the help of GTSRB Dataset from kaggle</h4>", unsafe_allow_html=True)
        st.write("This model is created with the help of [GTSRB Dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) from kaggle")
        st.write("It is a multi-class classification problem")
        st.write("It has total of 43 classes")
        st.write("It has around 40,000 images")
        st.image("0.png")
        st.image("1.png")
        st.image("2.png")
        
if __name__ == "__main__":
    	app()



