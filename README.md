# Skin Lesion Classification

In this project work, I have developed an AI model consisting of patient metadata and image data collection followed by applying deep learning model Efficientnet for skin lesion classification into suspicious and non-suspicious categories.

![ai_engine](https://github.com/user-attachments/assets/91d4a9cf-103f-40a5-89eb-aa943c04af23)

We have a total of six AI model by varing input data type as shown here-
![multi-modal](https://github.com/user-attachments/assets/95874de1-7d4d-40c0-8be7-bb8089f4f55f)





## How to Run
Download and install anaconda. Then clone or download  https://github.com/MdShafiqu/Skin-Lesion-Detection, Create a folder name my_repo and keep all files from repo inside it.  Implement following commands in anaconda prompt-

C:\> conda create -n tf_app pip python=3.7

C:\> activate tf_app

(tf_app) C:\>python -m pip install --upgrade pip

(tf_app) C:\>pip install tensorflow-object-detection-api

(tf_app) C:\> pip install flask

(tf_app) C:\> pip install tensorflow==1.13.1

(tf_app) C:\> conda install -c anaconda protobuf

(tf_app) C:\> pip install pillow

(tf_app) C:\> pip install lxml

(tf_app) C:\> pip install Cython

(tf_app) C:\> pip install contextlib2

(tf_app) C:\> pip install jupyter

(tf_app) C:\> pip install matplotlib

(tf_app) C:\> pip install scikit-image

(tf_app) C:\> pip install pandas

(tf_app) C:\> pip install opencv-python

Now test the web application by running following commands-

(tf_app) C:\> cd C:\my_repo

(tf_app) C:\my_repo> python main_app.py

Open a browser and enter address- http://127.0.0.1:5000/ or suggested address from the prompt to test the web app.
