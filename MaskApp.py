from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
from PIL import Image
import sqlite3

#Mask Detection

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

prototxtPath = "face_detector/deploy.prototxt"
weightsPath =  "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model("MaskDetector.h5")

class VideoProcessor:
    # detect faces in the frame and determine if they are wearing a face mask or not
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        (locs, preds) = detect_and_predict_mask(frm, faceNet, maskNet)
        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            # determine the class label and color we'll use to draw
            # the bounding box and text
            
            if mask > withoutMask:
                label = "Mask Detected!"
                color = (0, 255, 0)
            else:
                label = "No Mask Detected!"
                color = (0, 0, 255)
            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frm, label, (startX-20, startY - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frm, (startX, startY), (endX, endY), color, 2)
        return av.VideoFrame.from_ndarray(frm, format = 'bgr24')
    
#DB management
conn = sqlite3.connect('data.db')
c = conn.cursor()


def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT UNIQUE, password TEXT)')


def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', (username, password))
    conn.commit()


def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username=? AND password = ?', (username, password))
    data = c.fetchall()
    return data    

def main():
    st.sidebar.title("Prediction Mask App")
    menu = ["Home", "Login", "SignUp"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.header("Mask Detection")
        st.subheader("This is an app which predicts in real time if a person is wearing or not a mask!")
        image = Image.open("Mask.png")
        st.image(image)

    if choice == "SignUp":
        st.sidebar.subheader("Create New Account")
        new_user = st.sidebar.text_input("Username")
        new_password = st.sidebar.text_input("Password", type='password')

        if st.sidebar.button("Signup"):
            if new_user == "":
                st.sidebar.error("Please insert username!")
            elif new_password == "":
                st.sidebar.error("Please insert password!")
            else:
                create_usertable()
                try:
                    add_userdata(new_user, new_password)
                    st.sidebar.success("You have succesfully created a valid account!")
                    st.sidebar.info("Go to Login Menu to login!")
                except:
                    st.sidebar.error("Username already exists! Insert a new one!")
                    
    if choice == "Login":
        st.sidebar.subheader("Login Section")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')

        if st.sidebar.checkbox("Login"):
            
            create_usertable()
            if username == "":
                st.sidebar.error("Please insert username!")
            elif password == "":
                st.sidebar.error("Please insert password!")
            elif login_user(username, password):
                st.sidebar.success("Logged In as {} !".format(username))
                st.header("Mask Detection")
                image = Image.open("Mask.png")
                st.image(image)
                st.header("Webcam Live Feed")
                st.write("Click on START to use webcam, detect faces and verify if people wear mask")
                webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                               rtc_configuration=RTCConfiguration(
                               {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                               ))   
            else:
                st.sidebar.error("Incorrect Username/Password!")


if __name__ == "__main__":
    main()
