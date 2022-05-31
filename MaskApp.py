from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
from PIL import Image
import sqlite3
import streamlit as st
from streamlit_option_menu import option_menu
import pyttsx3


# Create sound for the case when no mask is detected
# tts = gtts.gTTS("Please put mask on!", lang="en")
# tts.save("PutMaskOn-EN.mp3")


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

# faceNet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

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
            engine = pyttsx3.init()
            engine.setProperty('rate', 100)
            if mask > withoutMask:
                label = "Mask Detected! " + str(int(pred[0]*100)) + "%"
                color = (0, 255, 0)
            else:
                engine.say("Please put the mask on!")
                label = "No Mask Detected! " + str(int(pred[1]*100)) + "%"
                color = (0, 0, 255)
                engine.runAndWait()
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
    # st.sidebar.title("Prediction Mask App")
    # menu = ["Home", "Login", "SignUp"]

    # choice = st.sidebar.selectbox("Menu", menu)

    with st.sidebar:
        selected = option_menu("Main Menu", ["Mask Detection", "Login", "SignUp"], 
            icons=['house'], menu_icon="cast", default_index=0)
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

             
    if selected == "Login":
        st.header("Login")
        st.text("")
        image = Image.open("Mask.png")
        st.image(image)

        st.subheader("Login Section")

        username = st.text_input("User Name")
        password = st.text_input("Password", type='password')


        if st.button("Login"):
            create_usertable()
            if username == "":
                st.error("Please insert username!")
            elif password == "":
                st.error("Please insert password!")
            elif login_user(username, password):
                st.success("Logged In as {} !".format(username)) 
                st.session_state.logged_in = True
            else:
                st.error("Incorrect Username/Password!")
    
    if selected == "Mask Detection":
        st.header("Mask Detection")
        st.subheader("This is an app which predicts in real time if a person is wearing or not a mask!")
        st.text("")
        st.write("Please go to Login section and log in first, to be able to use this app!")
        st.text("")
        image = Image.open("Mask.png")
        st.image(image)
        if  st.session_state.logged_in == True:
            st.header("Webcam Live Feed")
            st.write("Click on START to use webcam, detect faces and verify if people wear mask")
            webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                        rtc_configuration=RTCConfiguration(
                        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                        )) 
    if selected == "SignUp":
        admin_username = st.sidebar.text_input("Admin username")
        admin_password = st.sidebar.text_input("Admin password", type= 'password')
        adminBtn = st.sidebar.button("Login Admin")
        st.header("Register")
        st.subheader("Create account for new user!")
        st.text("")
        image = Image.open("Mask.png")
        st.image(image)
        if admin_username == "admin" and admin_password == "admin" and adminBtn:
            st.sidebar.success("Logged in succesfully!")
            st.sidebar.info("Create new user account!")
            st.subheader("Create New Account")
            new_user = st.text_input("Username")
            new_password = st.text_input("Password", type='password')
            if st.button("Create account"):
                if new_user == "":
                    st.error("Please insert username!")
                elif new_password == "":
                    st.error("Please insert password!")
                else:
                    create_usertable()
                    try:
                        add_userdata(new_user, new_password)
                        st.success("You have succesfully created a valid account!")
                        st.info("Go to Login Menu to login!")
                    except:
                        st.error("Username already exists! Insert a new one!")
        elif admin_username != "admin" and adminBtn:
            st.sidebar.error("Wrong or empty username!")
            st.sidebar.warning("Enter username again!")
        elif admin_password != "admin" and adminBtn:
            st.sidebar.error("Wrong or empty password!")
            st.sidebar.warning("Enter password again!")


if __name__ == "__main__":
    main()
