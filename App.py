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

class VideoProcessor1:
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
    
#Social Distancing
    
def Check(a,  b):
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    calibration = (a[1] + b[1]) / 2       
    if 0 < dist < 0.25 * calibration:
        return True
    else:
        return False
def Setup(yolo):
    global net, ln, LABELS
    weights = os.path.sep.join([yolo, "yolov3.weights"])
    config = os.path.sep.join([yolo, "yolov3.cfg"])
    labelsPath = os.path.sep.join([yolo, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")  
    net = cv2.dnn.readNetFromDarknet(config, weights)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def ImageProcess(image):
    global processedImg
    (H, W) = (None, None)
    frame = image.copy()
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    starttime = time.time()
    layerOutputs = net.forward(ln)
    stoptime = time.time()
    print("Video is Getting Processed at {:.4f} seconds per frame".format((stoptime-starttime))) 
    confidences = []
    outline = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            maxi_class = np.argmax(scores)
            confidence = scores[maxi_class]
            if LABELS[maxi_class] == "person":
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    outline.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
    box_line = cv2.dnn.NMSBoxes(outline, confidences, 0.5, 0.3)
    if len(box_line) > 0:
        flat_box = box_line.flatten()
        pairs = []
        center = []
        status = [] 
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(False)
        for i in range(len(center)):
            for j in range(len(center)):
                close = Check(center[i], center[j])
                if close:
                    pairs.append([center[i], center[j]])
                    status[i] = True
                    status[j] = True
        index = 0
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            if status[index] == True:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
            elif status[index] == False:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            index += 1
        for h in pairs:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
    processedImg = frame.copy()

yolo = r"C:\Users\botsc\Desktop\Detectie_masca\yolov3"
    
class VideoProcessor2:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        current_img = frm.copy()
        video = current_img.shape
        Setup(yolo)
        ImageProcess(current_img)
        Frame = processedImg
        return av.VideoFrame.from_ndarray(Frame, format = 'bgr24')
        

def main():
    activities = ["Mask Detection", "Social Distancing Verifying"]
    side_title = st.sidebar.header("STOP COVID SPREADING")
    choice = st.sidebar.selectbox("Select Activity", activities)
    if choice == "Mask Detection":
        st.header("Mask Detection")
        image = Image.open(r"C:\Users\botsc\Desktop\Detectie_masca\Mask.png")
        st.image(image)
        st.header("Webcam Live Feed")
        st.write("Click on START to use webcam, detect faces and verify if people wear mask")
        webrtc_streamer(key="key", video_processor_factory=VideoProcessor1,
                       rtc_configuration=RTCConfiguration(
                       {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                       )
                       )
    elif choice == "Social Distancing Verifying":
        st.header("Social Distancing Monitoring")
        image = Image.open(r"C:\Users\botsc\Desktop\Detectie_masca\Distancing.jpg")
        st.image(image)
        st.header("Webcam Live Feed")
        st.write("Click on START to use webcam, detect persons and verify if they respect social distancing")
        webrtc_streamer(key="key", video_processor_factory=VideoProcessor2,
                       rtc_configuration=RTCConfiguration(
                       {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                       )
                       )

if __name__ == "__main__":
    main()
