from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2 as cv
import numpy as np

face_classifier = cv.CascadeClassifier(r"D:\Minor2\haarcascade_frontalface_default.xml")
classifier =load_model(r"D:\Minor2\model.h5")

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv.resize(roi_gray,(48,48),interpolation=cv.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv.putText(frame,label,label_position,cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv.putText(frame,'No Faces',(30,80),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv.imshow('Emotion Detector',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()