#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf

app = Flask(__name__)

detector = HandDetector(maxHands=1)


model = tf.keras.models.load_model('MODEL/CNN-MODEL.h5')

offset = 20
imgSize = 300
model_input_size = (224, 224)
labels = ['1', '2', '3', '4', '5', 'A', 'B', 'C', 'I', 'L', 'Q', 'Y']

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                imgWhiteResized = cv2.resize(imgWhite, model_input_size)

                imgWhiteGray = cv2.cvtColor(imgWhiteResized, cv2.COLOR_BGR2GRAY)
                imgWhiteGray = imgWhiteGray / 255.0
                imgWhiteGray = np.expand_dims(imgWhiteGray, axis=-1)
                imgWhiteGray = np.expand_dims(imgWhiteGray, axis=0)

                prediction = model.predict(imgWhiteGray)
                index = np.argmax(prediction)
                
                cv2.rectangle(imgOutput, (x-offset, y-offset-65), (x-offset+100, y-offset-60+50), (0, 128, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y-40), cv2.FONT_HERSHEY_DUPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (0, 128, 0), 4)

            ret, buffer = cv2.imencode('.jpg', imgOutput)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




