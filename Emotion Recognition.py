# https://youtu.be/JmvmUWIP2v8
"""
Live prediction of emotion using pre-trained models. 
Uses haar Cascades classifier to detect face..
then, uses pre-trained models for emotion to predict them from 
live video feed. 

"""
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import deepface as DeepFace
import sys
import time
from flask import Flask, render_template, url_for, redirect,request
import random

import matplotlib.pyplot as plt
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

sys.path.append('../')

from datetime import datetime

def create_log(time_log,emotion_log):
    with open(r'emotion_log.txt', 'w') as fp:
        for x,y in zip(time_log, emotion_log):
            # write each item on a new line
            #print(text)
            fp.write("%s\n" % (x + ' ' + '-' + ' ' + y))

def randrange_float(start, stop, step):
    if step == 0:
        return start
    dec_count = str(step)
    dec_place = dec_count[::-1].find('.')
    if dec_place == -1:
        return int(random.randint(0, int((stop - start) / step)) * step + start)
    else:
        return round(random.randint(0, int((stop - start) / step)) * step + start, dec_place)

img_size = 224
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = load_model('adjusted_emotion_detection_model_200epochs.h5')

result_data = ['Neutral']
result_data_graph = ['Neutral']
time_for_log = ['00:00:00']

class_labels=['Negative', 'Neutral', 'Positive']

cap=cv2.VideoCapture(0)

mantime = 0

#VARIABLES PLACEHOLDER VALUES
radius = randrange_float(0, 400, 10) # "radius of shape <i>in pixels</i>", min: 0, max: 3000, step: 10, default: 500},
linered = 0 # "red component of line color", min: 0, max: 255, step: 1, default: 0, class: "red"},
linegreen = 0 #"green component of line color", min: 0, 255, 1, default: 0, class: "green"},
lineblue = 0
shape = 4 #randrange_float(2, 4, 1) # "shape (1: square, 2: circle, 3: triangle, 4: line)", min: 1, max: 4, step: 1, default: 1},
canvasred = 255
canvasgreen = 255
canvasblue = 255

#MAKE THE WEBSITE RUN, GENERATE PICTURE, DOWNLOAD IT, ALL IN THE BACKGROUND.
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', shape = shape, radius = radius, linered = linered, linegreen = linegreen, lineblue = lineblue, canvasblue = canvasblue, canvasred = canvasred, canvasgreen = canvasgreen)
    #return "Hello! this is the main page <h1>HELLO</h1>"

def enable_download_headless(browser,download_dir):
    browser.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
    params = {'cmd':'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': download_dir}}
    browser.execute("send_command", params)

def run_browser():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.chrome import ChromeDriverManager
    
    options = Options()
    options.add_argument("--disable-notifications")
    options.add_argument('--no-sandbox')
    options.add_argument('--verbose')
    options.add_experimental_option("prefs", {
        "download.default_directory": "C:\\tmp",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing_for_trusted_sources_enabled": False,
        "safebrowsing.enabled": False
    })
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-software-rasterizer')
    options.add_argument('--headless')
    
    driver = webdriver.Chrome(options=options,service=ChromeService(ChromeDriverManager().install()))

    enable_download_headless(driver, "C:\\Users\\lutiq\\Documents\\Generative Art Code\\images")
    
    driver.get('http://localhost:5000')

    time.sleep(20)

    driver.close()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

def run_server():
    app.run(port =5000, threaded = True)


while True:
    ret,frame=cap.read()
    labels=[]
    
    if mantime < 400:
        mantime = mantime + 1
    elif mantime == 400:
        mantime = 0
        
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    if mantime == 0:
        faces=face_classifier.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            
            #Get image ready for prediction
            roi=roi_gray.astype('float')/255.0  #Scale
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)  #Expand dims to get it ready for prediction (1, 48, 48, 1)
 
            
            preds=emotion_model.predict(roi)[0]  #Yields one hot encoded result for 7 classes
            label=class_labels[preds.argmax()]  #Find the label

            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            result_data_graph.append(label)

            if result_data[-1] == 'Positive':
                linered = 255 #randrange_float(0, 255, 1) # "red component of line color", min: 0, max: 255, step: 1, default: 0, class: "red"},
                linegreen = 138#randrange_float(0, 255, 1) #"green component of line color", min: 0, 255, 1, default: 0, class: "green"},
                lineblue = 0#randrange_float(0, 255, 1) # "blue component of line color", min: 0, 255, 1, default: 0, class: "blue"},
                shape = 2 #randrange_float(2, 4, 1) # "shape (1: square, 2: circle, 3: triangle, 4: line)", min: 1, max: 4, step: 1, default: 1},
                canvasred = 242
                canvasgreen = 242
                canvasblue = 174

            elif result_data[-1] == 'Negative':
                linered = 255 #randrange_float(0, 255, 1) # "red component of line color", min: 0, max: 255, step: 1, default: 0, class: "red"},
                linegreen = 255#randrange_float(0, 255, 1) #"green component of line color", min: 0, 255, 1, default: 0, class: "green"},
                lineblue = 255
                shape = 3 #randrange_float(2, 4, 1) # "shape (1: square, 2: circle, 3: triangle, 4: line)", min: 1, max: 4, step: 1, default: 1},
                canvasred = 0
                canvasgreen = 0
                canvasblue = 0

            elif result_data[-1] == 'Neutral':
                linered = randrange_float(50, 78, 1) # "red component of line color", min: 0, max: 255, step: 1, default: 0, class: "red"},
                linegreen = randrange_float(121, 168, 1) #"green component of line color", min: 0, 255, 1, default: 0, class: "green"},
                lineblue = randrange_float(52, 222, 1)
                shape = 4 #randrange_float(2, 4, 1) # "shape (1: square, 2: circle, 3: triangle, 4: line)", min: 1, max: 4, step: 1, default: 1},
                canvasred = 255
                canvasgreen = 255
                canvasblue = 255

            #oldstatus = result_data[-1]

            if label != result_data[-1]:
                result_data.append(label)
                #now = datetime.now()
                #current_time = now.strftime("%H:%M:%S")
                #time_for_log.append(current_time)
                #create_log(time_for_log,result_data)
            
            if result_data[-1] == 'Positive':
                linered = 255 #randrange_float(0, 255, 1) # "red component of line color", min: 0, max: 255, step: 1, default: 0, class: "red"},
                linegreen = 138#randrange_float(0, 255, 1) #"green component of line color", min: 0, 255, 1, default: 0, class: "green"},
                lineblue = 0#randrange_float(0, 255, 1) # "blue component of line color", min: 0, 255, 1, default: 0, class: "blue"},
                shape = 2 #randrange_float(2, 4, 1) # "shape (1: square, 2: circle, 3: triangle, 4: line)", min: 1, max: 4, step: 1, default: 1},
                canvasred = 242
                canvasgreen = 242
                canvasblue = 174

            elif result_data[-1] == 'Negative':
                linered = 255 #randrange_float(0, 255, 1) # "red component of line color", min: 0, max: 255, step: 1, default: 0, class: "red"},
                linegreen = 255#randrange_float(0, 255, 1) #"green component of line color", min: 0, 255, 1, default: 0, class: "green"},
                lineblue = 255
                shape = 3 #randrange_float(2, 4, 1) # "shape (1: square, 2: circle, 3: triangle, 4: line)", min: 1, max: 4, step: 1, default: 1},
                canvasred = 0
                canvasgreen = 0
                canvasblue = 0

            elif result_data[-1] == 'Neutral':
                linered = randrange_float(50, 78, 1) # "red component of line color", min: 0, max: 255, step: 1, default: 0, class: "red"},
                linegreen = randrange_float(121, 168, 1) #"green component of line color", min: 0, 255, 1, default: 0, class: "green"},
                lineblue = randrange_float(52, 222, 1)
                shape = 4 #randrange_float(2, 4, 1) # "shape (1: square, 2: circle, 3: triangle, 4: line)", min: 1, max: 4, step: 1, default: 1},
                canvasred = 255
                canvasgreen = 255
                canvasblue = 255

            #if label != oldstatus:
            import threading
            first_thread = threading.Thread(target=run_server)
            second_thread = threading.Thread(target=run_browser)
            first_thread.start()
            time.sleep(1)
            second_thread.start()

            time.sleep(12.62)
            
            ###graph
            # x_graph = [0]
            # counter_graph = []
            # y_graph = [0]
            # for emotion in result_data_graph:
            #     a_graph = len(counter_graph)
            #     x_graph.append(a_graph)
            #     counter_graph.append(emotion)
            #     if emotion == 'Neutral':
            #         y_graph.append(0)
            #     elif emotion == 'Positive':
            #         y_graph.append(1)
            #     else:
            #         y_graph.append(-1)

            # plt.plot(x_graph, y_graph)
            # plt.savefig('graph.png')

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #create_log(time,result_data)
        break
cap.release()
cv2.destroyAllWindows()



