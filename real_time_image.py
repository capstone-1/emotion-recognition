from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import os
import time
def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

# starting video streaming
start = time.time()
image_dir_path = 'videos/sample1_images'
emotionList = []
probList = []
for image_path in absoluteFilePaths(image_dir_path):
    if 'txt' in image_path:
        continue
    frame = cv2.imread(image_path)
    #reading the frame
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        emotionList.append(label)
        probList.append(emotion_probability)
    else: 
        emotionList.append('None')
        probList.append(0)
        continue
print(time.time()-start)
import datetime
image_interval = 1
time = datetime.datetime.strptime('00:00:00','%H:%M:%S')
with open(image_dir_path+'/emotions.txt','w') as file:
    for emotion, prob in zip(emotionList,probList):
        file.write(time.strftime("%H:%M:%S ")+emotion+' {:.2f}%'.format(prob*100) + '\n')
        time += datetime.timedelta(seconds=image_interval)
