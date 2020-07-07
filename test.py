import numpy as np
import cv2
from keras.models import model_from_json
from keras.models import load_model
#import imutils for displaying the images
import imutils
from playsound import playsound
#import datetime displaying the current date and time
import datetime
import pygame

import smtplib
from os.path import basename
from email.utils import formatdate
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.base import MIMEBase 
import time
import geocoder

def sendMail(FROM, PASS, TO, weapon, attachment = None):
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	server.login(FROM,PASS)
	msg = MIMEMultipart()
	msg['Subject'] = '[ALERT] ' + weapon + ' detected'
	msg['From'] =  FROM
	msg['To'] = TO
	g = geocoder.ip('me')
	body = 'A ' + weapon  + ' was detected around ' + str(formatdate(localtime = True)) + ' at  latitude: ' + str(g.latlng[0]) + " longitude: " + str(g.latlng[1])
	msg.attach(MIMEText(body, 'plain'))
	if(attachment != None):
		with open(attachment, 'rb') as fil:
			part = MIMEApplication(fil.read(), Name = basename(attachment))
		part['Content-Disposition'] = 'attachment; filename="%s"' % basename(attachment)
		msg.attach(part)
	server.sendmail(FROM, TO, msg.as_string())


width = 640
height = 480
threshold = 0.75 # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0

cap = cv2.VideoCapture(cameraNo)
cap.set(3,width)
cap.set(4,height)

#load model
model = model_from_json(open("model.json", "r").read())
#load weights
model1=model.load_weights('model.h5')

face_cascade= cv2.CascadeClassifier("harcacade_frontal_face.xml")


def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    # # ###########face part
    # ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    # if not ret:
    #     continue
    # gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    #
    # faces_detected = face_cascade.detectMultiScale(gray_img, 1.32, 5)
    #
    # for (x, y, w, h) in faces_detected:
    #     cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
    #     ray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
    #     ray = cv2.resize(roi_gray, (48, 48))

    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image",img)
    img = img.reshape(1,32,32,1)
    classIndex = int(model.predict_classes(img))
    # print(classIndex)
    predictions = model.predict(img)
    # print(predictions)
    probVal = np.amax(predictions)
    print(classIndex, probVal)
    max_index = np.argmax(predictions[0])

    weapon = ('granade', 'knife', 'gun', 'masked face', 'gun', 'gun', 'gun')
    predicted_weapon = weapon[max_index]



    if probVal > threshold:

        cv2.putText(imgOriginal, predicted_weapon + "   " + str(probVal),
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 1)
        cv2.putText(imgOriginal,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (10, imgOriginal.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1
                    )
        pygame.init()

        pygame.mixer.music.load("beep-06.mp3")
        # time.sleep(5) this was to make the sound delay but resulted in the slow performance


        pygame.mixer.music.play()


    cv2.imshow("Original Image", imgOriginal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

