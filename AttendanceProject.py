import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from tkinter import *


path = "ImagesAttendance"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    currentImg = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgS)
    encodesCurrentFrame = face_recognition.face_encodings(imgS,facesCurrentFrame)

    for encodeFace,faceLoc in zip(encodesCurrentFrame,facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
        else:
            name = "Unknown"
            text = "PRESS S FOR RECOGNIZATION"
            warningText = cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                warningText = 0
                print("scan new face")
                window = Tk()
                window.title("NameSpace")
                window.geometry('200x200')
                lbl = Label(window, text="Enter The Name")
                lbl.grid(column=0, row=0)
                txt = Entry(window, width=30)
                txt.grid(column=0, row=1)

                def clicked():
                    newName = txt.get()
                    newName += ".jpg"
                    cv2.imwrite(filename=f'{path}/{newName}', img=img)
                    imgNew = cv2.imread(f'{path}/{newName}')
                    cv2.imshow("taken photo", imgNew);
                    myList = os.listdir(path)
                    print(myList)
                    images.append(imgNew)
                    classNames.append(os.path.splitext(myList[-1])[0])
                    imgNew = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encodingNewFace = face_recognition.face_encodings(imgNew)[0]
                    encodeListKnown.append(encodingNewFace)
                    window.destroy()

                btn = Button(window, text="Save", command=clicked)
                btn.grid(column=0, row=2)
                window.mainloop()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("webcam",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

