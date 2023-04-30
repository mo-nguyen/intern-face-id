import cv2 as cv
# from preprocess import *

folder = 'A'

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
count_imgs = 200
while(count_imgs > 0):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    print(len(faces))
    if len(faces)!=0:
        # align
        pass
    for (x, y, w, h) in faces:
        # print(ret, x, y, w, h)
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv.imshow('frame', frame)
    count_imgs -= 1
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()