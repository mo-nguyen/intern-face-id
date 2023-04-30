import cv2 as cv
import os

# face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
# eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
name = 'mo'

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
folder_path = os.path.join('original_data/train', name)
os.mkdir(folder_path)
count_imgs = 1

while count_imgs <= 200:
    ret, frame = cap.read()
    cv.imshow('frame', frame)
    cv.imwrite(os.path.join(folder_path, name+ '_' + str(count_imgs) + '.jpg'), frame)
    count_imgs += 1
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()