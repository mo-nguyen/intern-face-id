import cv2 as cv
import numpy as np
import preprocess

def capture(face_recognizer, name):
    cap = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
    while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        clahe = preprocess.clahe_gray(gray)
        filtered = preprocess.gaussian(clahe)
        detected_faces = face_cascade.detectMultiScale(filtered, 1.1, 5)
        try:
            for (x, y, w, h) in detected_faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                roi = filtered[y:(y+h), x:(x+w)]
                detected_eyes = eye_cascade.detectMultiScale(roi, 1.1, 5)
                if len(detected_eyes) == 2:
                    eye_1 = detected_eyes[0]
                    eye_2 = detected_eyes[1]
                    if eye_1[0] < eye_2[0]:
                        left_eye = eye_1
                        right_eye = eye_2
                    else:
                        left_eye = eye_2
                        right_eye = eye_1

                    left_eye_center = ((left_eye[0]+left_eye[2]//2), (left_eye[1]+left_eye[3]//2))
                    left_eye_center_x = left_eye_center[0]
                    left_eye_center_y = left_eye_center[1]

                    right_eye_center = ((right_eye[0]+right_eye[2]//2), (right_eye[1]+right_eye[3]//2))
                    right_eye_center_x = right_eye_center[0]
                    right_eye_center_y = right_eye_center[1]

                    if left_eye_center_y > right_eye_center_y:
                        A = (right_eye_center_x, left_eye_center_y)
                        direction = -1
                    else:
                        A = (left_eye_center_x, right_eye_center_y)
                        direction = 1

                    delta_x = right_eye_center_x - left_eye_center_x
                    delta_y = right_eye_center_y - left_eye_center_y
                    angle = np.arctan(delta_y/delta_x)
                    angle = (angle*180)/np.pi

                    height, width = filtered.shape[:2]
                    center = (width//2, height//2)

                    M = cv.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv.warpAffine(filtered, M, (width, height))
                
                final_img = preprocess.crop_n_resize(rotated, face_cascade)
                label, confidence=face_recognizer.predict(final_img)
                print ("Confidence :", confidence)
                print("label :", label)
                if confidence > 25:
                    predict_name = 'unknown'
                else:
                    predict_name = name[label]
                cv.putText(frame, predict_name, (x,y), cv.FONT_HERSHEY_DUPLEX, 1.2, (255, 0, 0), 2)
        except Exception as e:
            print(e)
        cv.imshow('frame', frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    face_recognizer=cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read('LBPHrecognizer_database.xml')
    name = {0: 'mo', 1: 'hoan',}
    capture(face_recognizer, name)
