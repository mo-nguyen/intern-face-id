import cv2 as cv
import preprocess
import os
import numpy as np

def capture(name, main_dir):
    cap = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
    folder_path = os.path.join(main_dir, name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    count_imgs = 1
    while count_imgs <= 200:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        try:
            if len(detected_faces) == 1:
                (x, y, w, h) = detected_faces[0]
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                roi = gray[y:(y+h), x:(x+w)]
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

                    height, width = gray.shape[:2]
                    center = (width//2, height//2)

                    M = cv.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv.warpAffine(gray, M, (width, height))
                filtered = preprocess.filter(rotated)
                final_img = preprocess.crop_n_resize(filtered, face_cascade)
                img_path = os.path.join(folder_path, name+ '_' + str(count_imgs) + '.jpg')
                cv.imwrite(img_path, final_img)
                count_imgs += 1
        except Exception as e:
            print(e)
        cv.imshow('frame', frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def get_data(main_dir):
    faces = []
    ids = []
    folders = os.listdir(main_dir)
    for f in folders:
        folder_path = os.path.join(main_dir, f)
        imgs = os.listdir(folder_path)
        for i in imgs:
            img_path = os.path.join(folder_path, i)
            img = cv.imread(img_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces.append(gray)
            ids.append(int(f))
    return faces, ids

def train_classifier(faces, ids):                              
    face_recognizer=cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(ids))
    return face_recognizer

if __name__ == '__main__':
    
    main_dir = 'data/train'
    
    name = '0'
    capture(name, main_dir)

    faces, ids = get_data(main_dir)
    trained = train_classifier(faces, ids)
    trained.save('LBPHrecognizer_database.xml')