import cv2 as cv
import numpy as np

def align(gray, face_cascade, eye_cascade):
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in detected_faces:
        roi = gray[y:(y+h), x:(x+w)]
        detected_eyes = eye_cascade.detectMultiScale(roi, 1.3, 5)
        if len(detected_eyes)!=2:
            break
        else:
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
        return rotated

def filter(aligned):
    filtered = cv.medianBlur(aligned, 3)
    return filtered

def clahe_gray(gray):
    clahe = cv.createCLAHE(clipLimit=2)
    new_gray = clahe.apply(gray)
    return new_gray

def gaussian(img):
    filtered = cv.GaussianBlur(img, (3, 3), 0)
    return filtered

def crop_n_resize(filtered, face_cascade):
    detected_faces = face_cascade.detectMultiScale(filtered, 1.3, 5)
    for (x, y, w, h) in detected_faces:
        roi = filtered[y:(y+h), x:(x+w)]
    final_img = cv.resize(roi, (250, 250))
    return final_img