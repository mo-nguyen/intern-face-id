# import cv2 as cv
# import numpy as np
# from numpy.lib import median
# from PIL import Image
# img_path = 'emily.jpg'

# img = cv.imread(img_path)
# def resize(img, resolution):
#     height = img.shape[0]
#     width = img.shape[1]
#     scale = resolution/height
#     new_height = int(height*scale)
#     new_width = int(width*scale)
#     img = cv.resize(img, (new_width, new_height))
#     return cv.resize(img, (width, height))

# # new_img = resize(img, 45)
# new_img = img

# gray_img = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)

# # face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
# face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt.xml")
# # eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
# eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
# detected_faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

# for (c, r, w, h) in detected_faces:
#     # cv.rectangle(new_img, (c, r), (c+w, r+h), (0, 0, 255), 2)
#     roi_gray = gray_img[r:(r+h), c:(c+w)]
#     roi_color = new_img[r:(r+h), c:(c+w)]
#     detected_eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
#     index = 0
#     for (ec, er, ew, eh) in detected_eyes:
#         if index==0:
#             eye_1 = (ec, er, ew, eh)
#         elif index==1:
#             eye_2 = (ec, er, ew, eh)
#         # cv.rectangle(roi_color, (ec, er), (ec+ew, er+eh), (0, 0, 255), 2)
#         index +=1
#     if eye_1[0] < eye_2[0]:
#         left_eye = eye_1
#         right_eye = eye_2
#     else:
#         left_eye = eye_2
#         right_eye = eye_1
    
#     left_eye_center = ((left_eye[0]+left_eye[2]//2), (left_eye[1]+left_eye[3]//2))
#     left_eye_center_x = left_eye_center[0]
#     left_eye_center_y = left_eye_center[1]

#     right_eye_center = ((right_eye[0]+right_eye[2]//2), (right_eye[1]+right_eye[3]//2))
#     right_eye_center_x = right_eye_center[0]
#     right_eye_center_y = right_eye_center[1]

#     # cv.circle(roi_color, left_eye_center, 5, (255, 0, 0), -1)
#     # cv.circle(roi_color, right_eye_center, 5, (255, 0, 0), -1)
#     # cv.line(roi_color, left_eye_center, right_eye_center, (0, 255, 0), 3)

#     if left_eye_center_y > right_eye_center_y:
#         A = (right_eye_center_x, left_eye_center_y)
#         direction = -1
#     else:
#         A = (left_eye_center_x, right_eye_center_y)
#         direction = 1
#     # cv.circle(roi_color, A, 5, (255, 0, 0), -1)
#     # cv.line(roi_color, left_eye_center, A, (0, 255, 0), 3)
#     # cv.line(roi_color, right_eye_center, A, (0, 255, 0), 3)

#     delta_x = right_eye_center_x - left_eye_center_x
#     delta_y = right_eye_center_y - left_eye_center_y
#     angle = np.arctan(delta_y/delta_x)
#     angle = (angle*180)/np.pi
    
#     img_h, img_w = new_img.shape[:2]
#     center = (img_w//2, img_h//2)

#     M = cv.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv.warpAffine(new_img, M, (img_w, img_h))
#     gray_rotated = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)

# new_detect_face = face_cascade.detectMultiScale(gray_rotated)
# for (c1, r1, w1, h1) in new_detect_face:
#     # cv.rectangle(rotated, (c1, r1), (c1+w1, r1+h1), (0, 0, 255), 2)
#     crop_face = rotated[r1:(r1+h1), c1:(c1+w1)]

# gray_crop_face = cv.cvtColor(crop_face, cv.COLOR_BGR2GRAY)
# size = (200, 200)
# scaled_gray_crop_face = cv.resize(gray_crop_face, size)
# median_filter = cv.medianBlur(scaled_gray_crop_face, 3)
# gaussian_filter = cv.GaussianBlur(scaled_gray_crop_face, (3, 3), 0)

# compare = np.concatenate((gaussian_filter, median_filter, scaled_gray_crop_face), axis=0)

# clahe = cv.createCLAHE(clipLimit=2)
# clahe_gaussian = clahe.apply(gaussian_filter)
# clahe_median = clahe.apply(median_filter)
# clahe_gray = clahe.apply(scaled_gray_crop_face)

# compare_clahe = np.concatenate((clahe_gaussian, clahe_median), axis=0)
# compare_clahe = np.concatenate((compare_clahe, clahe_gray), axis=0)
# cv.imshow('compare clahe', compare_clahe)

# cv.imshow('compare', compare)
# cv.imshow('crop face', crop_face)
# cv.imshow('gray crop face', gray_crop_face)
# cv.imshow('rotated faces', rotated)

# cv.imshow('detected faces', new_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# recognizer = cv.face.LBPHFaceRecognizer_create()
# #recognizer.train(median_filter, np.array(['emily']))

import cv2
import numpy as np

path = 'emily.jpg'

img = cv2.imread(path)

print(img.shape)
width = img.shape[1]
height = img.shape[0]

img = img.reshape((width*height*3, 1))

print(img.shape)

a = np.random.randn(12288, 150)
b = np.random.randn(150, 45)

c = np.dot(a, b)

print(c.shape)