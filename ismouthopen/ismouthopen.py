from imutils import face_utils
from scipy.spatial import distance as dist
import dlib
import cv2
import os
from time import sleep
import numpy as np
import imutils

def eye_aspect_ratio(eye,hold):
    A = dist.euclidean(eye[14], eye[18])
    if A < int(hold):
        return ("close",A)
    else:
        return ("open",A)

def find_and_mark_face_parts_on_images(images,hold = 4):
    shape_predictor_path = "shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(shape_predictor_path)
    except:
        print("ERROR: Please download 'shape_predictor_68_face_landmarks.dat' from https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat")
        return

    for img in os.listdir(images):
        if img.endswith(".png") or img.endswith(".jpg"):
            image_path = os.path.join(images,img)

            image = cv2.imread("images/"+image_path)
            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 1)

            for (index,rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                for (name, (index, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    if name == "mouth":
                        ans = eye_aspect_ratio(shape[index:j],hold)
                        cv2.putText(image,f"{ans[0]}, euclidean: {round(float(ans[1]),2)}",(10,35),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

                        cv2.imshow("Image", image)
                        cv2.waitKey(0)

                    break

def find_and_mark_face_parts_on_webcam_or_video(video_link = 0,hold = 6):
    shape_predictor_path = "shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    cap = cv2.VideoCapture(video_link)

    while True:
        _,image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)

        for (index,rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (name, (index, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                if name == "mouth":
                    ans = eye_aspect_ratio(shape[index:j],hold)
                    cv2.putText(image,f"{ans[0]}, euclidean: {round(float(ans[1]),2)}",(10,35),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

                break

        cv2.imshow("Image", image)
        key = cv2.waitKey(10)
        sleep(0.1)

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break


find_and_mark_face_parts_on_images(os.getcwd())
