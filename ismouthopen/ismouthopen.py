from imutils import face_utils
from scipy.spatial import distance as dist
import dlib
import cv2
import os
from time import sleep
import numpy as np
import imutils

def mouth_aspect_ratio(mouth,hold):
    A = dist.euclidean(mouth[14], mouth[18])
    #print(str(mouth[14]), str(mouth[8]))
    # int(hold) 는 항상 4?
    if A < int(hold):
        return ("close",A)
    else:
        return ("open",A)

def mouth_aspect_ratio2(mouth,hold):
    A = dist.euclidean(mouth[4], mouth[14])
    #print(str(mouth[14]), str(mouth[8]))
    # int(hold) 는 항상 4?
    if A < 15:
        return ("top mouth weird",A)
    else:
        return ("top mouth ok",A)


# 이미지에서 입벌린걸 찾는 함수
def find_and_mark_face_parts_on_images(images,hold = 4):

    shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(shape_predictor_path)
    except:
        print("ERROR: Please download 'shape_predictor_68_face_landmarks.dat' "
              "from https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat")
        return

    for img in os.listdir(images):
        #print(str(img))
        if img.endswith(".png") or img.endswith(".jpg"):
            image_path = os.path.join(images,img)
            #print(str(images))

            image = cv2.imread(image_path)
            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 1)

            for (index,rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                #print(str(shape))

                for (name, (index, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    if name == "mouth":
                        ans = mouth_aspect_ratio(shape[index:j],hold)
                        ans2 = mouth_aspect_ratio2(shape[index:j], hold)

                        #cv2.putText(image,f"{ans[0]}, euclidean: {round(float(ans[1]),2)}",(10,35),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                        cv2.putText(image, f"{ans2[0]}, euclidean: {round(float(ans2[1]),2)}", (10, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

                        # extract the ROI of the face region as a separate image
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[index:j]]))
                        roi = image[y:y + h, x:x + w]
                        roi = imutils.resize(roi, width=256, inter=cv2.INTER_CUBIC)

                        cv2.imshow("ROI", roi)
                        cv2.imshow("Image", image)
                        cv2.waitKey(0)

                    break




find_and_mark_face_parts_on_images(os.getcwd())


# # 웹캠이나 비디오에서 입이 벌려져 있는지 찾는 함수
# def find_and_mark_face_parts_on_webcam_or_video(video_link = 0,hold = 6):
#     shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
#
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(shape_predictor_path)
#
#     cap = cv2.VideoCapture(video_link)
#
#     while True:
#         _,image = cap.read()
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#         rects = detector(gray, 1)
#
#         for (index,rect) in enumerate(rects):
#             shape = predictor(gray, rect)
#             shape = face_utils.shape_to_np(shape)
#
#             for (name, (index, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
#                 if name == "mouth":
#                     ans = mouth_aspect_ratio(shape[index:j],hold)
#                     cv2.putText(image,f"{ans[0]}, euclidean: {round(float(ans[1]),2)}",(10,35),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
#
#                 break
#
#         cv2.imshow("Image", image)
#         key = cv2.waitKey(10)
#         sleep(0.1)
#
#         if key == 27:
#             cap.release()
#             cv2.destroyAllWindows()
#             break
