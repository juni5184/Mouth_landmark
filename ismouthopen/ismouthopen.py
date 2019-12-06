from imutils import face_utils
from scipy.spatial import distance as dist
import dlib
import cv2
import os
from time import sleep
import numpy as np
import imutils
import argparse

# 윗입술, 아랫입술 각각 위아래 길이 재는 코드
# euclidean 거리를 이용함
def mouth_aspect_ratio(image, mouth):

    top_mouth = dist.euclidean(mouth[3], mouth[14])
    # print(str(mouth[14]), str(mouth[8]))
    cv2.line(image, tuple(mouth[3]), tuple(mouth[14]), (0, 0, 255), 3)

    bottom_mouth = dist.euclidean(mouth[18], mouth[9])
    cv2.line(image, tuple(mouth[18]), tuple(mouth[9]), (255, 0, 0), 3)

    return ("mouth distance check", top_mouth, bottom_mouth)

# 이미지에서 입벌린걸 찾는 함수
def find_and_mark_face_parts_on_images(images):
    print(images+"\images")
    image_dir= images+"\images"
    # 68 landmark 모델 이용해서 landmark detect
    shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(shape_predictor_path)
    except:
        print("ERROR: Please download 'shape_predictor_68_face_landmarks.dat' "
              "from https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat")
        return

    # dir 에 들어있는 이미지 불러오기
    for img in os.listdir(image_dir):
        print(img)
        # png 나 jpg 로 끝나는 파일이면 if 문에 들어감
        if img.endswith(".png") or img.endswith(".jpg"):
            # 이미지 이름 추출
            image_path = os.path.join(image_dir,img)
            #print(str(image_path))

            # 이미지 읽기, resize
            image = cv2.imread(image_path)
            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 1)

            for (index,rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # 코 맨위 점부터 턱 끝 점까지의 거리 계산
                face_distance = dist.euclidean(shape[8], shape[27])
                # 코 맨 위 점부터 턱 끝 점까지 선 긋기
                #cv2.line(image, tuple(shape[8]), tuple(shape[27]), (0, 255, 0), 3)


                # 왜인지는 모르겠는데 name이 계속 mouth 만 나온다
                for (name, (index, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    # 여기에서 name을 출력해봐도 계속 mouth만 출력됨
                    # 왜지???
                    if name == "mouth":

                        # 입 부분 추출해서 잘라내는 코드
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[index:j]]))
                        roi = image[y:y + h, x:x + w]
                        roi = imutils.resize(roi, width=256, inter=cv2.INTER_CUBIC)


                        ans = mouth_aspect_ratio(image, shape[index:j])
                        if(round(float(face_distance))/round(float(ans[1]))>10) :
                            print("top mouth weird / ", round(float(face_distance))/round(float(ans[1])))
                        else :
                            print("top mouth ok / ", round(float(face_distance)) / round(float(ans[1])))

                        if (round(float(face_distance)) / round(float(ans[2])) > 15):
                            print("bottom mouth weird / ", round(float(face_distance)) / round(float(ans[2])))
                        else:
                            print("bottom mouth ok / ", round(float(face_distance)) / round(float(ans[2])))


                        # 입 부분 blur 검사
                        fm= cv2.Laplacian(roi, cv2.CV_64F).var()
                        text = "Not Blurry"
                        if fm < 5:
                             text = "Blurry"

                        cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        print("{}: {:.2f}".format(text, fm))

                        print("=========================================")


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
