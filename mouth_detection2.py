# USAGE
# python mouth_detection.py -p "shape_predictor_68_face_landmarks.dat" -i "images/example_01.jpg"

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from matplotlib import pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the original input image and detect faces in the grayscale image
# cv2.imshow("Input", image)
rects = detector(gray, 2)

# loop over the face detections
for rect in rects:
	# extract the ROI of the *original* face, then align the face
	# using facial landmarks
	(x, y, w, h) = rect_to_bb(rect)
	# faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
	faceAligned = fa.align(image, gray, rect)

	# display the output images
	# cv2.imshow("Original", faceOrig)
	# cv2.imshow("Aligned", faceAligned)
	# cv2.waitKey(0)


# load the input image, resize it, and convert it to grayscale
image = imutils.resize(faceAligned, width=256)
gray = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

# ===========================image align===============================
# loop over the face detections
# detect face
roi = rects[0] # region of interest
shape = predictor(gray, roi)
shape = face_utils.shape_to_np(shape)

# extract mouth
mouth = shape[48:68]
top = min(mouth[:,1])
bottom = max(mouth[:,1])

# extend contour for masking
mouth = np.append(mouth, [ w-1, mouth[-1][1] ]).reshape(-1, 2)
mouth = np.append(mouth, [ w-1, h-1 ]).reshape(-1, 2)
mouth = np.append(mouth, [ 0, h-1 ]).reshape(-1, 2)
mouth = np.append(mouth, [ 0, mouth[0][1] ]).reshape(-1, 2)
contours = [ mouth ]

# generate mask
mask = np.ones((h,w,1), np.uint8(0)) * 255 # times 255 to make mask 'showable'
cv2.drawContours(mask, contours, -1, 0, -1) # remove below mouth

# apply to image
result = cv2.bitwise_and(image, image, mask = mask)
result = result[top:bottom, roi.left():roi.left()+roi.width()] # crop ROI
#cv2.imwrite('result.png', result);
cv2.imshow('masked image', result)
cv2.waitKey(0)
