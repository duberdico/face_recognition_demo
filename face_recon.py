	
# import libraries
import imutils
from imutils import paths
from imutils import face_utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import dlib
import face_recognition
from scipy.spatial import distance as dist

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

	
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords


ears = np.full((50,1),np.nan)

# Initialize plot.
fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('Eyes')

lw = 3
alpha = 0.9
line, = ax.plot(np.array(range(50)),ears, c='b', lw=lw)
plt.xlim((0,50))
plt.grid()
plt.ion()
plt.show()


# initialize the video stream, then allow the camera sensor to warm up
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#the network has already been trained to create 128-d embeddings on a dataset of ~3 million images.
mode = 0
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print(ears.shape)
ear = np.full((1,1),np.nan)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = frame

    rects = detector(gray, 1)

    # Display the resulting frame
    boxes = face_recognition.face_locations(gray, model="hog")
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        if mode > 0:
            # show the face number
            cv2.putText(gray, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        if mode == 1:
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(gray, (x, y), 4, (0, 0, 255), -1)
        
        if mode == 2:
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            #   average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            ears = np.append(ears[1:],ear)
            line.set_ydata(ears)
            plt.ylim((0,np.nanmax(ears)))
            fig.canvas.draw()
        else:
            ears = np.full((50,1),np.nan)



    cv2.imshow('frame',gray)
    
    #encodings = face_recognition.face_encodings(gray, boxes)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('a'):
        mode = 0
    if key & 0xFF == ord('s'):
        mode = 1
    if key & 0xFF == ord('d'):
        mode = 2

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()