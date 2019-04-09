import numpy as np
import cv2


#reading the video by using cv2 inbuilt function
cap = cv2.VideoCapture("Results_background.avi")

#capturing two frames
ret, frame_one = cap.read()
ret, frame_two = cap.read()

#converting to grayscale
frame_gray = cv2.cvtColor(frame_one, cv2.COLOR_BGR2GRAY)

#get good features to track
p0 = cv2.goodFeaturesToTrack(frame_gray, 60, 0.01, 10.0, False)

while ret:

    #gets the difference of frame one and two
    difference = cv2.absdiff(frame_one, frame_two)

    #grayscale
    grey = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    #threshold
    ret, thresh = cv2.threshold(grey, 20, 255, 0)

    #contours
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    #compute box around vehicles
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    #draw
    cv2.drawContours(frame_one, contours, -1, (255, 0, 0), 1)

    cv2.imshow("test", frame_one)

    #just to delay the frames
    k = cv2.waitKey(25) & 0xff
    if k == 27:
        break

    #re assigning frames
    frame_one = frame_two
    ret, frame_two = cap.read()

cv2.destroyAllWindows()
cap.release()



#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
#http://connor-johnson.com/2014/02/18/linear-regression-with-python/