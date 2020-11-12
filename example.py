import deepeye
import cv2

eye_tracker = deepeye.DeepEye()

cap = cv2.VideoCapture(0)
ret = True
while ret == True:
    ret, frame = cap.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    coords = eye_tracker.run(frame_gray)

    im_with_keypoints = cv2.circle(frame,
                                   (int(coords[0]), int(coords[1])), 8,
                                   (0, 0, 255), 2)

    cv2.imshow('DeepEye', im_with_keypoints)

    cv2.waitKey(1)
