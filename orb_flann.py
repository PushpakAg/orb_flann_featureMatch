import cv2

reference_image = cv2.imread(r"C:\Users\pushp\OneDrive\Pictures\Camera Roll\WIN_20231030_20_17_34_Pro.jpg")
reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
keypoints_reference, descriptors_reference = orb.detectAndCompute(reference_image_gray, None)

flann_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
flann = cv2.FlannBasedMatcher(flann_params, {})

myCam = cv2.VideoCapture(0) 
myCam.set(cv2.CAP_PROP_FRAME_WIDTH,720)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
myCam.set(cv2.CAP_PROP_FPS,30)
myCam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

while True:
    ret, frame = myCam.read()
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    keypoints_frame, descriptors_frame = orb.detectAndCompute(frame_grey, None)
    matches = flann.knnMatch(descriptors_reference, descriptors_frame, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2: 
            m, n = match_pair
            if m.distance < 0.75* n.distance:
                good_matches.append(m)

    result = cv2.drawMatches(reference_image, keypoints_reference, frame, keypoints_frame, good_matches, None)

    cv2.imshow('output_win', result)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

myCam.release()
cv2.destroyAllWindows()
