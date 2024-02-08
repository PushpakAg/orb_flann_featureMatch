import cv2
import numpy as np

def trans_rot_vectors(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    keypoints1, descriptor1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptor2 = orb.detectAndCompute(gray2, None)
    matches = matcher.match(descriptor1, descriptor2)

    pt1 = []
    pt2 = []
    for i in matches:
        p1 = keypoints1[i.queryIdx].pt
        p2 = keypoints2[i.trainIdx].pt
        pt1.append(p1)
        pt2.append(p2)
    
    points1 = np.float32(pt1)
    points2 = np.float32(pt2)

    fund_matrix, ign = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    ess_matrix, ign = cv2.findEssentialMat(points1, points2)
    ign, r, t, ignre = cv2.recoverPose(ess_matrix, points1, points2)

    rot_vector, _ = cv2.Rodrigues(r) #this converts rot_matrix to rot_vector using rodrigues formula  no idea

    return t, rot_vector

# Load two frames
frame1 = cv2.imread(r"C:\Users\pushp\OneDrive\Pictures\Screenshots\Screenshot 2024-02-08 215404.png")
frame2 = cv2.imread(r"C:\Users\pushp\OneDrive\Pictures\Screenshots\Screenshot 2024-02-08 215409.png")

t, rot_vector = trans_rot_vectors(frame1, frame2)

# Print results
print("Translation vector" + "\n" , t)
print("Rotational vector" + "\n" , rot_vector)