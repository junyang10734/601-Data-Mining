"""
Image Stitching Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random


# find keypoints and use SIFT descriptors to extract features for these keypoints
def sift(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray_img, None)
    kp = np.float32([item.pt for item in kp])
    return kp, des


# match keypoints between two images
def matchKeypoints(des1, des2):
    # use KNN-matching algorithm to get the two closest descriptor
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    knn_match = matcher.knnMatch(des1, des2, 2)  # return DMatch, including queryIdx，trainIdx，distance
    matches = []

    # calculate the ratio between the two closest distance
    # if the ratio is greater than the predetermined value, it is taken as the final match.
    for item in knn_match:
        if len(item) == 2 and item[0].distance < item[1].distance * 0.75:
            matches.append((item[0].trainIdx, item[0].queryIdx))
    return matches


# compute the homography matrix using RANSAC algorithm
def findHomography(kp1, kp2, matches):
    if len(matches) > 4:
        pt1 = np.float32([kp1[i] for (_, i) in matches])
        pt2 = np.float32([kp2[i] for (i, _) in matches])
        (M, status) = cv2.findHomography(pt1, pt2, cv2.RANSAC, 5.0)  # return the matches along with the homograpy matrix and status of each matched point
        return M
    else:
        return None


#  stitch the two given images
def stitch(img1, img2, HM):
    # Applies a perspective transformation to an image.
    wrap = cv2.warpPerspective(img1, HM, (img1.shape[1] + img2.shape[1], img1.shape[0]+img2.shape[0]))
    # stitch the left image
    wrap[0:img2.shape[0], 0:img2.shape[1]] = img2

    # remove the black border
    rows, cols = np.where(wrap[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    result = wrap[min_row:max_row, min_col:max_col, :]
    return result


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    kpl, desl = sift(left_img)
    kpr, desr = sift(right_img)
    matches = matchKeypoints(desr, desl)
    matrix = findHomography(kpr, kpl, matches)
    result = stitch(right_img, left_img, matrix)
    return result

    raise NotImplementedError


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')

    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg', result_image)

