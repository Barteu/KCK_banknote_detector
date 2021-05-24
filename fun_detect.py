import numpy as np
import cv2
import math


def findDes(images,orb):
    desList = []
    for img0 in images:
        kp, des = orb.detectAndCompute(img0, None)
        desList.append(des)
    return desList


def findID(img, desList,orb, thres=11):

    img_array = []
    img_array.append( img.copy())
    img_array.append(cv2.flip(img,0))
    img_array.append(cv2.flip(img, 1))
    img_array.append(cv2.flip(img, -1))

    finalVal = -1

    for img1 in img_array:


        kp2, des2 = orb.detectAndCompute(img1, None)
        bf = cv2.BFMatcher()
        matchList = []

        try:
            for des in desList:
                matches = bf.knnMatch(des, des2, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
                matchList.append(len(good))
        except:
            pass

        if len(matchList) != 0:

            if max(matchList) > thres:
                finalVal = matchList.index(max(matchList))
                break

    return finalVal


def banknotByORB(desList,orb,img):

    img_original = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    id = findID(img, desList, orb)
    if id != -1:

        return math.floor(id/2)

    return -1
