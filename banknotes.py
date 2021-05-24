import numpy as np
import cv2
import os

#funkcje do przetwarzania zdjec
from fun import *
#funkcje do wykrywania banknotow
from fun_detect import *


filename = "1.jpg"
path="Resources/latwe10/"
path_save = "wyniki30v/"


###--- Brute-Force Matcher
path_queryImages = 'Resources/banknoty'
orb = cv2.ORB_create(nfeatures=1000)

# Import Images
imagesQuery = []
classNames = []
myList = os.listdir(path_queryImages)
for cl in myList:
    imgCur = cv2.imread(f'{path_queryImages}/{cl}', 0)
    imagesQuery.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])

desList = findDes(imagesQuery,orb)
###---


b_v = [10,20,50,100,200]
b = [0,0,0,0,0]     # tablica z iloscia odnalezionych poszczegolnych banknotow
bFinall = [0,0,0,0,0]


img = cv2.imread(path +filename)
img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


for hue in range(15):
    for sat in range(2):

        lower2 = np.array([hue*3,sat*3, 0])
        upper2 = np.array([179, 255, 255])
        mask2= cv2.inRange(imgHSV,lower2,upper2) #maska z petli
        imgResult= cv2.bitwise_and(img,img,mask=mask2)

        imgContour = img.copy()
        imgThres = preProcessing( imgResult )

        contours = getContours4(imgThres,imgContour) #czworokaty
        contoursNot4 = getContours4(imgThres,imgContour,4,90) #nie czworokaty


        contoursCutted=[]
        for cnt in contoursNot4:
             if(len(cnt)<9):
                contoursCutted.append(cnt)
             else:
                 cutContours(img.copy(),cnt,contoursCutted)

        imgsWarped = []
        for cnt in contours:
            imgsWarped.append(getWarp(cnt,img))

        for i, picture in enumerate(imgsWarped):
            a = banknotByORB(desList, orb, picture)
            if a >= 0:
                b[a] += 1
                cv2.putText(imgContour, str(b_v[a]), centeroidnp(contours[i],1), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 5)
                cv2.putText(imgContour, str(b_v[a]), centeroidnp(contours[i],1), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 4)


        for i, contour in enumerate(contoursCutted):
            x, y, w, h = cv2.boundingRect(contour)
            imgCpy = img.copy()

            pictures ,  edges =  getPseudoWarp(imgCpy,contour,x, y, w, h )

            for ind, pic in enumerate(pictures):
                edges1 = np.array(edges[ind])
                picture1 = np.array(pic)

                a = banknotByORB(desList, orb, picture1)
                if a >= 0:
                    b[a] += 1
                    cv2.putText(imgContour, str(b_v[a]), centeroidnp(edges1, 2), cv2.FONT_HERSHEY_COMPLEX,
                                1.5,
                                (0, 0, 0), 5)
                    cv2.putText(imgContour, str(b_v[a]), centeroidnp(edges1, 2), cv2.FONT_HERSHEY_COMPLEX,
                                1.5,
                                (0, 0, 255), 4)

        if(np.sum(b)>np.sum(bFinall)):
            bFinall=b
            printSum(bFinall, b_v, imgContour)
            imgStacked = stackImages(0.5, ([img, imgResult], [imgThres, imgContour]))
            #imgStacked = stackImages(1000/img.shape[1], ([img, imgContour]))
            try:
                imgStackedWarped = stackImages(0.3 , imgsWarped)
            except:
                pass

        b=[0,0,0,0,0]


try:
    cv2.imshow("stacked Images", imgStacked)
    #cv2.imwrite(path_save+"2"+filename, imgStacked)
except:
    pass
try:
    cv2.imshow("warped Images", imgStackedWarped)
except:
    pass

print("RESULT: |", bFinall[0],": 10PLN |",bFinall[1],": 20PLN |",bFinall[2],": 50PLN |",bFinall[3],": 100PLN |",
      bFinall[4],": 200PLN |")


cv2.waitKey(0)




