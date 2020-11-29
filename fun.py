import numpy as np
import cv2
import math

def empty(a):
    pass


def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))

    imgDial = cv2.dilate(imgCanny,kernel,iterations=1)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)

    return imgThres


def getContours4(img,imgContour,min_vertexes=3,max_vertexes=4):
    contourList = []

    if(min_vertexes<4):
        contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    else:
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>20000:
            cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if len(approx) == 4 and min_vertexes==3:
                contourList.append(np.reshape(approx,(4,2)))
            elif   len(approx)>min_vertexes and  len(approx)<max_vertexes:
                contourList.append(approx)

    contourList = np.asarray(contourList)
    return contourList


def getContoursNoDraw(img):
    contourList = []

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) >3:
                contourList.append(approx)

    contourList = np.asarray(contourList)
    return contourList


def cutContours(img,cnt,contoursCutted):

    resultTab = []
    wycinekTab = []

    mask = np.zeros_like(img)

    cv2.drawContours(mask, cnt, -1, (255, 255, 255), -1)
    cv2.fillPoly(mask, pts=[cnt], color=(255, 255, 255))

    dystanse = distances( cnt)
    dystans_min = min_distance(cnt)/1.1


    for dys in (dystanse):
        if( dys[0] < dystans_min):
            mask = cv2.line(mask,(cnt[dys[1]][0][0],cnt[dys[1]][0][1]),(cnt[dys[2]][0][0],cnt[dys[2]][0][1]),(0,0,0),4)

    result = cv2.bitwise_and(img, mask)
    resultGrey = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    contours = getContoursNoDraw(resultGrey)



    for cnt_ in contours:
        contoursCutted.append(cnt_)


def distances(cnt):
    distances = []
    for i in range(len(cnt)):
        for j  in range(i+2,len(cnt)):
            distances.append([math.sqrt(  (cnt[i][0][1]-cnt[j][0][1] )**2  + (cnt[i][0][0]-cnt[j][0][0] )**2  ),i,j])

    return distances


def min_distance(cnt):
    distances = []
    for i in range(len(cnt)-1):
        distances.append(math.sqrt((cnt[i][0][1] - cnt[i+1][0][1]) ** 2 + (cnt[i][0][0] - cnt[i+1][0][0]) ** 2))
    distances.append(math.sqrt((cnt[i][0][1] - cnt[len(cnt)-1][0][1]) ** 2 + (cnt[i][0][0] - cnt[len(cnt)-1][0][0]) ** 2))

    distances=np.array(distances)
    return np.amin(distances)


def degrees(cnt):
    wsp=[]
    for i in range(len(cnt)):

        if(i+1!=len(cnt)):
            wsp.append( [math.atan( (cnt[i][0][1]-cnt[i+1][0][1])/(cnt[i][0][0]-cnt[i+1][0][0]) )*180/math.pi,i,i+1 ])
        else:
            wsp.append([math.atan((cnt[i][0][1] - cnt[0][0][1]) / (cnt[i][0][0] - cnt[0][0][0]))*180/math.pi,i,0 ])
    return wsp


def cutUsingDegrees(katy,cnt):
    katy = np.array(katy)
    dis = katy[np.argsort(katy[:,0])]

    vertexes=[]

    for i in range(1,len(dis)-1):
        if(   1.1*(dis[i-1][0]+90)>(dis[i][0]+90) and (dis[i][0]+90)*1.1>(dis[i+1][0]+90) ):
            vertexes.append( [dis[i-1][1],dis[i-1][2],dis[i][1],dis[i][2],dis[i+1][1],dis[i+1][2] ])
            break

    points  = []

    if(len(vertexes)>0):
        for i in range(len(vertexes[0])):
            points.append( cnt[int(vertexes[0][i])]  )

    return points


def getPseudoWarp(img,cnt,x, y, w, h ):

    resultTab =[]
    wycinekTab=[]
    mask = np.zeros_like(img)
    cv2.drawContours(mask, cnt, -1, (255,255,255), -1)
    cv2.fillPoly(mask, pts=[cnt], color=(255, 255, 255))

    result = cv2.bitwise_and(img, mask)
    wycinki=[]
    wycinek = cutUsingDegrees(degrees(cnt), cnt)

    if(len(wycinek)>0):

        wycinek = np.array(wycinek)
        mask0 = np.zeros_like(img)
        cv2.drawContours(mask0, wycinek, -1, (255, 255, 255), -1)
        cv2.fillPoly(mask0, pts=[wycinek], color=(255, 255, 255))
        result0 = cv2.bitwise_and(result, mask0)
        result0 = result0[y:y + h, x:x + w]


        mask3 = cv2.bitwise_xor(mask,mask0)
        img2 = cv2.bitwise_and(img,mask3)
        mask3Grey =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        cnt2 =getContoursNoDraw(mask3Grey)

        result2 = img2[y:y + h, x:x + w]


        resultTab.append(result0)
        wycinekTab.append(wycinek)


        for cnt3 in cnt2:

            if(len(cnt3)<6):
                resultTab.append(result2)
                wycinekTab.append(cnt3)
            else:

                wycinek2 = cutUsingDegrees(degrees(cnt3), cnt3)

                if (len(wycinek2) > 0):

                    wycinek2 = np.array(wycinek2)
                    mask2 = np.zeros_like(img)
                    cv2.drawContours(mask2, wycinek2, -1, (255, 255, 255), -1)
                    cv2.fillPoly(mask2, pts=[wycinek2], color=(255, 255, 255))
                    result3 = cv2.bitwise_and(result, mask2)
                    result3 = result3[y:y + h, x:x + w]


                    resultTab.append(result3)
                    wycinekTab.append(wycinek2)

                    mask3 = cv2.bitwise_or(mask0, mask2)
                    mask3 = cv2.bitwise_xor(mask, mask3)

                    img2 = cv2.bitwise_and(img, mask3)
                    mask3Grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    cnt4 = getContoursNoDraw(mask3Grey)

                    if(len(cnt4)>0):
                        result2 = img2[y:y + h, x:x + w]
                        resultTab.append(result2)
                        wycinekTab.append(cnt4[0])


        return resultTab,wycinekTab

    resultTab.append(result)
    wycinekTab.append(cnt)

    return resultTab,wycinekTab



def centeroidnp(arr,type):
    length = arr.shape[0]

    if(type==1):
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
    else:
        sum_x = np.sum(arr[:, 0, 0])
        sum_y = np.sum(arr[:, 0, 1])

    return  int(sum_x/length)-50, int(sum_y/length)


def reorder (myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)

    #print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)

    min_dif_point = myPoints[np.argmin(diff)]
    max_dif_point = myPoints[np.argmax(diff)]

    if( math.sqrt(  (max_dif_point[0]-myPointsNew[3][0][0])**2 + (max_dif_point[1]-myPointsNew[3][0][1])**2  )
            > math.sqrt(  (min_dif_point[0]-myPointsNew[3][0][0])**2 + (min_dif_point[1]-myPointsNew[3][0][1])**2  )) :

        myPointsNew[1] = min_dif_point
        myPointsNew[2] = max_dif_point
    else:
        myPointsNew[1] = max_dif_point
        myPointsNew[2] = min_dif_point

    return myPointsNew


def getWarp(biggest,img):

    biggest = reorder(biggest)

    widthImg =   800
    heightImg =  400

    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    #imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    #imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))

    return imgOutput


def makeWhite(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(imgGray, 100, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImage


def printSum(b,b_v,imgContour):
    sum = 0
    for i in range(len(b)):
        sum = sum + b[i] * b_v[i]
    cv2.putText(imgContour, "SUMA: "+str(sum), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 5)
    cv2.putText(imgContour, "SUMA: "+str(sum), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 4)



def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


#
# def getPseudoWarpBig(img,cnt,x, y, w, h ):
#
#     resultTab = []
#     wycinekTab = []
#
#     mask = np.zeros_like(img)
#     cv2.drawContours(mask, cnt, -1, (255, 255, 255), -1)
#     cv2.fillPoly(mask, pts=[cnt], color=(255, 255, 255))
#
#
#     dystanse = distances( cnt)
#     dystans_min = min_distance(cnt)/1.1
#
#     for dys in (dystanse):
#         if( dys[0] < dystans_min):
#             mask = cv2.line(mask,(cnt[dys[1]][0][0],cnt[dys[1]][0][1]),(cnt[dys[2]][0][0],cnt[dys[2]][0][1]),(0,0,0),5)
#
#     result = cv2.bitwise_and(img, mask)
#     resultGrey = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#
#
#
#     contours = getContoursNoDraw(resultGrey)
#
#     for i,cnt_ in enumerate(contours):
#
#         mask0 = np.zeros_like(img)
#         cv2.drawContours(mask0, cnt_, -1, (255, 255, 255), -1)
#         cv2.fillPoly(mask0, pts=[cnt_], color=(255, 255, 255))
#         result0 = cv2.bitwise_and(img,mask0)
#         result0= result0[y:y + h, x:x + w]
#
#         resultTab.append(result0)
#         wycinekTab.append(cnt_)
#
#     return resultTab,wycinekTab
