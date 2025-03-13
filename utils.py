import cv2
import numpy as np

def stackImages(scale,imgArray, labels=[]):

    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                if imgArray[x][y] is None:
                    imgArray[x][y] = np.zeros((height, width, 3), np.uint8)
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                        None, scale, scale
                    )
                if len(imgArray[x][y].shape) == 2:  # Grayscale image
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            if imgArray[x] is None:
                imgArray[x] = np.zeros((height, width, 3), np.uint8)
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]),
                                         None, scale, scale)
            if len(imgArray[x].shape) == 2:  # Grayscale image
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    if labels:
        labels = [str(label) for label in labels]
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for row in range(rows):
            for col in range(cols):
                label_index = row * cols + col
                if label_index < len(labels):
                    label = labels[label_index]
                    cv2.putText(ver, label, (col * eachImgWidth + 10, row * eachImgHeight + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver

def rectContour(contours):
    rectCon = []
    
    for i in contours:
        area = cv2.contourArea(i)
        # print("Area:",area)
        if area > 50:
            peri=cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            # print("Corner Pointer",approx)
            if len(approx)==4:
                rectCon.append(i)
    rectCon = sorted(rectCon,key=cv2.contourArea,reverse=True)
    return rectCon

def getCornerPoints(cont):
    peri=cv2.arcLength(cont,True)
    approx = cv2.approxPolyDP(cont,0.02*peri,True)
    return approx


def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),dtype=int)
    add = myPoints.sum(1)
    # print(add)
    # print(myPoints)
    myPointsNew[0] = myPoints[np.argmin(add)] #[0,0]
    myPointsNew[3] = myPoints[np.argmax(add)] #[w,h]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)] #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[0,h]
    # print(diff)

    return myPointsNew

def splitBoxes(img):
    rows = np.vsplit(img,5)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
    return boxes

def showAnswers(img,myIndex,grading,ans,questions,choices):
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)

    for x in range(0,questions):
        myAns = myIndex[x]
        cX = (myAns*secW)+secW//2
        cY = (x*secH)+secH//2

        if grading[x]==1:
            myColor = (0,255,0)
        else:
            myColor = (0,0,255)
            correctAns = ans[x]
            cv2.circle(img,((correctAns*secW)+secW//2,(x*secH)+secH//2) ,30,(0,250,0),cv2.FILLED)

        cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)

    return img
