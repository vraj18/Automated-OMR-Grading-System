import cv2
import numpy as np
import utils  # Ensure this module contains necessary functions like rectContour, getCornerPoints, etc.

wImg, hImg = 500, 500
questions = 5
choices = 5
ansArray = [1, 2, 0, 1, 3]

# Function to stack images in a grid
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
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0][0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

# Specify the path to the image file
img_path = "/Users/vrajborad/Desktop/VsCode/Projects/automated_OMR_Grading/4.png"

# Try to load the image
img = cv2.imread(img_path)
if img is None:
    print(f"Error: Unable to load image at '{img_path}'. Please check the file path.")
    exit()  # Handle file loading failure appropriately

# Preprocessing
img = cv2.resize(img, (wImg, hImg))
imgContours = img.copy()
imgFinal = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

try:
    # Finding all contours
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    # Find rectangles
    rectCon = utils.rectContour(contours)
    biggestContour = utils.getCornerPoints(rectCon[0])
    gradePoints = utils.getCornerPoints(rectCon[1])

    if biggestContour.size != 0 and gradePoints.size != 0:
        # Draw biggest contours
        cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
        cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

        biggestContour = utils.reorder(biggestContour)
        gradePoints = utils.reorder(gradePoints)

        # Perspective transformation to warp image
        pt1 = np.float32(biggestContour)
        pt2 = np.float32([[0, 0], [wImg, 0], [0, hImg], [wImg, hImg]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        imgWarpColour = cv2.warpPerspective(img, matrix, (wImg, hImg))

        ptG1 = np.float32(gradePoints)
        ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
        imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

        # Apply threshold
        imgWarpGray = cv2.cvtColor(imgWarpColour, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGray, 180, 255, cv2.THRESH_BINARY_INV)[1]

        # Split the image into boxes and get non-zero pixel counts for each
        boxes = utils.splitBoxes(imgThresh)
        myPixelValue = np.zeros((questions, choices))
        countC = 0
        countR = 0
        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            myPixelValue[countR][countC] = totalPixels
            countC += 1
            if countC == choices:
                countR += 1
                countC = 0

        # Find the selected choices based on the max pixel values
        myIndex = []
        for x in range(0, questions):
            arr = myPixelValue[x]
            myIndexVal = np.where(arr == np.amax(arr))
            myIndex.append(myIndexVal[0][0])

        # Grading
        grading = [1 if ansArray[x] == myIndex[x] else 0 for x in range(0, questions)]
        score = sum(grading) / len(grading) * 100
        print("Score:", score)

        # Displaying answers
        imgResult = imgWarpColour.copy()
        imgResult = utils.showAnswers(imgResult, myIndex, grading, ansArray, questions, choices)
        imgRawDrawing = np.zeros_like(imgWarpColour)
        imgRawDrawing = utils.showAnswers(imgRawDrawing, myIndex, grading, ansArray, questions, choices)

        # Inverse perspective transformation
        invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
        imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (wImg, hImg))

        # Adding score to the grade image
        imgRawGrade = np.zeros_like(imgGradeDisplay)
        cv2.putText(imgRawGrade, str(int(score)) + "%", (60, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (25, 240, 10), 3)

        invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
        imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (wImg, hImg))

        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

        # Stack and display all images
        stackedImages = stackImages(0.6, [
            [img, imgContours, imgBiggestContours],
            [imgWarpColour, imgThresh, imgFinal]
        ])
        cv2.imshow("All Images", stackedImages)

    else:
        print("Error: Could not find contours for processing.")

except Exception as e:
    print(f"Error: {e}")

# Save the result if 's' is pressed
if cv2.waitKey(0) & 0xFF == ord('s'):
    cv2.imwrite("FinalResult.jpg", imgFinal)

# Cleanup
cv2.destroyAllWindows()
