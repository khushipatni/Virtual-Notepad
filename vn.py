import numpy as np
import cv2
from collections import deque

#default called trackbar function
def setValues(x):
   print("")


# Creating the trackbars needed for adjusting the marker colour
cv2.namedWindow("Color detectors")
cv2.resizeWindow("Color detectors",750, 240)
cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180, setValues)
cv2.createTrackbar("Upper Sat", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Upper Val", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180, setValues)
cv2.createTrackbar("Lower Sat", "Color detectors", 72, 255, setValues)
cv2.createTrackbar("Lower Val", "Color detectors", 49, 255, setValues)


# Giving different arrays to handle colour points of different colour
blpoints = [deque(maxlen=1024)]
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
cpoints = [deque(maxlen=1024)]
ppoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
black_index = 0
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
cyan_index = 0
purple_index = 0

#The kernel to be used for dilation purpose
kernel = np.ones((5,5),np.uint8)

colors = [(0,0,0),(255, 0, 0), (0, 255, 0), (0, 0, 255),(0,255,255),(255,255,0),(255,0,255),]
colorIndex = 0

# Here is code for Canvas setup
paintWindow = np.zeros((471,700,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (100,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (120,1), (180,65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (200,1), (260,65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (280,1), (340,65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (360,1), (420,65), colors[3], -1)
paintWindow = cv2.rectangle(paintWindow, (440,1), (500,65), colors[4], -1)
paintWindow = cv2.rectangle(paintWindow, (520,1), (580,65), colors[5], -1)
paintWindow = cv2.rectangle(paintWindow, (600,1), (660,65), colors[6], -1)

cv2.putText(paintWindow, "CLEAR", (45, 33), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLACK", (125, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (210, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (287, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (375, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (441, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "CYAN", (529, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "PURPLE", (601, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


# Loading the default webcam of PC.

cap = cv2.VideoCapture(0)

# Keep looping
while True:
    # Reading the frame from the camera
    ret, frame = cap.read()
    # Flipping the frame to see same side of yours
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame,(700,471))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
    u_sat = cv2.getTrackbarPos("Upper Sat", "Color detectors")
    u_val = cv2.getTrackbarPos("Upper Val", "Color detectors")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
    l_sat = cv2.getTrackbarPos("Lower Sat", "Color detectors")
    l_val = cv2.getTrackbarPos("Lower Val", "Color detectors")
    Upper_hsv = np.array([u_hue, u_sat, u_val])
    Lower_hsv = np.array([l_hue, l_sat, l_val])


    # Adding the colour buttons to the live frame for colour access
    frame = cv2.rectangle(frame, (40, 1),  (100, 65), (122, 122, 122), -1)
    frame = cv2.rectangle(frame, (120, 1), (180, 65), colors[0], -1)
    frame = cv2.rectangle(frame, (200, 1), (260, 65), colors[1], -1)
    frame = cv2.rectangle(frame, (280, 1), (340, 65), colors[2], -1)
    frame = cv2.rectangle(frame, (360, 1), (420, 65), colors[3], -1)
    frame = cv2.rectangle(frame, (440, 1), (500, 65), colors[4], -1)
    frame = cv2.rectangle(frame, (520, 1), (580, 65), colors[5], -1)
    frame = cv2.rectangle(frame, (600, 1), (660, 65), colors[6], -1)

    cv2.putText(frame, "CLEAR", (45, 33), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLACK", (125, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (210, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (287, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (375, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (441, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "CYAN", (529, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "PURPLE", (601, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)

    # Identifying the pointer by making its mask

    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=1)

    # Find contours for the pointer after idetifying it
    cnts,_ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Ifthe contours are formed
    if len(cnts) > 0:
        # sorting the contours to find biggest
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Calculating the center of the detected contour
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # Now checking if the user wants to click on any button above the screen
        if center[1] <= 65:
            if 40 <= center[0] <= 100: # Clear Button
                blpoints = [deque(maxlen=512)]
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]
                cpoints = [deque(maxlen=512)]
                ppoints = [deque(maxlen=512)]

                black_index = 0
                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0
                cyan_index = 0
                purple_index = 0

                paintWindow[67:, :, :] = 255
            elif 120 <= center[0] <= 180:
                    colorIndex = 0 # Black
            elif 200 <= center[0] <= 260:
                    colorIndex = 1 # blue
            elif 280 <= center[0] <= 340:
                    colorIndex = 2 # green
            elif 360 <= center[0] <= 420:
                colorIndex = 3  # Red
            elif 440 <= center[0] <= 500:
                colorIndex = 4  # yellow
            elif 520 <= center[0] <= 580:
                colorIndex = 5  # cyan
            elif 581 <= center[0] <= 600:
                colorIndex = 6  # purple
        else :
            if colorIndex == 0:
                blpoints[black_index].appendleft(center)
            elif colorIndex == 1:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 2:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 3:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 4:
                ypoints[yellow_index].appendleft(center)
            elif colorIndex == 5:
                cpoints[cyan_index].appendleft(center)
            elif colorIndex == 6:
                ppoints[purple_index].appendleft(center)
    # Append the next dequeues when nothing is detected to avoid messing up
    else:
        blpoints.append(deque(maxlen=512))
        black_index += 1
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1
        cpoints.append(deque(maxlen=512))
        cyan_index += 1
        ppoints.append(deque(maxlen=512))
        purple_index += 1

  # Draw lines of all the colors on the canvas and frame
    points = [blpoints,bpoints, gpoints, rpoints, ypoints, cpoints, ppoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Show all the windows
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)
    cv2.imshow("mask", Mask)

    img_counter = 0
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, paintWindow)
        # print("{} written!".format(img_name))
        img_counter += 1


# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()