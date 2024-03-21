import cv2
import numpy as np

min_contour_width = 20  # 40
min_contour_height = 20  # 40
offset = 2  # 10
crossing_width = 960  # 550
matches = []
cars = 0
backSub = cv2.createBackgroundSubtractorMOG2()


def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture('video_preview.mp4')

if cap.isOpened():
    ret, frame1 = cap.read()
    crossing_width = int(cap.get(3) // 2)
    width, height = cap.get(3), cap.get(4)
else:
    ret = False
ret, frame1 = cap.read()

while ret:
    d = backSub.apply(frame1)
    open = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((4, 4)))
    ret, th = cv2.threshold(open, 254, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((4, 4)), 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)

        contour_valid = (w >= min_contour_width) and (
                h >= min_contour_height)

        if not contour_valid:
            continue
        cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

        cv2.line(frame1, (crossing_width, 0), (crossing_width, 1080), (0, 255, 0), 2)
        centroid = get_centroid(x, y, w, h)
        matches.append(centroid)
        cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
        cx, cy = get_centroid(x, y, w, h)
        for (x, y) in matches:
            if x < (crossing_width + offset) and x > (crossing_width - offset):
                cars = cars + 1
                matches.remove((x, y))

    # cv2.drawContours(frame,contours,-1,(0,0,255),2)
    cv2.putText(frame1, f"Total cars: {cars}", (15, int(height) - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("Original", frame1)
    # cv2.imshow("Difference", open)
    if cv2.waitKey(1) == 27:
        break
    ret, frame1 = cap.read()
# print(matches)
cv2.destroyAllWindows()
cap.release()