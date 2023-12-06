import os
import cv2
import numpy as np

from volume import draw_volume_and_calculate

folder = "./CT/"

if not os.path.exists(folder):
    raise FileExistsError

only_img3 = sorted(f for f in os.listdir(folder) if f.startswith('IMG-0003'))

pts = []

for index, filename in enumerate(only_img3):
    if index < 60 or index > 62:
        continue
    original = cv2.imread(os.path.join(folder, filename))
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3)), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    empty_frame = np.zeros_like(gray)
    empty_frame = cv2.drawContours(empty_frame, [max_contour], -1, 255, -1)
    prepared = cv2.bitwise_and(gray, gray, mask=empty_frame)





    mask = np.zeros_like(gray)
    mask[(0 < prepared) & (prepared < 50)] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
    mask = cv2.dilate(mask, np.ones((3, 3)), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    max_contour = max(contours, key=cv2.contourArea)

    output = cv2.bitwise_and(prepared, prepared, mask=mask)
    cv2.drawContours(output, contours, -1, 255, 2)

    mm_scale = 0.8
    for i in range(0, len(contours)):
        for j in range(0, len(contours[i])):
            for g in range(0, len(contours[i][j])):
                pts.append(np.array(list([pixel * mm_scale for pixel in contours[i][j][g]]) + [0.75 * index]))

    # cv2.imshow('Result', np.hstack([prepared, output]))
    # cv2.waitKey(0)
pts = np.array(pts)
print(pts)
draw_volume_and_calculate(pts)