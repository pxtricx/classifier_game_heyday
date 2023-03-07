import numpy as np
import cv2 as cv
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

map_img = cv.imread('heyday_screenshot.jpg', cv.IMREAD_UNCHANGED)
corn_img = cv.imread('heyday_corn.jpg', cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(map_img, corn_img, cv.TM_CCOEFF_NORMED)

#get best match position
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

print('best match top left position: %s' % str(max_loc))
print('best match confidence: %s' % max_val)

threshold = 0.8
if max_val >= threshold:
    print('Found needle.')

    #หามิติของรูป hero
    corn_w = corn_img.shape[1]
    corn_h = corn_img.shape[0]

    top_left = max_loc
    bottom_right = (top_left[0] + corn_w, top_left[1] + corn_h)

    cv.rectangle(map_img, top_left, bottom_right,
                    color=(0,255,0), thickness=2, lineType=cv.LINE_4)

    cv.imshow('Result',map_img)
    cv.waitKey()

else:
    print('Needle not found.')