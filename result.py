import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision
#from hsvfilter import HsvFilter 

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#WindowCapture.list_window_names()
#exit()
#vision_tomato = Vision('heyday_carrot_processed.jpg')
#vision_tomato.init_control_gui()
#hsv_filter = HsvFilter(18, 255, 255, 31, 255, 255, 38, 0, 50, 0)
wincap = WindowCapture('Bluestacks')

cascade_carrot = cv.CascadeClassifier('cascade/cascade.xml')

vision_carrot = Vision(None)

loop_time = time()
while(True):

    screenshot = wincap.get_screenshot()
    #processed_image = vision_tomato.apply_hsv_filter(screenshot, hsv_filter)
    #carrot = cv.imread('heyday_screenshot.jpg', cv.IMREAD_UNCHANGED)
    # object detections
    #cv.imshow('Unprocessed', screenshot)
    rectangles = cascade_carrot.detectMultiScale(screenshot)

    detection_image = vision_carrot.draw_rectangles(screenshot, rectangles)

    #output_image = vision_tomato.draw_rectangles(screenshot, rectangles)

    cv.imshow('Matches', detection_image)

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press(q) = exit
    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break

    # press(f) = save screenshot to positive folder (include object)
    elif key == ord('f'):
        cv.imwrite('positive/{}.jpg'.format(loop_time), screenshot)

    # press(d) = save screenshot to negative folder (exclude object)
    elif key == ord('d'):
        cv.imwrite('negative/{}.jpg'.format(loop_time), screenshot)

print('Done.')