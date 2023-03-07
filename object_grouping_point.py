import numpy as np
import cv2 as cv
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def findClickPositions(corn_img_path, map_img_path, threshold=0.5, debug_mode=None):
    map_img = cv.imread(map_img_path, cv.IMREAD_UNCHANGED)
    corn_img = cv.imread(corn_img_path, cv.IMREAD_UNCHANGED)    #ข้าวโพดโตแล้ว

    corn_w = corn_img.shape[1]
    corn_h = corn_img.shape[0]

    method = cv.TM_CCOEFF_NORMED
    result = cv.matchTemplate(map_img, corn_img, method)   

    print(result)

    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))
    print(locations)
 
    rectangle = []

    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), corn_w, corn_h]
        rectangle.append(rect) 
    print(rectangle)


    rectangle, weight = cv.groupRectangles(rectangle, 1, 0.5)
    #print(rectangle)

    points = []
    if len(rectangle):
        print('Found corn.')


        line_color = (0, 255, 0)
        line_type = cv.LINE_4
        marker_color = (0, 0, 255)
        marker_type = cv.MARKER_CROSS

        #loop แม่งให้หมด หาที่อยู่ของ object แล้วก็วาดสี่เหลี่ยม
        for (x, y, w, h) in rectangle:

            center_x = x + int(w/2)
            center_y = y + int(h/2)
            
            points.append((center_x,center_y))

            if debug_mode == 'rectangle':
            
                #กำหนดตำแหน่งของกรอบ (box)
                top_left = (x,y)
                bottom_right = (x + w, y + h)
                #วาดกรอบ
                cv.rectangle(map_img, top_left, bottom_right, line_color, line_type)
            
            elif debug_mode == 'points':
                cv.drawMarker(map_img, (center_x, center_y), marker_color, marker_type)

     
        if debug_mode:    
            cv.imshow('Matches', map_img)
            cv.waitKey()

    return points

points = findClickPositions('heyday_corn.jpg', 'heyday_screenshot.jpg',  threshold=0.49, debug_mode='points')
print(points)
