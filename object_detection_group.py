import numpy as np
import cv2 as cv
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


map_img = cv.imread('heyday_test.jpg', cv.IMREAD_UNCHANGED)
corn_img = cv.imread('heyday_corntest.jpg', cv.IMREAD_UNCHANGED)    #ข้าวโพดโตแล้ว
bcorn_img = cv.imread('heyday_corn_.jpg', cv.IMREAD_UNCHANGED)       #ข้าวโพดเด็กน้อย


corn_w = corn_img.shape[1]
corn_h = corn_img.shape[0]

bcorn_w = bcorn_img.shape[1]
bcorn_h = bcorn_img.shape[0]

result = cv.matchTemplate(map_img, corn_img, cv.TM_CCOEFF_NORMED)   #ข้าวโพดโตแล้ว
result2 = cv.matchTemplate(map_img, bcorn_img, cv.TM_CCOEFF_NORMED) #ข้าวโพดเด็กน้อย

print(result)
print(result2)

threshold = 0.55    #ค่าความคล้ายของรูป(ค่ามาก = แม่นมาก = อาจจะตรวจเจอได้น้อย) #ค่าของข้าวโพดโตแล้ว
threshold2 = 0.4    #ค่าของข้าวโพดเด็กน้อย
locations = np.where(result >= threshold)
locations2 = np.where(result2 >= threshold2)
locations = list(zip(*locations[::-1]))
locations2 = list(zip(*locations2[::-1]))
print(locations)
print(locations2)

rectangle = []
rectangle2 = []
for loc in locations:
    rect = [int(loc[0]), int(loc[1]), corn_w, corn_h]
    rectangle.append(rect) 
print(rectangle)

for loc2 in locations2:
    rect2 = [int(loc2[0]), int(loc2[1]), bcorn_w, bcorn_h]
    rectangle2.append(rect2)
print(rectangle2)

rectangle, weight = cv.groupRectangles(rectangle, 1, 0.5)
print(rectangle)

rectangle2, weight = cv.groupRectangles(rectangle2, 1, 0.5)
print(rectangle2)

if len(rectangle):
    print('Found corn.')

    corn_w = corn_img.shape[1]
    corn_h = corn_img.shape[0]

    bcorn_w = bcorn_img.shape[1]
    bcorn_h = bcorn_img.shape[0]

    line_color = (0, 255, 0)
    line_type = cv.LINE_4

    #loop แม่งให้หมด หาที่อยู่ของ object แล้วก็วาดสี่เหลี่ยม
    for (x, y, w, h) in rectangle:
        #กำหนดตำแหน่งของกรอบ (box)
        top_left = (x,y)
        bottom_right = (x + w, y + h)
        #วาดกรอบ
        cv.rectangle(map_img, top_left, bottom_right, line_color, line_type)

    #หากรอบของข้าวโพดเด็กน้อย
    for (x, y, w, h) in rectangle2:
        top_left = (x,y)
        bottom_right = (x + w, y + h)
        #วาดกรอบ
        cv.rectangle(map_img, top_left, bottom_right, line_color, line_type)

    cv.imshow('Matches', map_img)
    cv.waitKey()

else:
    print('Corn not found.')