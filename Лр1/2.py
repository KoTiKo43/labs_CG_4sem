import numpy as np
from PIL import Image
from math import cos, sin, pi, sqrt

def dotted_line(image, x0, y0, x1, y1, color):
    count = 50
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def dotted_line_v2(image, x0, y0, x1, y1, color):
    count = sqrt((x0 - x1)**2 + (y0 - y1)**2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def x_loop_line(image, x0, y0, x1, y1, color):
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def x_loop_line_hotfix_1(image, x0, y0, x1, y1, color):
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def x_loop_line_hotfix_2(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < (y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

def x_loop_line_v2(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

def x_loop_line_no_y_calc(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if (derror > 0.5):
            derror -= 1.0
            y += y_update


def x_loop_line_no_y_calc_v2_for_some_unknown_reason(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    y = y0
    dy = 2 * (x1 - x0) * abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if (derror > 2 * (x1 - x0) * 0.5):
            derror -= 2 * (x1 - x0) * 1.0
            y += y_update

def bresenham_line(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update

H, W = 200, 200

x0, y0 = 100, 100
radius = 95
image_mtrx = np.zeros((H, W), dtype=np.uint8)

for i in range(13):
    x = round(100 + radius * cos(2 * pi * i / 13))
    y = round(100 + radius * sin(2 * pi * i / 13))
#    dotted_line(image_mtrx, x0, y0, x, y, 255)
#    dotted_line_v2(image_mtrx, x0, y0, x, y, 255)
#    x_loop_line(image_mtrx, x0, y0, x, y, 255)
#    x_loop_line_hotfix_1(image_mtrx, x0, y0, x, y, 255)
#    x_loop_line_hotfix_2(image_mtrx, x0, y0, x, y, 255)
#    x_loop_line_v2(image_mtrx, x0, y0, x, y, 255)
#    x_loop_line_no_y_calc(image_mtrx, x0, y0, x, y, 255)
#    x_loop_line_no_y_calc_v2_for_some_unknown_reason(image_mtrx, x0, y0, x, y, 255)
    bresenham_line(image_mtrx, x0, y0, x, y, 255)


image = Image.fromarray(image_mtrx, mode='L')
image.save('2_9.png')
image.show()
