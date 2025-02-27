import numpy as np
from PIL import Image

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
    derror = 0
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

def parser(s: str, model_v_matrix: list, model_p_matrix: list):
    s_splt = s.split()
        
    if s_splt[0] == 'v':
        model_v_matrix.append(tuple(map(lambda x: float(x), s_splt[1:])))
    elif s_splt[0] == 'f':
        s_splt_dash = []
        for ss in s_splt[1:]:
            s_splt_dash.append(tuple(map(lambda x: int(x), ss.split('/'))))
        model_p_matrix.append([model_v_matrix[s_splt_dash[0][0] - 1], model_v_matrix[s_splt_dash[1][0] - 1], model_v_matrix[s_splt_dash[2][0] - 1]])

def scale_round(num):
    return -round(num * 5000 + 500)

def scale_round_list(l):
    return list(map(lambda x: scale_round(x), l))

model_v_matrix = []
model_p_matrix = []

with open('model_1.obj') as file:
    for s in file:
        parser(s, model_v_matrix, model_p_matrix)

H, W = 1000, 1000

image_mtrx = np.zeros((H, W), dtype=np.uint8)

for polygon in model_p_matrix:
    vertex1, vertex2, vertex3 = scale_round_list(polygon[0]), scale_round_list(polygon[1]), scale_round_list(polygon[2])

    bresenham_line(image_mtrx, vertex1[0], vertex1[1], vertex2[0], vertex2[1], 255)
    bresenham_line(image_mtrx, vertex2[0], vertex2[1], vertex3[0], vertex3[1], 255)
    bresenham_line(image_mtrx, vertex1[0], vertex1[1], vertex3[0], vertex3[1], 255)

image = Image.fromarray(image_mtrx, mode='L')
image.save('6.png')
image.show()
