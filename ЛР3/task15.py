import numpy as np
from PIL import Image, ImageOps
from math import ceil, floor, sqrt, cos, sin, pi

H, W = 2000, 2000
u0, v0 = W/2, H/2
ax, ay = 1000, 1000
img_arr = np.zeros((H, W, 3), dtype=np.uint8)
z_buffer = np.full((H, W), np.inf, dtype=float)


def baricentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return (lambda0, lambda1, lambda2)


def get_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    return ((y1 - y2) * (z1 - z0) - (y1 - y0) * (z1 - z2), (z1 - z2) * (x1 - x0) - (x1 - x2) * (z1 - z0),
            (x1 - x2) * (y1 - y0) - (y1 - y2) * (x1 - x0))


l = [0, 0, 1]
def norm_scalar(n):
    global l
    return np.dot(l, n) / (sqrt((l[0] ** 2 + l[1] ** 2 + l[2] ** 2)) * sqrt((n[0] ** 2 + n[1] ** 2 + n[2] ** 2)))


def draw_triangle(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, color1, color2, color3):
    x0_proj, y0_proj = ax * x0 / z0 + u0, ay * y0 / z0 + v0
    x1_proj, y1_proj = ax * x1 / z1 + u0, ay * y1 / z1 + v0
    x2_proj, y2_proj = ax * x2/ z2 + u0, ay * y2 / z2 + v0

    xmin, xmax, ymin, ymax = floor(min(x0_proj, x1_proj, x2_proj)), ceil(max(x0_proj, x1_proj, x2_proj)), floor(min(y0_proj, y1_proj, y2_proj)), ceil(
        max(y0_proj, y1_proj, y2_proj))
    if (xmin < 0):
        xmin = 0
    if (xmax > W):
        xmax = W
    if (ymin < 0):
        ymin = 0
    if (ymax > H):
        ymax = H
    if norm_scalar(norm)>0:
        return
    else:
        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                coords = baricentric_coordinates(i, j, x0_proj, y0_proj, x1_proj, y1_proj, x2_proj, y2_proj)
                z_source = coords[0]*z0 + coords[1]*z1 + coords[2]*z2
                if coords[0] >= 0 and coords[1] >= 0 and coords[2] >= 0:
                    if z_source <= z_buffer[j][i]:
                        img_mat[j][i][0], img_mat[j][i][1], img_mat[j][i][2] = color1, color2, color3
                        z_buffer[j][i] = z_source

def turn(x, y, z, a, b, g, tx = 0, ty = 0, tz = 0):
    R = np.array([[1, 0, 0], [0, cos(a), sin(a)], [0, -sin(a), cos(a)]]).dot([[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]]).dot(np.array([[cos(g), sin(g), 0], [-sin(g), cos(g), 0], [0, 0, 1]]))
    return R.dot(np.array([x, y, z])) + np.array([tx, ty, tz])

# draw_triangle(img_arr, 250*2, 250*2, 500*2, 250*2 - 1, 250*2, 500*2, 255, 255, 255)
# image = Image.fromarray(img_arr, mode='RGB')
# image.show()
# draw_triangle(img_arr, 2500, 2500, 500*2, 250*2 - 1, 250*2, 500*2, 255, 255, 255)
# image = Image.fromarray(img_arr, mode='RGB')
# image.show()

file = open(r'C:\\Users\Nikita\Documents\КГ Лабы\labs_CG_4sem\Лр2\model_1.obj')

arr_n = []
arr = []
for s in file:
    spl_s = s.split()
    if (spl_s[0] == 'v'):
        arr.append([float(x) for x in spl_s[1:]])
    if (spl_s[0] == 'f'):
        arr_n.append([int(x.split('/')[0]) - 1 for x in spl_s[1:]])

for i in arr:
    i[0], i[1], i[2] = turn(i[0], i[1], i[2], 0, pi/2, 0, ty = -0.03, tz = 0.1)

for i in arr_n:
    x0, y0, z0 = arr[i[0]][0], arr[i[0]][1], arr[i[0]][2]
    x1, y1, z1 = arr[i[1]][0], arr[i[1]][1], arr[i[1]][2]
    x2, y2, z2 = arr[i[2]][0], arr[i[2]][1], arr[i[2]][2]

    norm = get_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    draw_triangle(img_arr, x0, y0, z0, x1, y1, z1, x2, y2, z2, -255*norm_scalar(norm), 0, 0)

image = Image.fromarray(img_arr, mode='RGB')
image = ImageOps.flip(image)
image.show()
image.save(r'C:\\Users\Nikita\Documents\КГ Лабы\labs_CG_4sem\ЛР3\Крол3.png')
