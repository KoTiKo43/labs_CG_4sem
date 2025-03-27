import numpy as np
from PIL import Image, ImageOps
from math import ceil, floor, sqrt, cos, sin, pi

H, W = 2000, 2000
Ht, Wt = 1024,1024
u0, v0 = W/2, H/2
ax, ay = 1000, 1000
img_arr = np.zeros((H, W, 3), dtype=np.uint8)
img_arr_text = np.zeros((H, W,3), dtype=np.uint8)
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
                        intence = color1 * coords[0] + color2 * coords[1] + color3 * coords[2]
                        img_mat[j][i][0], img_mat[j][i][1], img_mat[j][i][2] = -255*intence, -100*intence, 0
                        z_buffer[j][i] = z_source

def turn(x, y, z, a, b, g, tx = 0, ty = 0, tz = 0):
    R = np.array([[1, 0, 0], [0, cos(a), sin(a)], [0, -sin(a), cos(a)]]).dot([[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]]).dot(np.array([[cos(g), sin(g), 0], [-sin(g), cos(g), 0], [0, 0, 1]]))
    return R.dot(np.array([x, y, z])) + np.array([tx, ty, tz])

def draw_texture(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, coords_t0, coords_t1, coords_t2, file_texture,i0,i1,i2):
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
                        textur_c = [round(Ht*(coords[0]*coords_t0[0]+coords[1]*coords_t1[0]+coords[2]*coords_t2[0])),
                                    round(Wt*(coords[0]*coords_t0[1]+coords[1]*coords_t1[1]+coords[2]*coords_t2[1]))]
                                                                     
                        color = file_texture.getpixel((textur_c[0],textur_c[1]))

                        img_mat[j][i] = (#-color[0]*(i0*coords[0]+i1*coords[1]+i2*coords[2]),
                                         #-color[1]*(i0*coords[0]+i1*coords[1]+i2*coords[2]),
                                         #-color[2]*(i0*coords[0]+i1*coords[1]+i2*coords[2]))
                                         color)
                        z_buffer[j][i] = z_source

file = open(r'C:\\Users\Nikita\Documents\КГ Лабы\labs_CG_4sem\Лр4\zayac.obj')
file_texture =Image.open(r'C:\\Users\Nikita\Documents\КГ Лабы\labs_CG_4sem\Лр4\zayac_texture.jpg')
file_texture=ImageOps.flip(file_texture)
arr_n = []
arr = []
arr_texture_coords = []
for s in file:
    spl_s = s.split()
    if (spl_s[0] == 'v'):
        arr.append([float(x) for x in spl_s[1:]])
    if (spl_s[0] == 'vt'):
        arr_texture_coords.append([float(x) for x in spl_s[1:]])
    if (spl_s[0] == 'f'):
        arr_n.append([(int(x.split('/')[0]) - 1, int(x.split('/')[1]) - 1) for x in spl_s[1:]])

vn_cal = np.zeros((len(arr),3))

for i in arr:
    i[0], i[1], i[2] = turn(i[0], i[1], i[2], 0, pi/2, 0, ty = -0.03, tz = 0.01)

for i in arr_n:
    x0, y0, z0 = arr[i[0][0]][0], arr[i[0][0]][1], arr[i[0][0]][2]
    x1, y1, z1 = arr[i[1][0]][0], arr[i[1][0]][1], arr[i[1][0]][2]
    x2, y2, z2 = arr[i[2][0]][0], arr[i[2][0]][1], arr[i[2][0]][2]

    norm = get_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    vn_cal[i[0][0]-1]+=norm
    vn_cal[i[1][0]-1]+=norm
    vn_cal[i[2][0]-1]+=norm

for i in range(len(vn_cal)):
    vn_cal[i] = vn_cal[i]/sqrt(vn_cal[i][0]**2+vn_cal[i][1]**2+vn_cal[i][2]**2)

for i in arr_n:    
    x0, y0, z0 = arr[i[0][0]][0], arr[i[0][0]][1], arr[i[0][0]][2]
    x1, y1, z1 = arr[i[1][0]][0], arr[i[1][0]][1], arr[i[1][0]][2]
    x2, y2, z2 = arr[i[2][0]][0], arr[i[2][0]][1], arr[i[2][0]][2]

    norm = get_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    draw_triangle(img_arr, x0, y0, z0, x1, y1, z1, x2, y2, z2, norm_scalar(vn_cal[i[0][0]-1]), norm_scalar(vn_cal[i[1][0]-1]), norm_scalar(vn_cal[i[2][0]-1]))
    draw_texture(img_arr_text, x0, y0, z0, x1, y1, z1, x2, y2, z2, 
                 (arr_texture_coords[i[0][1]][0], arr_texture_coords[i[0][1]][1]), 
                 (arr_texture_coords[i[1][1]][0], arr_texture_coords[i[1][1]][1]), 
                 (arr_texture_coords[i[2][1]][0], arr_texture_coords[i[2][1]][1]),
                 file_texture,
                 norm_scalar(vn_cal[i[0][0]-1]), norm_scalar(vn_cal[i[1][0]-1]), norm_scalar(vn_cal[i[2][0]-1]))

image = Image.fromarray(img_arr, mode='RGB')
image2 = Image.fromarray(img_arr_text, mode='RGB')
image = ImageOps.flip(image)
image2 = ImageOps.flip(image2)
image.show()
image2.show()
image.save(r'C:\\Users\Nikita\Documents\КГ Лабы\labs_CG_4sem\ЛР4\Крол.png')
image2.save(r'C:\\Users\Nikita\Documents\КГ Лабы\labs_CG_4sem\ЛР4\КролТекст.png')
