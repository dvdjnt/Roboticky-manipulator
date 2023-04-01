from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
import numpy as np

"""
author: David Janto
date: april 2023
"""


def deg2rad(deg):
    return (deg / 180) * np.pi

def calculateOrigin(origin, fi1, fi2, fi3, l1, l2, l3):
    return origin

def calculateA(origin, fi1, fi2, fi3, l1, l2, l3):
    A_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, l1],
        [0, 0, 0, 1]])
    return np.matmul(A_matrix, origin.T)


def calculateB(origin, fi1, fi2, fi3, l1, l2, l3):
    B_matrix = np.array([
        [np.cos(fi1), -np.sin(fi1) * np.cos(fi2), np.sin(fi1) * np.sin(fi2), np.sin(fi1) * np.sin(fi2) * l2],
        [np.sin(fi1), np.cos(fi1) * np.cos(fi2), -np.sin(fi2) * np.cos(fi1), -np.sin(fi2) * np.cos(fi1) * l2],
        [0, np.sin(fi2), np.cos(fi2), l2 * np.cos(fi2) + l1],
        [0, 0, 0, 1]
    ])
    return np.matmul(B_matrix, origin.T)


def calculateC(origin, fi1, fi2, fi3, l1, l2, l3):
    C_matrix = np.array([
        [np.cos(fi1), np.sin(fi1) * (-np.cos(fi2 + fi3)), np.sin(fi1) * np.sin(fi2 + fi3),
         np.sin(fi1) * (l3 * (np.sin(fi2 + fi3)) + l2 * np.sin(fi2))],
        [np.sin(fi1), np.cos(fi1) * np.cos(fi2 + fi3), -np.cos(fi1) * np.sin(fi2 + fi3),
         -l3 * np.cos(fi1) * np.sin(fi2 + fi3) - np.sin(fi2) * np.cos(fi1) * l2],
        [0, np.sin(fi2 + fi3), np.cos(fi2 + fi3), l3 * np.cos(fi2 + fi3) + l2 * np.cos(fi2) + l1],
        [0, 0, 0, 1]
    ])
    return np.matmul(C_matrix, origin.T)


def translate(mode, input_matrix, d):
    Tx = np.array([
        [1, 0, 0, d],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    Ty = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, d],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    Tz = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]])

    if mode == "x":
        return np.matmul(Tx, input_matrix.T)
    if mode == "y":
        return np.matmul(Ty, input_matrix.T)
    if mode == "z":
        return np.matmul(Tz, input_matrix.T)


def rotate(mode, input_matrix, angle):
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -1 * (np.sin(angle)), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]])

    Ry = np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-1 * (np.sin(angle)), 0, np.cos(angle), 0],
        [0, 0, 0, 1]])

    Rz = np.array([
        [np.cos(angle), -1 * (np.sin(angle)), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    if mode == "x":
        return np.matmul(Rx, input_matrix.T)
    if mode == "y":
        return np.matmul(Ry, input_matrix.T)
    if mode == "z":
        return np.matmul(Rz, input_matrix.T)


class ButtonHandle:
    i = 0

    def __init__(self, value):
        self.i = value

    def buttonFunc(self, event):
        if self.i == 0:
            print("turning off~")
            bVectors.label.set_text("ON")
            self.i = 1
        elif self.i == 1:
            print("turning on~")
            bVectors.label.set_text("OFF")
            self.i = 0

        plt.draw()


def calculatePoints(fi1, fi2, fi3, l1, l2, l3):
    fi1 = deg2rad(fi1)
    fi2 = deg2rad(fi2)
    fi3 = deg2rad(fi3)

    origin = np.array([0, 0, 0, 1])

    # postupne: -> posuvat si origin
    # transformujeme postupne body v serii
    # rotacie vzdy robime od predchadzajuceho bodu
    #
    # jedna matica pre kazdy bod: -> postupne pridavat transform k resultu (ale odzadu)
    # sekvencne ale odzadu, mozno inverzne ukony..

    # A - translate(z,l1) od origin
    # B - translate(z,l2)->rotate(x,fi2)->rotate(y,fi1) od A
    # C - translate(z,l3)->rotate(x,fi3)

    # currentPoint = origin
    # currentPoint = translate("z", currentPoint, l3)
    # currentPoint = rotate("x", currentPoint, fi3)
    #
    # currentPoint = translate("z", currentPoint, l2)
    # currentPoint = rotate("x", currentPoint, fi2)
    # currentPoint = rotate("z", currentPoint, fi1)
    #
    # currentPoint = translate("z", currentPoint, l1)
    # C = currentPoint
    # # ----------------------------------------------------
    # currentPoint = origin
    #
    # currentPoint = translate("z", currentPoint, l2)
    # currentPoint = rotate("x", currentPoint, fi2)
    # currentPoint = rotate("z", currentPoint, fi1)
    #
    # currentPoint = translate("z", currentPoint, l1)
    # B = currentPoint
    # # ---------------------------------------------------
    # currentPoint = origin
    # currentPoint = translate("z", currentPoint, l1)
    # A = currentPoint

    A = calculateA(origin, fi1, fi2, fi3, l1, l2, l3)
    B = calculateB(origin, fi1, fi2, fi3, l1, l2, l3)
    C = calculateC(origin, fi1, fi2, fi3, l1, l2, l3)

    return np.vstack((origin[:3], A[:3], B[:3], C[:3]))


def calculateCoordinateVectors(fi1, fi2, fi3, l1, l2, l3):
    # vectors of coordinate system
    d = 50  # scaling of vector
    relativeSystem = np.array([
        [d, 0, 0, 1],
        [0, d, 0, 1],
        [0, 0, d, 1],
    ])

    calculate_map = {0: calculateOrigin, 1: calculateA, 2: calculateB, 3: calculateC}

    xyzAll = np.empty((3, 4)) # preload

    for i in range(0, 4):   # xyz vektorov smeru pre vsetky body do output[]
        print()
        print(i)
        x = calculate_map[i](relativeSystem[0], fi1, fi2, fi3, l1, l2, l3)
        y = calculate_map[i](relativeSystem[1], fi1, fi2, fi3, l1, l2, l3)
        z = calculate_map[i](relativeSystem[2], fi1, fi2, fi3, l1, l2, l3)
        xyz = np.vstack((x, y, z))

        print('xyz')
        print(xyz)

        xyzAll = np.vstack((xyzAll, xyz))

    return xyzAll[3:]


def update(val, points):
    # points not used??? check!
    print('points more')
    print(points)
    points = calculatePoints(slider_fi1.val, slider_fi2.val, slider_fi3.val, l1, l2, l3)
    # coordinateVectors = calculateCoordinateVectors()
    # pridat vykreslenie vektorov
    # scatter navyse
    scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

    # scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black', s=35)

    print(points)

    for i in range(len(lines)):
        xs = [points[i, 0], points[i + 1, 0]]
        ys = [points[i, 1], points[i + 1, 1]]
        zs = [points[i, 2], points[i + 1, 2]]
        lines[i].set_data(xs, ys)
        lines[i].set_3d_properties(zs)
    fig.canvas.draw_idle()


# -------------
matrix_one = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
vector_one = np.array([
    [1],
    [1],
    [1]
])

# zadanie [mm]
l1 = 203  # 203
l2 = 178  # 381
l3 = 178  # 559

fi1_min = -90
fi1_max = 90

fi2_min = -55
fi2_max = 125

fi3_min = 0
fi3_max = 150  # zadanie

# uhly
fi1 = 0
fi2 = 0
fi3 = 0

# calculate init points
points = calculatePoints(fi1, fi2, fi3, l1, l2, l3)
print('points')
print(points)
vectors = calculateCoordinateVectors(fi1, fi2, fi3, l1, l2, l3)
print('vectors')
print(vectors)

# figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# plot points
scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black', s=35)
# scatter2 = ax.scatter(coordinateVectors[:, 0], coordinateVectors[:, 1], coordinateVectors[:, 2])

# lines between points
lines = []
linesVectors = []

# body
for i in range(len(points) - 1):
    xs = [points[i, 0], points[i + 1, 0]]
    ys = [points[i, 1], points[i + 1, 1]]
    zs = [points[i, 2], points[i + 1, 2]]
    line, = ax.plot(xs, ys, zs, color='orange')
    lines.append(line)
    print(i)
    print(xs)
    print()
    print(ys)
    print()
    print(zs)

# vektory smeru (posunute suradnicove systemy)
colors = ['red', 'green', 'blue']
j = 0
for i in range(0, len(points)):
    k = 0
    while k < 3:
        xs = [points[i][0], vectors[j][0]]
        ys = [points[i][1], vectors[j][1]]
        zs = [points[i][2], vectors[j][2]]
        line, = ax.plot(xs, ys, zs, color=colors[k])
        k = k + 1
        linesVectors.append(line)
        j = j + 1

    # xs = [points[i][0], coordinateVectors[i][0]]
    # ys = [points[i][1], coordinateVectors[i][1]]
    # zs = [points[i][2], coordinateVectors[i][2]]
    # print(i)
    # print(xs)
    # print()
    # print(ys)
    # print()
    # print(zs)
    # #
    # # xs = [coordinateVectors[i, 0], coordinateVectors[i + 1, 0]]
    # # ys = [coordinateVectors[i, 1], coordinateVectors[i + 1, 1]]
    # # zs = [coordinateVectors[i, 2], coordinateVectors[i + 1, 2]]
    # line, = ax.plot(xs, ys, zs)
    # lines.append(line)

# quit()

# sliders
sliders = []

# plt.axes([left, bottom, width, height], ...) - suradnice pre box
slider_fi1 = Slider(plt.axes([0.25, 0.06, 0.65, 0.03]), f'fi1', fi1_min, fi1_max, valinit=fi1)
slider_fi2 = Slider(plt.axes([0.25, 0.03, 0.65, 0.03]), f'f2', fi2_min, fi2_max, valinit=fi2)
slider_fi3 = Slider(plt.axes([0.25, 0, 0.65, 0.03]), f'f3', fi3_min, fi3_max, valinit=fi3)
sliders.append((slider_fi1, slider_fi2, slider_fi3))

# buttons
bax = plt.axes([0.05, 0.05, 0.15, 0.075])
bVectors = Button(bax, label="Vectors OFF")
print(bVectors.label.get_text)

bh = ButtonHandle(0)
bVectors.on_clicked(bh.buttonFunc)
print(bVectors.label.get_text)

for slider_fi1, slider_fi2, slider_fi3 in sliders:
    # posielame parametre kvoli shadowing
    slider_fi1.on_changed(lambda val: update(val, points))
    slider_fi2.on_changed(lambda val: update(val, points))
    slider_fi3.on_changed(lambda val: update(val, points))

# render
# plot config
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('roboticky manipulator 3D', loc='center', fontsize=30)
ax.axis('equal')
fig.subplots_adjust(top=0.9, bottom=0.15)
plt.show()
