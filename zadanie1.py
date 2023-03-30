from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

"""
author: David Janto
date: april 2023
"""

def deg2rad(deg):
    return (deg/180)*np.pi

def calculateA(origin, l1):
    A_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, l1],
        [0, 0, 0, 1]])
    return np.matmul(A_matrix, origin.T)

def calculateB(origin, fi1, fi2, l2):
    B_matrix = np.array([
        [np.cos(fi1), -np.sin(fi1)*np.cos(fi2), np.sin(fi1)*np.sin(fi2), np.sin(fi1)*np.sin(fi2)*l2],
        [np.sin(fi1), np.cos(fi1)*np.cos(fi2), -np.sin(fi2)*np.cos(fi1), -np.sin(fi2)*np.cos(fi1)*l2],
        [0, np.sin(fi2), np.cos(fi2), l2*np.cos(fi2)+l1],
        [0, 0, 0, 1]
    ])
    return np.matmul(B_matrix, origin.T)

def calculateC(origin, fi1, fi2, fi3, l2, l3):
    C_matrix = np.array([
        [np.cos(fi1), np.sin(fi1)*(-np.cos(fi2+fi3)), np.sin(fi1)*np.sin(fi2+fi3), np.sin(fi1)*(l3*(np.sin(fi2+fi3))+l2*np.sin(fi2))],
        [np.sin(fi1), np.cos(fi1)*np.cos(fi2+fi3), -np.cos(fi1)*np.sin(fi2+fi3), -l3*np.cos(fi1)*np.sin(fi2+fi3)-np.sin(fi2)*np.cos(fi1)*l2],
        [0, np.sin(fi2+fi3), np.cos(fi2+fi3), l3*np.cos(fi2+fi3)+l2*np.cos(fi2)+l1],
        [0, 0, 0, 1]
    ])
    return np.matmul(C_matrix, origin.T)



# Update the plot whenever a slider value changes
def update(val, points):
    # points not used??? check!

    points = calculatePoints(slider_fi1.val, slider_fi2.val, slider_fi3.val, l1, l2, l3)

    scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

    print(points)

    for i in range(len(lines)):
        xs = [points[i, 0], points[i+1, 0]]
        ys = [points[i, 1], points[i+1, 1]]
        zs = [points[i, 2], points[i+1, 2]]
        lines[i].set_data(xs, ys)
        lines[i].set_3d_properties(zs)
    fig.canvas.draw_idle()

def calculatePoints(fi1, fi2, fi3, l1, l2, l3):
    fi1 = deg2rad(fi1)
    fi2 = deg2rad(fi2)
    fi3 = deg2rad(fi3)

    origin = np.array([0, 0, 0, 1])

    # vectors of coordinate system
    relativeSystem = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

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
    A = calculateA(origin, l1)
    B = calculateB(origin,fi1, fi2, l2)
    C = calculateC(origin, fi1, fi2, fi3, l2, l3)

    return np.vstack((origin[:3], A[:3], B[:3], C[:3]))

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
        [0, np.cos(angle), -1*(np.sin(angle)), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]])
    
    Ry = np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-1*(np.sin(angle)), 0, np.cos(angle), 0],
        [0, 0, 0, 1]])
    
    Rz = np.array([
        [np.cos(angle), -1*(np.sin(angle)), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    
    if mode == "x":
        return np.matmul(Rx, input_matrix.T)
    if mode == "y":
        return np.matmul(Ry, input_matrix.T)
    if mode == "z":
        return np.matmul(Rz, input_matrix.T)

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
l1 = 203 # 203
l2 = 178 # 381
l3 = 178 # 559

fi1_min = -90
fi1_max = 90

fi2_min = -55
fi2_max = 125

fi3_min = 0
fi3_max = 150   #zadanie

# uhly
fi1 = 0
fi2 = 0
fi3 = 0

points = calculatePoints(fi1, fi2, fi3, l1, l2, l3)
print(points)

# figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# plot points
scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# lines between points
lines = []
for i in range(len(points)-1):
    xs = [points[i, 0], points[i+1, 0]]
    ys = [points[i, 1], points[i+1, 1]]
    zs = [points[i, 2], points[i+1, 2]]
    line, = ax.plot(xs, ys, zs)
    lines.append(line)

# Define sliders for each point
sliders = []

# plt.axes([left, bottom, width, height], ...) - suradnice pre box
slider_fi1 = Slider(plt.axes([0.25, 0.06, 0.65, 0.03]), f'fi1', fi1_min, fi1_max, valinit=fi1)
slider_fi2 = Slider(plt.axes([0.25, 0.03, 0.65, 0.03]), f'f2', fi2_min, fi2_max, valinit=fi2)
slider_fi3 = Slider(plt.axes([0.25, 0, 0.65, 0.03]), f'f3', fi3_min, fi3_max, valinit=fi3)
sliders.append((slider_fi1, slider_fi2, slider_fi3))

for slider_fi1, slider_fi2, slider_fi3 in sliders:

    # posielame parametre kvoli shadowing / global word
    slider_fi1.on_changed(lambda val: update(val, points))
    slider_fi2.on_changed(lambda val: update(val, points))
    slider_fi3.on_changed(lambda val: update(val, points))

# plot config
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('roboticky manipulator 3D', loc='center', fontsize=25)
ax.axis('equal')
fig.subplots_adjust(top=0.9, bottom=0.15)

# Show the plot
plt.show()
