from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def deg2rad(deg):
    return (deg/180)*np.pi

def calculatePoints(pointMatrix, fi1, fi2, fi3):
    global l1, l2, l3

    origin = np.array([0, 0, 0, 1])

    A = rotate("z", origin, deg2rad(fi1))
    A = translate("z", A, l1)

    B = rotate("x", A, deg2rad(fi2))
    B = translate("z", B, l2)

    C = rotate("x", B, deg2rad(fi3))
    C = translate("z", C, l3)

    pointMatrix = np.vstack((origin[:3], A[:3], B[:3], C[:3]))

    return pointMatrix

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
fi3_max = 150

# uhly
fi1 = 0
fi2 = 0
fi3 = 0

points = []
points = calculatePoints(points, fi1, fi2, fi3)
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


# Update the plot whenever a slider value changes
def update(val):
    global points

    points = calculatePoints(points, slider_fi1.val, slider_fi2.val, slider_fi3.val)

    scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

    print(points)

    for i in range(len(lines)):
        xs = [points[i, 0], points[i+1, 0]]
        ys = [points[i, 1], points[i+1, 1]]
        zs = [points[i, 2], points[i+1, 2]]
        lines[i].set_data(xs, ys)
        lines[i].set_3d_properties(zs)
    fig.canvas.draw_idle()


for slider_fi1, slider_fi2, slider_fi3 in sliders:
    # posielame parametre kvoli shadowing
    slider_fi1.on_changed(update)
    slider_fi2.on_changed(update)
    slider_fi3.on_changed(update)

# plot config
plt.xlabel('X')
plt.ylabel('Y')
ax.set_title('roboticky manipulator 3D', loc='center', fontsize=25)
ax.axis('equal')
fig.subplots_adjust(top=0.9, bottom=0.15)

# Show the plot
plt.show()
