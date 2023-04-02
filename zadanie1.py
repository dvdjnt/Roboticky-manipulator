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
    fi1 = deg2rad(fi1)
    fi2 = deg2rad(fi2)
    fi3 = deg2rad(fi3)

    origin_matrix = np.array([
        [np.cos(fi1), -1 * (np.sin(fi1)), 0, 0],
        [np.sin(fi1), np.cos(fi1), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    return np.matmul(origin_matrix, origin.T)

def calculateA(origin, fi1, fi2, fi3, l1, l2, l3):
    fi1 = deg2rad(fi1)
    fi2 = deg2rad(fi2)
    fi3 = deg2rad(fi3)

    A_matrix = np.array([
        [np.cos(fi1), -np.sin(fi1), 0, 0],
        [np.sin(fi1), np.cos(fi1), 0, 0],
        [0, 0, 1, l1],
        [0, 0, 0, 1]])
    return np.matmul(A_matrix, origin.T)

def calculateB(origin, fi1, fi2, fi3, l1, l2, l3):
    fi1 = deg2rad(fi1)
    fi2 = deg2rad(fi2)
    fi3 = deg2rad(fi3)

    B_matrix = np.array([
        [np.cos(fi1), -np.sin(fi1) * np.cos(fi2), np.sin(fi1) * np.sin(fi2), np.sin(fi1) * np.sin(fi2) * l2],
        [np.sin(fi1), np.cos(fi1) * np.cos(fi2), -np.sin(fi2) * np.cos(fi1), -np.sin(fi2) * np.cos(fi1) * l2],
        [0, np.sin(fi2), np.cos(fi2), l2 * np.cos(fi2) + l1],
        [0, 0, 0, 1]
    ])
    return np.matmul(B_matrix, origin.T)

def calculateC(origin, fi1, fi2, fi3, l1, l2, l3):
    fi1 = deg2rad(fi1)
    fi2 = deg2rad(fi2)
    fi3 = deg2rad(fi3)

    C_matrix = np.array([
        [np.cos(fi1), np.sin(fi1) * (-np.cos(fi2 + fi3)), np.sin(fi1) * np.sin(fi2 + fi3),
         np.sin(fi1) * (l3 * (np.sin(fi2 + fi3)) + l2 * np.sin(fi2))],
        [np.sin(fi1), np.cos(fi1) * np.cos(fi2 + fi3), -np.cos(fi1) * np.sin(fi2 + fi3),
         -l3 * np.cos(fi1) * np.sin(fi2 + fi3) - np.sin(fi2) * np.cos(fi1) * l2],
        [0, np.sin(fi2 + fi3), np.cos(fi2 + fi3), l3 * np.cos(fi2 + fi3) + l2 * np.cos(fi2) + l1],
        [0, 0, 0, 1]
    ])
    return np.matmul(C_matrix, origin.T)

def calculatePoints(fi1, fi2, fi3, l1, l2, l3):

    origin = np.array([0, 0, 0, 1])

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

def calculateWorkspaceXY(fi1_max, fi1_min, fi2_max, fi2_min, fi3_max, fi3_min, l1, l2, l3):
    pointsXY = []
    origin = np.array([0, 0, 0, 1])

    # perioda vzorkovania (v uhloch)
    T = 3

    # zhora, vpred
    # +T kvoli poslednemu cislu (cely kruh)
    # indexy idu v smere hodinkovych ruciciek
    for i in range(fi1_min, fi1_max+T, T):
        pointsXY.append(calculateC(origin, i, fi2_min, 0, l1, l2, l3))

    # zhora, vzadu
    for i in range(fi1_min, fi1_max+T, T):
        pointsXY.append(calculateC(origin, i, 90, 0, l1, l2, l3))

    # zhora, sprava
    for i in range(fi3_min, fi3_max, T):
        pointsXY.append(calculateC(origin, fi1_max, 90, i, l1, l2, l3))

    # zhora, zlava
    for i in range(fi3_min, fi3_max+T, T):
        pointsXY.append(calculateC(origin, fi1_min, 90, i, l1, l2, l3))

    # zhora, stred
    for i in range(fi2_min, fi2_max, T):
        pointsXY.append(calculateC(origin, fi1_min, fi2_max, i, l1, l2, l3))

    pointsXY = np.vstack((pointsXY))
    scatter = ax.scatter(pointsXY[:, 0], pointsXY[:, 1], 0, color='blue', s=10)

    return scatter

def calculateWorkspaceXZ(fi1_max, fi1_min, fi2_max, fi2_min, fi3_max, fi3_min, l1, l2, l3):
    pointsXZ = []
    origin = np.array([0, 0, 0, 1])

    # perioda vzorkovania (v uhloch)
    T = 3

    # zboku, polkruznica1
    for i in range(fi2_min, fi2_max, T):
        pointsXZ.append(calculateC(origin, 0, i, fi3_min, l1, l2, l3))

    # zboku, polkruznica2
    for i in range(fi3_min, fi3_max+T, T):
        pointsXZ.append(calculateC(origin, 0, fi2_max, i, l1, l2, l3))

    for i in reversed(range(fi2_min, fi2_max, T)):
        pointsXZ.append(calculateC(origin, 0, i, fi3_max, l1, l2, l3))

    for i in reversed(range(fi3_min, fi3_max, T)):
        pointsXZ.append(calculateC(origin, 0, fi2_min, i, l1, l2, l3))


    pointsXZ = np.vstack((pointsXZ))
    scatter = ax.scatter(0, pointsXZ[:, 1], pointsXZ[:, 2], color='purple', s=10)

    return scatter

class ButtonHandle:
    i = None
    name = None
    labelHandle = None
    linesHandle = None

    def __init__(self, value, name, labelHandle, linesHandle):
        self.i = value
        self.name = name
        self.labelHandle = labelHandle
        self.linesHandle = linesHandle

    def getName(self):
        return self.name

    def getState(self):
        if (self.i == 0):
            return False
        elif (self.i == 1):
            return True

    def buttonFunc(self, event):
        if self.i == 0:
            self.labelHandle.label.set_text(self.name+": ON")
            self.i = 1
            self.visibility(self.linesHandle, True)
            print("turning on~ i = " + str(self.i))
        elif self.i == 1:
            self.labelHandle.label.set_text(self.name + ": OFF")
            self.i = 0
            self.visibility(self.linesHandle, False)
            print("turning off~ i = " + str(self.i))

        fig.canvas.draw()

    def visibility(self, linesHandle, value):
        try:
            for i in range(len(linesHandle)):
                for j in range(len(linesHandle[i])):
                    linesHandle[i][j].set_visible(value)
        except:
            for j in range(len(linesHandle)):
                linesHandle[j].set_visible(value)



def update(val, points, lines):

    print('points')
    print(points)
    points = calculatePoints(slider_fi1.val, slider_fi2.val, slider_fi3.val, l1, l2, l3)
    vectors = calculateCoordinateVectors(slider_fi1.val, slider_fi2.val, slider_fi3.val, l1, l2, l3)

    scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

    print(points)

    for i in range(len(lines)):
        xs = [points[i, 0], points[i + 1, 0]]
        ys = [points[i, 1], points[i + 1, 1]]
        zs = [points[i, 2], points[i + 1, 2]]
        lines[i].set_data(xs, ys)
        lines[i].set_3d_properties(zs)

    j = 0
    for i in range(0, len(points)):
        k = 0
        while k < 3:
            xs = [points[i][0], vectors[j][0]]
            ys = [points[i][1], vectors[j][1]]
            zs = [points[i][2], vectors[j][2]]
            linesVectors[j].set_data(xs, ys)
            linesVectors[j].set_3d_properties(zs)
            k = k + 1
            j = j + 1

    fig.canvas.draw()

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

# figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# calculate init points
points = calculatePoints(fi1, fi2, fi3, l1, l2, l3)
print('points')
print(points)
vectors = calculateCoordinateVectors(fi1, fi2, fi3, l1, l2, l3)
print('vectors')
print(vectors)

# plot joints
scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black', s=35)

# lines between joints
lines = []
linesVectors = []

for i in range(len(points) - 1):
    xs = [points[i, 0], points[i + 1, 0]]
    ys = [points[i, 1], points[i + 1, 1]]
    zs = [points[i, 2], points[i + 1, 2]]
    line, = ax.plot(xs, ys, zs, color='orange', linewidth=3)
    lines.append(line)

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

# pracovny priestor
workspaceXZ_scatter = calculateWorkspaceXZ(fi1_max, fi1_min, fi2_max, fi2_min, fi3_max, fi3_min, l1, l2, l3)
workspaceXY_scatter = calculateWorkspaceXY(fi1_max, fi1_min, fi2_max, fi2_min, fi3_max, fi3_min, l1, l2, l3)

# sliders
sliders = []

# plt.axes([left, bottom, width, height], ...) - suradnice pre box
slider_fi1 = Slider(plt.axes([0.3, 0.06, 0.65, 0.03]), f'fi1', fi1_min, fi1_max, valinit=fi1)
slider_fi2 = Slider(plt.axes([0.3, 0.03, 0.65, 0.03]), f'fi2', fi2_min, fi2_max, valinit=fi2)
slider_fi3 = Slider(plt.axes([0.3, 0, 0.65, 0.03]), f'fi3', fi3_min, fi3_max, valinit=fi3)
sliders.append((slider_fi1, slider_fi2, slider_fi3))

# buttons
bax = plt.axes([0.02, 0.05, 0.2, 0.08])
bVectors = Button(bax, label='Vectors: ON')

bax = plt.axes([0.02, 0.2, 0.2, 0.08])
bWorkSpace = Button(bax, label='Workspace: ON')

bhvector = ButtonHandle(1, 'Vectors', bVectors, linesVectors)
bhworkspace = ButtonHandle(1, 'Workspace', bWorkSpace, [workspaceXY_scatter, workspaceXZ_scatter])

bVectors.on_clicked(bhvector.buttonFunc)
bWorkSpace.on_clicked(bhworkspace.buttonFunc)

for slider_fi1, slider_fi2, slider_fi3 in sliders:
    # posielame parametre kvoli shadowing
    slider_fi1.on_changed(lambda val: update(val, points, lines))
    slider_fi2.on_changed(lambda val: update(val, points, lines))
    slider_fi3.on_changed(lambda val: update(val, points, lines))

# plot config

ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_title('Roboticky manipulator 3D', loc='center', fontsize=30)
ax.axis('equal')
fig.subplots_adjust(top=0.9, bottom=0.15)

# render
plt.show()
print('wait')
