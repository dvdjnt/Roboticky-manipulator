from matplotlib import pyplot as plt
import numpy as np

def getSizeOf(object):
    counter = 0
    for item in object:
        counter+=1
    return counter

def translate(mode,input_matrix,d):
    print('translate '+mode+', d:'+str(d))
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
        return Tx.dot(input_matrix)
    if mode == "y":
        return Ty.dot(input_matrix)
    if mode == "z":
        return Tz.dot(input_matrix)
     
def rotate(mode,input_matrix,angle):
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
        return Rx.dot(input_matrix)
    if mode == "y":
        return Ry.dot(input_matrix)
    if mode == "z":
        return Rz.dot(input_matrix)

# [mm]
l1 = 203
l2 = 178
l3 = 178

fi1_min = -90
fi1_max = 90

fi2_min = -55
fi2_max = 125

fi3_min = 0
fi3_max = 150

# ----------------

fi1 = 80
fi2 = 100
fi3 = 90

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

# --------------
origin = np.array([
    [0],
    [0],
    [0],
    [1]
    ])


A = translate("z", origin, l1)
print('A')
print(A)

B = rotate("x", A, fi2)
B = translate("z", B, l2)
print('B')
print(B)

C = rotate("x", B, fi3)
C = translate("z", C, l3)
print('C')
print(C)

#xx = np.vstack([x[0::2], x[1::2]])
#yy = np.vstack([y[0::2], y[1::2]])

fig = plt.figure()
ax = plt.axes(projection="3d")

points = [origin, A, B, C]

sizeOfPoints = getSizeOf(points)

print('points indexing..')
print(points[0][0])
print(points[0][1])
print(points[0][2])


print('cycle')

for i in range(0, sizeOfPoints-1):
    # plot([A(x),B(x)],[A(y),B(y)],[A(z),B(z)])
    # for j in range(0,3):
    plt.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], [points[i][2], points[i+1][2]])


# for i in range(0, len(A), 2):
#     plt.plot(A[i:i+2], A[i:i+2], 'ro-')
#
#
ax.plot3D(A[0], A[1], A[2], 'red', label='robotic manipulator')
plt.show()

