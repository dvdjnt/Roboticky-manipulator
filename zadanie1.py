from matplotlib import pyplot as plt
import numpy as np

def translate(mode,input_matrix,d):
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
        return Tx.dot(input)
    if mode == "y":
        return Ty.dot(input)
    if mode == "z":
        return Tz.dot(input)
     
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
        return Rx.dot(input)
    if mode == "y":
        return Ry.dot(input)
    if mode == "z":
        return Rz.dot(input)   

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
B = np.array([
    [0],
    [0],
    [0],
    [1]
    ])

#A = B.dot(vector_one) # nasobenie matic
#print(A)


A = translate("x",B,5)
print(type(A))
"""
A = translate("y",A,8)
print(type(A))

A = translate("z",A,-2)

print(A)

xx = np.vstack([x[0::2],x[1::2]])
yy = np.vstack([y[0::2],y[1::2]])

fig = plt.figure()
for i in range(0, len(A), 2):
    plt.plot(A[i:i+2], A[i:i+2], 'ro-')



ax = plt.axes(projection="3d")
ax.plot3D(A[0],A[1],A[2],'red')
plt.show()
"""
