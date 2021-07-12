import numpy as np
import math

def rotation2AxisAngle(R):
    trace = np.array(R).trace()
    angle = math.acos((trace-1)/2)

    denominator = math.sqrt((R[2][1] - R[1][2])**2+(R[0][2] - R[2][0])**2+(R[1][0] - R[0][1])**2)
    x = (R[2][1] - R[1][2])/denominator
    y = (R[0][2] - R[2][0])/denominator
    z = (R[1][0] - R[0][1])/denominator 
    axis = np.array([x,y,z])
    axis = axis / np.linalg.norm(axis, 2)

    return axis, angle

def rotation2lieAlgebra(R):
    axis, angle = rotation2AxisAngle(R)
    return angle * axis


if __name__ == "__main__":
    R = [[1,0,0],
        [0,0,1],
        [0,-1,0]]
    print(rotation2lieAlgebra(R))
    
