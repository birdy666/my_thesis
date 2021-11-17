import numpy as np
import math

def rotation2AxisAngle(R):
    trace = np.array(R).trace()
    """他有些旋轉矩陣可能是經度造成的問題會有trace < -1 which is not possible"""
    if trace < -1:
        trace = -1 + 0.000000001
    angle = min(math.acos((trace-1)/2), math.pi-0.000000001)

    denominator = math.sqrt((R[2][1] - R[1][2])**2+(R[0][2] - R[2][0])**2+(R[1][0] - R[0][1])**2)
    x = (R[2][1] - R[1][2])/denominator
    y = (R[0][2] - R[2][0])/denominator
    z = (R[1][0] - R[0][1])/denominator 
    axis = np.array([x,y,z])
    axis = axis / np.linalg.norm(axis, 2)

    return axis, angle

def rotation2so3(R):
    axis, angle = rotation2AxisAngle(R)
    return angle * axis

# Euler–Rodrigues formula
def so32rotation(so3):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    theta = min(math.pi, abs(math.sqrt(np.dot(so3, so3))))    
    axis = so3 / theta
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return ([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def AxisAngle2Quaternion(axis, theta):
    q0 = math.cos(theta/2)
    q1 = axis[0]*math.sin(theta/2)
    q2 = axis[1]*math.sin(theta/2)
    q3 = axis[2]*math.sin(theta/2)
    
    return [q0, q1, q2, q3]

def Quaternion2Rotationx(Q):
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    R = np.array([[r00, r01, r02],
                [r10, r11, r12],
                [r20, r21, r22]])
                            
    return R

def Rotation2Quaternion(R):
    axis, angle = rotation2AxisAngle(R)
    Q = AxisAngle2Quaternion(axis, angle)
    return Q


if __name__ == "__main__":
    R = [[0.7912924885749817, 0.044100284576416016, -0.6098452806472778], 
    [0.04430155083537102, -0.9989093542098999, -0.014752416871488094], 
    [-0.6098306775093079, -0.015343617647886276, -0.7923831939697266]]
    print(0.7912924885749817-0.9989093542098999-0.7923831939697266)
    trace = -1 + 0.000001
    angle = math.acos((trace-1)/2)
    print(angle)

    print((Rotation2Quaternion(R)))
   
    
    
