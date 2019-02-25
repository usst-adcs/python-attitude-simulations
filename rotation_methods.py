import numpy as np

def quaternions_to_dcm(b, scalar_last=False):
    """
    This function generates the DCM corresponding to the input quaternion coordinates
    :param b: quaternions
    :return: DCM
    """
    if scalar_last:
        dcm = [[b[3]**2 + b[0]**2 - b[1]**2 - b[2]**2, 2*(b[0]*b[1] + b[3]*b[2]), 2*(b[0]*b[2] - b[3]*b[1])],
               [2*(b[0]*b[1] - b[3]*b[2]), b[3]**2 - b[0]**2 + b[1]**2 - b[2]**2, 2*(b[1]*b[2] + b[3]*b[0])],
               [2*(b[0]*b[2] + b[3]*b[1]), 2*(b[1]*b[2] - b[3]*b[0]), b[3]**2 - b[0]**2 - b[1]**2 + b[2]**2]]
    else:
        dcm = [[b[0]**2 + b[1]**2 - b[2]**2 - b[3]**2, 2*(b[1]*b[2] + b[0]*b[3]), 2*(b[1]*b[3] - b[0]*b[2])],
               [2*(b[1]*b[2] - b[0]*b[3]), b[0]**2 - b[1]**2 + b[2]**2 - b[3]**2, 2*(b[2]*b[3] + b[0]*b[1])],
               [2*(b[1]*b[3] + b[0]*b[2]), 2*(b[2]*b[3] - b[0]*b[1]), b[0]**2 - b[1]**2 - b[2]**2 + b[3]**2]]
    return np.array(dcm)

def dcm_to_quaternions(dcm, scalar_last=False):
    """
    This function generates the quaternion coordinates corresponding to the DCM input
    note: this function has a SINGULARITY
    note: the positive value of b0 represents the 'short angle rotation'
    :param dcm: DCM
    :return: quaternion coordinate vector
    """
    b0 = (1/2)*np.sqrt(np.trace(dcm) + 1)
    if scalar_last:
        return np.array([(dcm[1, 2] - dcm[2, 1])/(4*b0), (dcm[2, 0] - dcm[0, 2])/(4*b0),
                         (dcm[0, 1] - dcm[1, 0]) / (4*b0), b0])
    else:
        return np.array([b0, (dcm[1, 2] - dcm[2, 1])/(4*b0), (dcm[2, 0] - dcm[0, 2])/(4*b0),
                         (dcm[0, 1] - dcm[1, 0]) / (4*b0)])

def sheppards_method(dcm, scalar_last=False):
    """
    This function generates the quaternion coordinates corresponding to the DCm input.

    This is called sheppard's method, which does not contain a singularity like the 'common' way to do this.

    :param dcm: DCM
    :return: quaternion coordinate vector
    """
    trace = np.trace(dcm)
    b_2 = (1/4)*np.array([(1+trace), (1 + 2*dcm[0, 0] - trace), (1 + 2*dcm[1, 1] - trace), (1 + 2*dcm[2, 2] - trace)])
    argmax = np.argmax(b_2)
    b = np.zeros(4)
    b[argmax] = np.sqrt(b_2[argmax])

    # There is probably a cleaner way to do these checks
    if argmax == 0:
        b[1] = (dcm[1, 2] - dcm[2, 1])/(4*b[0])
        b[2] = (dcm[2, 0] - dcm[0, 2])/(4*b[0])
        b[3] = (dcm[0, 1] - dcm[1, 0])/(4*b[0])
    elif argmax == 1:
        b[0] = (dcm[1, 2] - dcm[2, 1])/(4*b[1])
        b[2] = (dcm[0, 1] + dcm[1, 0])/(4*b[1])
        b[3] = (dcm[2, 0] + dcm[0, 2])/(4*b[1])
    elif argmax == 2:
        b[0] = (dcm[2, 0] - dcm[0, 2])/(4*b[2])
        b[1] = (dcm[0, 1] + dcm[1, 0])/(4*b[2])
        b[3] = (dcm[1, 2] + dcm[2, 1])/(4*b[2])
    elif argmax == 3:
        b[0] = (dcm[0, 1] - dcm[1, 0])/(4*b[3])
        b[1] = (dcm[2, 0] + dcm[0, 2])/(4*b[3])
        b[2] = (dcm[1, 2] + dcm[2, 1])/(4*b[3])

    # last step to make sure we have the 'short rotation'
    if b[0] < 0:
        b = -b

    # move the scalar component to the end if requested
    if scalar_last:
        b[0], b[1], b[2], b[3] = b[1], b[2], b[3], b[0]

    return b

if __name__ == '__main__':
    phi = np.deg2rad(179.99999)
    e1 = 0.3
    e2 = -0.2
    e3 = np.sqrt(1 - e1**2 - e2**2)
    b0 = np.cos(phi/2)
    b1 = e1 * np.sin(phi/2)
    b2 = e2 * np.sin(phi/2)
    b3 = e3 * np.sin(phi/2)
    b = np.array([b0, b1, b2, b3])

    dcm = quaternions_to_dcm(b)

    print(b)
    print(dcm)

    b1 = dcm_to_quaternions(dcm, True)
    b2 = sheppards_method(dcm, True)

    print(b1)
    print(b2)
    print(1e2*(b2/b1 - 1))
