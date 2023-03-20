import numpy as np
import math
from scipy import special
import scipy


def matrix_inverse(m):
    d = len(m)
    m = 0.5 * np.add(m, np.transpose(m)) + 1e-8 * np.eye(d)
    m = np.transpose(np.linalg.cholesky(m))
    m = np.linalg.inv(m)
    return np.dot(m, np.transpose(m))


def kernel_fcn(X1, X2, sigma_f=0.5, l=2):
    sqdist = ((X1[:, :, None] - X2[:, :, None].T) ** 2).sum(1)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def integral_kernel_fcn(co1, co2, sigma_f=1, l=0.5):
    p0, p1, p2, p3, p4, p5, p6, p7 = [], [], [], [], [], [], [], []
    area1 = []
    area2 = []

    for i in co1:
        p0.append(i[0])
        p1.append(i[1])
        p4.append(i[2])
        p5.append(i[3])
        area1.append((i[1] - i[0]) * (i[3] - i[2]))
    for i in co2:
        p2.append(i[0])
        p3.append(i[1])
        p6.append(i[2])
        p7.append(i[3])
        area2.append((i[1] - i[0]) * (i[3] - i[2]))

    p0 = np.array([p0]).T
    p1 = np.array([p1]).T
    p2 = np.array([p2])
    p3 = np.array([p3])
    p4 = np.array([p4]).T
    p5 = np.array([p5]).T
    p6 = np.array([p6])
    p7 = np.array([p7])

    area1 = np.array([area1]).T
    area2 = np.array([area2])

    area = area1 * area2

    return covariance_cal(p0, p1, p2, p3, p4, p5, p6, p7, sigma_f, l) / area


def covariance_cal(p0, p1, p2, p3, p4, p5, p6, p7, sigma_f, l):
    def term_cal(a, b):
        return (b - a) * special.erf((1 / np.sqrt(2 * l**2)) * (a - b)) - np.exp(
            -0.5 * (a - b) ** 2 / l**2
        ) / np.sqrt(0.5 * np.pi / l**2)

    return (
        0.5
        * np.pi
        * l**2
        * sigma_f**2
        * (term_cal(p1, p3) - term_cal(p1, p2) - term_cal(p0, p3) + term_cal(p0, p2))
        * (term_cal(p5, p7) - term_cal(p5, p6) - term_cal(p4, p7) + term_cal(p4, p6))
    )
