import os
from typing import *

import pandas as pd
import numpy as np
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException

baseDir = os.path.join(os.path.dirname(__file__), "..")
faultsDir = os.path.join(baseDir, "faults")
if not os.path.isdir(faultsDir):
    os.mkdir(faultsDir)
dfFaults = pd.read_excel(os.path.join(baseDir, "faults.xlsx"), engine='openpyxl')
dfStations = pd.read_csv(os.path.join(baseDir, "stations.csv"))
try:
    # clientUSGS = Client("USGS", timeout=30)
    # clientIRIS = Client("IRIS", timeout=30)
    pass
except FDSNException:
    print("Client not accessible, retry later.")

def angularMean(angles_deg: List[float]):
    N = len(angles_deg)
    angles_deg = np.array(angles_deg)
    mean_c = 1.0 / N * np.sum(np.exp(1j * angles_deg * np.pi/180.0))
    return np.angle(mean_c, deg=True)

def angularDiff(x, y):
    # suppose x, y both [0,360)
    xy = abs(x - y)
    return min(360 - xy, xy)

def mag2moment(x):
    # https://en.wikipedia.org/wiki/Moment_magnitude_scale
    return 10 ** (x * 3/2 + 16.1) / 1e7

def mag2mw(m, magType):

    # Braunmiller, J., & Nábělek, J. (2008).
    # Segmentation of the Blanco Transform Fault Zone from earthquake analysis:
    # Complex tectonics of an oceanic transform fault.
    # Journal of Geophysical Research: Solid Earth, 113(B7).
    # https://doi.org/10.1029/2007JB005213
    def mb2mw(x):
        return 1.32 + 0.81 * x

    def ms2mw(x):
        return 2.40 + 0.62 * x

    if magType == "mb":
        return mb2mw(m)
    elif magType == "ms":
        return ms2mw(m)
    else:
        return m

def isTransform(fm, criterion=25.0):
    # check rake angle, transform event should have small rake angle
    if fm == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]:
        return False
    if np.isnan(fm[0]):
        return False
    rake = fm[-1] % 180
    return rake <= criterion or (180.0 - rake) <= criterion

def moment2RuptureLength(mw):
    # Wells, D. L., & Coppersmith, K. J. (1994).
    # New empirical relationships among magnitude, rupture length, rupture width, rupture area, and surface displacement.
    # Bulletin of the Seismological Society of America, 84(4), 974–1002.

    # subsurface rupture length
    return 10 ** (-2.57 + 0.62 * mw)

def momentAlongStrike(xdist, xrs, mws, npts=500):
    # notice xdist in [km]
    arr = np.zeros(npts)
    dx = xdist / npts
    rr = np.linspace(0, 1, npts)
    for (xr, mw) in zip(xrs, mws):
        l = moment2RuptureLength(mw) # [km]
        lr = l / xdist
        left = max(0, xr - lr/2)
        right = min(1, xr + lr/2)
        ileft = np.argmax(rr >= left)
        iright = np.argmax(rr > right)
        arr[ileft: iright] += mag2moment(mw) / l / 1e3
    return arr, rr

def momentAlongStrikeEllipsoid(xdist, xrs, mws, npts=500):
    arr = np.zeros(npts)
    dx = xdist / npts
    rr = np.linspace(0, 1, npts)
    for (xr, mw) in zip(xrs, mws):
        l = moment2RuptureLength(mw) # [km]
        lr = l / xdist
        left = max(0, xr - lr/2)
        right = min(1, xr + lr/2)
        ileft = np.argmax(rr >= left)
        iright = np.argmax(rr > right)
        # arr[ileft: iright] += mag2moment(mw) / l / 1e3
        m = mag2moment(mw)
        a = lr / 2
        b = m * 2 / np.pi / a # ellipsoid, assume a = 0.5, s = pi * a * b
        for i in range(ileft, iright):
            arr[i] += np.sqrt((1 - (rr[i] - left - a) ** 2 / a ** 2) * b ** 2) / xdist / 1e3
    return arr, rr

def creepMask(arr, rr, left, right, thred):
    mask = np.zeros(len(arr))
    ileft = np.argmax(rr >= left)
    iright = np.argmax(rr > right)
    thred = np.max(arr) * thred

    for i in range(ileft, iright):
        if arr[i] <= thred:
            mask[i] = 1

    p1, p2 = [], []
    for i in range(1, len(mask)):
        if mask[i] == 0 and mask[i - 1] == 1:
            p2.append(i)
        if mask[i] == 1 and mask[i - 1] == 0:
            p1.append(i)

    return list(zip(p1, p2)), mask[ileft: iright]

def creepPercentage(arr, rr, left, right, thred):
    ileft = np.argmax(rr >= left)
    iright = np.argmax(rr > right)
    total = iright - ileft
    count = 0
    thred = np.max(arr) * thred
    for i in range(ileft, iright):
        if arr[i] <= thred:
            count += 1

    return count / total
