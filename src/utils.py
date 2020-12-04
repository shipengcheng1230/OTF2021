import os
from typing import *

import pandas as pd
import numpy as np
from obspy.clients.fdsn import Client

baseDir = os.path.join(os.path.dirname(__file__), "..")
faultsDir = os.path.join(baseDir, "faults")
if not os.path.isdir(faultsDir):
    os.mkdir(faultsDir)
dfFaults = pd.read_excel(os.path.join(baseDir, "faults.xlsx"))
dfStations = pd.read_csv(os.path.join(baseDir, "stations.csv"))
try:
    client = Client("IRIS", timeout=30)
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
    return 10 ** (x * 3/2 + 16.1)

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
