from config import INCREMENT, PRESSURECUTOFF
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import os
import scipy
import scipy.ndimage
from matplotlib.colors import LogNorm

def multipoleCalculator(plotData, Radius, CenterX, CenterY, INCREMENT, m, size):
    '''Calculates the multipole moment of order m for the given 2D plot data.
    Equation taken from Laura Lopez's paper on multipole moments of supernova remnants.'''
    a = 0.0
    b = 0.0

    # Determine the bounding box of the circle, clamped to the image dimensions.
    startX = max(0, int(math.floor(CenterX - Radius)))
    endX   = min(plotData.shape[0], int(math.ceil(CenterX + Radius)) + 1)
    startY = max(0, int(math.floor(CenterY - Radius)))
    endY   = min(plotData.shape[1], int(math.ceil(CenterY + Radius)) + 1)


    startX = 0
    startY = 0
    endX = size
    endY = size
    for x in range(startX, endX, INCREMENT):
        for y in range(startX, endX, INCREMENT): #changed to startX and endX to make it integrate over the whole sphere
            # Compute the coordinates relative to the center.
            xNew = x - CenterX
            yNew = y - CenterY

            # Check if the (x,y) lies within the circle.
            if (xNew**2 + yNew**2) <= (10* Radius**2): # Artifact of old code, it currently does the whole square as the pressure parser fixes the density to be very small outside the radius, so it doesn't matter
                R = math.hypot(xNew, yNew)
                phi = math.atan2(yNew, xNew)
                
                xrayValue = plotData[x, y]
                area = INCREMENT**2  # the integration element
                
                # Accumulate contributions.
                a += xrayValue * (R**m) * math.cos(m * phi) * area
                b += xrayValue * (R**m) * math.sin(m * phi) * area

    if m == 0:
        P = (a * math.log(Radius))**2
    else:
        P = (a*a + b*b) / (2 * m * m * (Radius**(2*m)))
        
    #print(f"a is {a:.4e} and b is {b:.4e} when m = {m}")
    print(f"P{m} = {P:.4e}")
    return P
    