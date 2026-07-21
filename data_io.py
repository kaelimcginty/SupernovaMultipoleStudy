from config import PRESSURECUTOFF, PARAMETERS
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import os
import scipy
import scipy.ndimage
from matplotlib.colors import LogNorm


def dataRetriever(fileExtension, number): # Works upto file number 999
    '''Retrieves the data from the specified file, applies pressure parsing, and returns the mirrored data along with size, time, and dx.'''
    zeroNumber = 4 - len(str(number))
    fileName = fileExtension + '0' * zeroNumber + str(number) + '.h5'
    f = h5py.File(fileName, 'r')

    rawData = f['Data']['Cells'][PARAMETERS]
    rawData = np.array(rawData)
    size = np.shape(rawData)[2]
    rawData = pressureParser(rawData, size)

    mirroredData=rawData


    time = f['Grid']['T'][0]
    dx = f['Grid']['dxs'][0]

    return(mirroredData, size, time, dx)


def pressureParser(rawData, size):
    '''Applies a pressure cutoff to the raw data. If the pressure at a given point is below the specified PRESSURECUTOFF, 
    the corresponding density is set to a very small value (0.0000001).'''
    PressureIndex = PARAMETERS.index(1)
    DensityIndex = PARAMETERS.index(0)
    for xCord in range(0,size):
        for yCord in range(0,size):
            for zCord in range(0,size):
                if rawData[PressureIndex, xCord, yCord, zCord] < PRESSURECUTOFF:
                    rawData[DensityIndex, xCord, yCord, zCord] = 0.0000001
    return(rawData)


def dataMirrorer(rawData):
    '''Mirrores the data along the x=0, y=0, and z=0 axes to create a full 3D representation of the simulation.
    Only needed when the simulation data is in one octant. The function returns the mirrored data.'''
    # Step 1: Mirror along the x=0 axis (2nd dimension)
    print("Old data shape:", rawData.shape)  # Should print (6, 512, 512, 512)
    mirrored_x = np.concatenate([rawData[:, ::-1, :, :], rawData], axis=1)  # Mirror along x=0

    # Step 2: Mirror along the y=0 axis (3rd dimension)
    mirrored_xy = np.concatenate([mirrored_x[:, :, ::-1, :], mirrored_x], axis=2)  # Mirror along y=0

    # Step 3: Mirror along the z=0 axis (4th dimension)
    mirrored_xyz = np.concatenate([mirrored_xy[:, :, :, ::-1], mirrored_xy], axis=3)  # Mirror along z=0

    # Final mirrored data shape
    print("New data shape:", mirrored_xyz.shape)  # Should print (6, 512, 512, 512)
    return(mirrored_xyz)


def lambdaFromFolder(folder):
    '''Accesses the value of lambbda from the folder name. Leading zeros in the folder name shift the decimal point.'''
    #Convert folder name like '100', '10', '1', '01', '001' into its Lambda value.
    #Leading zeros shift the decimal point: '01' -> 0.1, '001' -> 0.01, etc.
    leadingZeros = len(folder) - len(folder.lstrip('0'))
    remaining = folder.lstrip('0')
    if remaining == '':  # edge case: folder is all zeros, e.g. "0"
        remaining = '0'
    value = int(remaining) / (10 ** leadingZeros)
    return value