

from config import STRENGTH, PRESSURECUTOFF, PARAMETERS
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import os
import scipy
import scipy.ndimage
from matplotlib.colors import LogNorm

def magneticField(mirroredData, size):
    '''Applies a magnetic field to the simulation data based on the density and velocity components.
    The magnetic field is calculated using a specified strength and the spatial coordinates of the data points.
    The function returns the modified data with the magnetic field applied.'''
    #Make xyz position maps
    Use = mirroredData
    print("Data shape:", np.shape(Use))

    # Extract Density (we keep density from the original data)
    Density = mirroredData[0, :, :, :]
    Pressure = mirroredData[1, :, :, :]

    # Get the dimensions of the cube (assumed cubic)
    nx, ny, nz = Density.shape

    # Create coordinate arrays such that the center of the cube is at (0, 0, 0).
    # For example, for nx=512, this gives x values from -256 to 255.
    x = np.arange(nx) - nx // 2 + 0.5
    y = np.arange(ny) - ny // 2 + 0.5
    z = np.arange(nz) - nz // 2 + 0.5

    # Create 3D grids for X, Y, and Z. Use 'ij' indexing so that the shapes match Density.
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    mirroredData[2,:,:,:] = X
    mirroredData[3,:,:,:] = Y
    mirroredData[4,:,:,:] = Z

    #rotatedData = mirroredData   REMOVE

    VelocityX = Use[2,:,:,:]
    VelocityY = Use[3,:,:,:]
    VelocityZ = Use[4,:,:,:]

    # Square the density and velocity components
    DensitySquared = np.square(Density)
    VelocityXSquared = np.square(VelocityX)
    VelocityYSquared = np.square(VelocityY)
    VelocityZSquared = np.square(VelocityZ)

    # Compute the magnetic data based on the angle and velocity components.
    C = STRENGTH + (1-STRENGTH) * (1 + (Z**2) / (Z**2 + X**2 + Y**2 + 0.0001))

    MagneticData = np.multiply(C, DensitySquared)


    return(MagneticData)
