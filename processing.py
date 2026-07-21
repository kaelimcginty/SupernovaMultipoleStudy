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


def getRadius(mirroredData, size, dx):
    '''Calculate the radius of the object based on its density distribution and a specified pressure cutoff. 
    The radius is determined by finding the maximum distance from the center of the cube to any point 
    where the density exceeds a threshold derived from the maximum density.'''
    Density = mirroredData[0, :, :, :]
    maxDensity = np.max(Density)
    threshold = maxDensity * PRESSURECUTOFF
    indices = np.where(Density >= threshold)
    x_indices, y_indices, z_indices = indices
    x_center = size // 2
    y_center = size // 2
    z_center = size // 2
    distances = np.sqrt((x_indices - x_center) ** 2 + (y_indices - y_center) ** 2 + (z_indices - z_center) ** 2)
    radius = np.max(distances) * dx
    print(f"Calculated radius: {radius:.4f}")
    return radius


def dataRotator(rawData, theta, phi, size, fileNumber):
    '''Rotates the data based on the given angles theta and phi. The rotation is applied in two steps.'''

    if theta == 0 and phi == 0:
        return rawData

    print("rotating file number " + str(fileNumber))

    rotated = rawData

    if theta != 0:
        rotated = scipy.ndimage.rotate(
            rotated,
            theta,
            axes=(2, 3),      # rotate y-x plane
            reshape=False,
            order=1,
            mode='nearest'
        )

    if phi != 0:
        rotated = scipy.ndimage.rotate(
            rotated,
            phi,
            axes=(1, 2),      # adjust axes depending on intended plane
            reshape=False,
            order=1,
            mode='nearest'
        )

    return rotated


def CenterOfMass(plotData, size):
    '''Calculates the center of mass of the given 2D plot data. 
    The center of mass is computed as the weighted average of the coordinates.'''
    numX = 0
    numY = 0
    den = 0
    #print(size)
    for xCord in range(0, size):
        for yCord in range(0,size):
            numX = numX + xCord * plotData[xCord,yCord]
            numY = numY + yCord * plotData[xCord,yCord]
            den = den + plotData[xCord,yCord]
    CofMX = numX / den
    CofMY = numY / den
    print("Center of mass is at [" +str(CofMX) + ',' + str(CofMY) + ']')
    return(CofMX,CofMY)