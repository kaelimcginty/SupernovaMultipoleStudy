import numpy as np
import h5py
import scipy.ndimage
from config import PARAMETERS, PRESSURECUTOFF

class Checkpoint:
    def __init__(self, fileExtension, number):
        self.fileExtension = fileExtension
        self.number = number
        self._load()

    def _load(self):
        '''Loads raw data from the h5 file and applies the pressure cutoff.'''
        zeroNumber = 4 - len(str(self.number))
        fileName = self.fileExtension + '0' * zeroNumber + str(self.number) + '.h5'
        f = h5py.File(fileName, 'r')

        rawData = f['Data']['Cells'][PARAMETERS]
        self.data = np.array(rawData)
        self.size = np.shape(self.data)[2]
        self.apply_pressure_cutoff()

        self.time = f['Grid']['T'][0]
        self.dx = f['Grid']['dxs'][0]

    def apply_pressure_cutoff(self):
        
        pressureIndex = PARAMETERS.index(1)
        densityIndex = PARAMETERS.index(0)
        mask = self.data[pressureIndex] < PRESSURECUTOFF
        self.data[densityIndex][mask] = 10**(-7)

    def get_radius(self):
        '''Calculate the radius of the object based on its density distribution and a specified pressure cutoff. 
        The radius is determined by finding the maximum distance from the center of the cube to any point 
        where the density exceeds a threshold derived from the maximum density.'''
        Density = self.data[0, :, :, :]
        maxDensity = np.max(Density)
        size = self.size
        threshold = maxDensity * PRESSURECUTOFF
        indices = np.where(Density >= threshold)
        x_indices, y_indices, z_indices = indices
        x_center = size // 2
        y_center = size // 2
        z_center = size // 2
        distances = np.sqrt((x_indices - x_center) ** 2 + (y_indices - y_center) ** 2 + (z_indices - z_center) ** 2)
        radius = np.max(distances) * self.dx
        print(f"Calculated radius: {radius:.4f}")
        self.radius = radius

    def rotate(self, theta, phi):
        '''Rotates the data based on the given angles theta and phi. The rotation is applied in two steps.'''

        if theta == 0 and phi == 0:
            self.rotatedData = self.data
            return

        print("rotating file number " + str(self.number))

        rotated = self.data

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
                axes=(1, 2),      # z-y axis
                reshape=False,
                order=1,
                mode='nearest'
            )
        
        self.rotatedData = rotated

    def center_of_mass(self, plotData):
        '''Calculates the center of mass of the given 2D plot data. 
        The center of mass is computed as the weighted average of the coordinates.'''
        numX = 0
        numY = 0
        den = 0
        for xCord in range(0, self.size):
            for yCord in range(0, self.size):
                numX = numX + xCord * plotData[xCord,yCord]
                numY = numY + yCord * plotData[xCord,yCord]
                den = den + plotData[xCord,yCord]
        CofMX = numX / den
        CofMY = numY / den
        print("Center of mass is at [" +str(CofMX) + ',' + str(CofMY) + ']')
        self.CofMX = CofMX
        self.CofMY = CofMY