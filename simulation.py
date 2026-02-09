#! /usr/bin/env python3
##############################
# Autonomous Systems 2021/2022 Mapping Project
#
# This is the script that simulates the 2-D Occupancy Grid Mapping algorithm.
#   Group 13:
#       - 93079, Haohua Dong
#       - 96158, André Ferreira
#       - 96195, Duarte Cerdeira
#       - 96230, Inês Pinto
#
##############################
import matplotlib.pyplot as plt
import numpy as np
from bresenham import bresenham
import utils

# Probabilities chosen by the user to define the occupancy values
P_occupied = 0.6
P_free = 0.3
P_prior = 0.5


def log_odds(p):
    '''Calculate the log-odd probability'''
    return np.log(p/(1-p))


def restore_p(l):
    '''Restore the probability from log-odds '''
    return 1-1 / (1+np.exp(l))


class Map():

    def __init__(self, xlim, ylim, resolution, p):

        # limits of the map
        self.xlim = xlim
        self.ylim = ylim
        self.resolution = resolution  # grid resolution in [m]

        # cell values from (x, y) 2-D map
        x = np.arange(xlim[0], xlim[1] + resolution, step=resolution)
        y = np.arange(ylim[0], ylim[1] + resolution, step=resolution)

        self.xsize = len(x)  # number of cells in x direction
        self.ysize = len(y)  # number of cells in y direction
        self.map_size = self.xsize*self.ysize  # area of the map

        self.alpha = 0.1  # thickness of the obstacle
        self.z_max = None   # max reading distance from the laser scan

        # initial log-odd probability map matrix
        self.log_map = np.full((self.xsize, self.ysize), log_odds(p))

        # log probabilities to update the map
        self.l_occupied = log_odds(P_occupied)
        self.l_free = log_odds(P_free)

    def map_coordinates(self, x_continuous, y_continuous):
        '''
        Convert (x,y) continuous coordinates to discrete coordinates
        '''
        x = int((x_continuous - self.xlim[0]) / self.resolution)
        y = int((y_continuous - self.ylim[0]) / self.resolution)

        return (x, y)

    def calculate_map(self, z, angles, x, y, yaw, z_max, z_min):
        """
        Compute the occupancy-grid map for a given sensor/robot data
        """
        self.z_max = z_max
        self.z_min = z_min

        # initial (x, y) for Bresenham's algorithm (robot position)
        x1, y1 = self.map_coordinates(x, y)

        # run algorithm for all range and angle measurements
        for angle, dist in zip(angles, z):

            # ignore range values that are NaN and only update map when range is inside the range limits
            if (not np.isnan(dist)) and dist < self.z_max and dist > self.z_min:

                # angle of the laser + orientation of the robot
                theta = angle+yaw
                if theta > np.pi:
                    theta -= 2*np.pi
                elif theta < -np.pi:
                    theta += 2*np.pi

                # obstacle position
                x2 = x + dist * np.cos(theta)
                y2 = y + dist * np.sin(theta)

                # obstacle position with assumed object thickness alpha
                x3 = x + (dist + self.alpha) * np.cos(theta)
                y3 = y + (dist + self.alpha) * np.sin(theta)

                # discretize coordinates for Bresenham's algorithm (obstacle position)
                x2, y2 = self.map_coordinates(x2, y2)

                # discretize coordinates for Bresenham's algorithm (obstacle position with thickness alpha)
                x3, y3 = self.map_coordinates(x3, y3)

                # all cells between the robot position to the laser hit cell are free
                for (x_bresenham, y_bresenham) in bresenham(Map, x1, y1, x2, y2):
                    self.log_map[y_bresenham, x_bresenham] += self.l_free

                # obstacle cells hit from laser are occupied
                    for(x_bresemham, y_bresemham) in bresenham(Map, x2, y2, x3, y3):
                        self.log_map[y_bresemham,
                                     x_bresemham] += self.l_occupied

        return self.log_map

    def return_map(self):
        return self.log_map


if __name__ == '__main__':
    # Test micro-simulator
    # Desired limits and resolution of the map
    xlim = [-10, 10]
    ylim = [-10, 10]
    resolution = 0.1
    z_max = 10
    z_min = 0.1
    # Laser data

    z = [3/np.sin(np.pi*(6/24)), 3/np.sin(np.pi*(7/24)), 3/np.sin(np.pi*(8/24)), 3/np.sin(np.pi*(9/24)), 3/np.sin(np.pi*(10/24)), 3/np.sin(np.pi*(11/24)), 3,
         3/np.sin(np.pi*(13/24)), 3/np.sin(np.pi*(14/24)), 3/np.sin(np.pi*(15/24)), 3/np.sin(np.pi*(16/24)), 3/np.sin(np.pi*(17/24)), 3/np.sin(np.pi*(18/24))]
    angles = [np.pi*(6/24), np.pi*(7/24), np.pi*(8/24), np.pi*(9/24), np.pi*(10/24), np.pi*(11/24), np.pi *
              (12/24), np.pi*(13/24), np.pi*(14/24), np.pi*(15/24), np.pi*(16/24), np.pi*(17/24), np.pi*(18/24)]
    # Robot Pose Data
    x = 0
    y = 0
    yaw = 0

    # initialize 2-D occupancSy grid map
    Map = Map(xlim, ylim, resolution, P_prior)

    # final occupancy grid map
    occupancy_map = Map.calculate_map(z, angles, x, y, yaw, z_max, z_min)
    x = -5
    y = 5
    z = [2/np.sin(np.pi*(6/24)), 2/np.sin(np.pi*(5/24)), 2/np.sin(np.pi *
                                                                  (4/24)), 2/np.sin(np.pi*(3/24)), 5, 5, 5, 5, 5, 5, 5, 5, 5]
    angles = [np.pi*(-6/24), np.pi*(-5/24), np.pi*(-4/24),
              np.pi*(-3/24), np.pi*(-2/24), np.pi*(-1/24), 0, np.pi *
              (1/24), np.pi*(2/24), np.pi*(3/24),
              np.pi*(4/24), np.pi*(5/24), np.pi*(6/24)]

    occupancy_map = Map.calculate_map(z, angles, x, y, yaw, z_max, z_min)

    # plot results in plots.py
    occupancy_map = 1 - np.divide(1, 1+np.exp(occupancy_map))
    plt.figure(1)
    utils.plot_map(occupancy_map, resolution, xlim, ylim)
    plt.savefig('simulation_wall.pdf', dpi=600)
    plt.show()
