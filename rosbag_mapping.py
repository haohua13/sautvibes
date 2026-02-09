#! /usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from bresenham import bresenham
import time

# Probabilities chosen by the user to define the occupancy values
P_occupied = 0.6
P_free = 0.3
P_prior = 0.5
width = 1600
height = 1600


def log_odds(p):
    '''Calculate the log-odd probability'''
    return np.log(p/(1-p))


def restore_p(l):
    '''Restore the probability from log-odds '''
    exp = 1 + np.exp(l)
    return 1-(1/exp)


class Map():

    def __init__(self, xlim, ylim, resolution, p):

        # limits of the map

        self.xlim = xlim
        self.ylim = ylim
        self.resolution = resolution  # grid resolution in [m]
        self.width = width
        self.height = height

        # cell values from (x, y) 2-D map
        x = np.arange(xlim[0], xlim[1] + resolution, step=resolution)
        y = np.arange(ylim[0], ylim[1] + resolution, step=resolution)

        self.xsize = len(x)  # number of cells in x direction
        self.ysize = len(y)  # number of cells in y direction
        self.map_size = self.xsize*self.ysize  # area of the map

        self.alpha = 0.06  # thickness of the obstacle
        self.z_max = None   # max reading distance from the laser scan
        self.z_min = None  # min reading distance from the laser scan

        # initial log-odd probability map matrix
        self.log_map = np.full((self.width, self.height), log_odds(p))
        # self.log_map = np.full((self.xsize, self.ysize), log_odds(p))

        # log probabilities to update the map
        self.l_occupied = log_odds(P_occupied)
        self.l_free = log_odds(P_free)

        # for computational time purposes
        self.step_time = 0
        self.sim_time = 0
        self.step = 0
        self.start = 0
        self.end = 0

    def map_coordinates(self, x_continuous, y_continuous):
        '''
        Convert (x,y) continuous coordinates to discrete coordinates
        '''
        x = int((x_continuous + self.xlim[1]) / self.resolution)
        y = int((y_continuous + self.ylim[1]) / self.resolution)

        return (x, y)

    def calculate_map(self, z, angles, x, y, yaw, z_max, z_min):
        """
        Compute the occupancy-grid map for a given sensor/robot data
        """
        self.start = time.perf_counter()
        self.z_max = z_max
        self.z_min = z_min

        # initial (x, y) for Bresenham's algorithm (robot position)
        x1, y1 = self.map_coordinates(x, y)

        # run algorithm for all range and angle measurements
        for angle, dist in zip(angles, z):

            # ignore range values that are NaN and only update map when range is inside range limits
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
                    self.log_map[x_bresenham, y_bresenham] += self.l_free

                # obstacle cells hit from laser are occupied
                for(x_bresemham, y_bresemham) in bresenham(Map, x2, y2, x3, y3):
                    self.log_map[x_bresemham,
                                 y_bresemham] += self.l_occupied

        self.end = time.perf_counter()
        self.step_time = self.end-self.start
        self.sim_time += self.step_time
        self.start = self.step_time
        self.step += 1
        # prints the time taken in each algorithm run
        print('\nStep %d : %d [ms]' % (self.step, self.step_time*1000))
        return self.log_map

    def return_map(self):
        '''Return calculated map'''
        return self.log_map

    def return_times(self):
        '''Print the average step time of the algorithm and total simulation time'''

        print('\nAverage step time: %d [ms]' %
              ((self.sim_time*1000)/self.step))
        print('\nTotal Simulation time:%.3f[s]' % self.sim_time)


if __name__ == '__main__':
    main()
