#! /usr/bin/env python3
# This script contains auxiliary functions for the occupancy-grid mapping algorithm
import matplotlib.pyplot as plt
import cv2 as cv2
from bagpy import bagreader
import pandas as pd
import numpy as np
import math


def maximum_likelihood(_map):
    """
    Calculate the maximum likelihood map by clipping the occupancy grid map at 0.5 threshold.
    """
    _map[_map < 0.5] = 0
    _map[_map > 0.5] = 1
    return _map


def restore_p(matrix):
    """ Restore probability matrix from log-odds."""
    return (1 - np.divide(1, 1+np.exp(matrix)))


def compare_maps(map1, map2):
    """ Function to compare two maps. Reference map is map1.
    Evaluation metric is the sum of the value of the
    pixel of the difference map divided by the number of free cells and occupied cells.
     """
    difference_map = abs(map1-map2)

    total_difference = sum(sum(difference_map))
    free = (map1 == 0).sum()
    occupied = (map1 == 1).sum()
    total = free+occupied
    error = (total_difference/total) * 100


def occupancy(map1):
    '''To test different thresholds for occupancy map map'''
    map1[map1 > 0.85] = 1
    map1[map1 < 0.1] = 0
    map1[(map1 >= 0.1) & (map1 <= 0.85)] = 0.5
    return map1


def plot_map(occupancy_map, resolution, xlim, ylim):
    """Plot a simple figure to represent the occupancy map """

    plt.imshow(occupancy_map, 'Greys')
    # To represent the x-y limits of the map
    plt.xlim([0, len(occupancy_map)-1])
    plt.ylim([0, len(occupancy_map[0])-1])
    locs, labels = plt.xticks()
    labels = [float(item)*resolution-xlim[0] for item in locs]
    plt.xticks(locs, labels)
    locs, labels = plt.yticks()
    labels = [float(item)*resolution-ylim[0] for item in locs]
    plt.yticks(locs, labels)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.colorbar()


def plot_trajectory(bag_file):
    """ Plot the trajectory (x, y) of the robot """

    b = bagreader(bag_file)
    pose = b.message_by_topic('/amcl_pose')
    posedf = pd.read_csv(pose)

    posedf.plot(x='pose.pose.position.x', y='pose.pose.position.y',
                xlabel='x [m]', ylabel='y [m]', label='Trajectory of the robot')
    plt.title('Robot position in 2D',
              fontsize=14, fontweight='bold')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot_trajectory('bag1.bag')
