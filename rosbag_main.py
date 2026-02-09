#! /usr/bin/env python3

##############################
# Autonomous Systems 2021/2022 Mapping Project
#
# This is the main script that reads a specific rosbag and creates an 2-D occupancy map.
#   Group 13:
#       - 93079, Haohua Dong
#       - 96158, André Ferreira
#       - 96195, Duarte Cerdeira
#       - 96230, Inês Pinto
#
##############################

import time
import numpy as np
import math
# to read bags
import rosbag
# mapping algorithm for rosbags
import rosbag_mapping
# for auxiliary functions
import utils
# ROS python api library
import rospy
import matplotlib.pyplot as plt

# desired limits and resolution of the map
P_prior = 0.5
xlim = [-15, 15]
ylim = [-15, 15]
resolution = 0.02


class Rosbag_handler():
    def __init__(self):

        # Robot pose parameters
        self.position_x = 0
        self.position_y = 0
        self.initial_x = 0
        self.initial_y = 0
        self.orientation = [0, 0, 0, 0]
        self.cov = 0
        self.pose_time = None

        # LRF parameters
        self.angle_min = None
        self.angle_max = None
        self.angle_increment = None
        self.time_increment = None
        self.range_min = None
        self.range_max = None
        self.ranges = 0
        self.scan_time = None
        self.angles = None

    def poseCallback(self, msg_pose):

        self.pose_time = msg_pose.header.stamp.secs
        self.position_x = msg_pose.pose.pose.position.x
        self.position_y = msg_pose.pose.pose.position.y
        self.orientation = [msg_pose.pose.pose.orientation.x, msg_pose.pose.pose.orientation.y,
                            msg_pose.pose.pose.orientation.z, msg_pose.pose.pose.orientation.w]
        self.cov = msg_pose.pose.covariance

    def scanCallback(self, msg_scan):

        self.scan_time = msg_scan.header.stamp.secs
        self.angle_min = msg_scan.angle_min
        self.angle_max = msg_scan.angle_max
        self.angle_increment = msg_scan.angle_increment
        self.time_increment = msg_scan.time_increment
        self.range_min = msg_scan.range_min
        self.range_max = msg_scan.range_max
        self.ranges = msg_scan.ranges
        self.ranges = np.where(np.isnan(self.ranges), 0, self.ranges)
        self.angles = np.linspace(
            self.angle_min, self.angle_max, len(self.ranges))

    def run_algorithm(self, i):
        # save initial positions of robot as (0, 0)
        yaw = 2 * math.atan2(self.orientation[2],
                             self.orientation[3])
        # returns data for mapping algorithm
        return self.ranges, self.angles, self.position_x-self.initial_x, self.position_y-self.initial_y, yaw, self.range_max, self.range_min

    def check_timestamps(self):

        difference = abs(self.pose_time - self.scan_time)
        # print(difference)
        if difference <= 1:
            return True
        else:
            return False


def read_bag(pose, bagname):
    # initialize 2-D occupancy grid map
    Map = rosbag_mapping.Map(xlim, ylim, resolution, P_prior)
    bag = rosbag.Bag(bagname)
    handler = Rosbag_handler()
    i = 0
    Scan = False
    Pose = False
    initial_time = time.time()
    # reads rosbag
    for topic, msg, t in bag.read_messages():
        if topic == pose:
            handler.poseCallback(msg)
            Pose = True
        elif topic == '/scan':
            handler.scanCallback(msg)
            Scan = True
        i += 1
        # only run algorithm when we get both pose and scan messages
        if (not(Scan) or not(Pose)):
            continue
        # check if time stamps of messages are synchronized
        if (handler.check_timestamps()):
            z, angles, x, y, yaw, z_max, z_min = handler.run_algorithm(i)
            occupancy_map = Map.calculate_map(
                z, angles, x, y, yaw, z_max, z_min)
        Pose = False
        Scan = False
    bag.close()
    sim_time = time.time()-initial_time
    # computational time of the mapping algorithm
    Map.return_times()
    # computational time for reading rosbags
    print('\nTotal simulation time outside: %.3f[s]' % sim_time)
    return occupancy_map


def main():
    amcl_pose = '/amcl_pose'
    bagname = 'bag1.bag'
    map1 = read_bag(amcl_pose, bagname)
    bagname = 'bag2.bag'
    map2 = read_bag(amcl_pose, bagname)
    bag = rosbag.Bag('bag3.bag')
    for topic, msg, t in (bag.read_messages()):
        if topic == '/map':
            gmapping_map = msg.data
            break
    gmapping_matrix = np.reshape(gmapping_map, (992, 1600))
    gmapping_matrix = gmapping_matrix.astype(float)
    gmapping_matrix[gmapping_matrix == 100.0] = 1
    gmapping_matrix[gmapping_matrix == -1.0] = 0.5


if __name__ == '__main__':
    main()
