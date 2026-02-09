#! /usr/bin/env python3

##############################
# Autonomous Systems 2021/2022 Mapping Project
#
# This is the main script that subscribes to laser scans and robot poses
# from a rosbag and publishes a topic /my_map that contains an 2-D occupancy map.
#   Group 13:
#       - 93079, Haohua Dong
#       - 96158, André Ferreira
#       - 96195, Duarte Cerdeira
#       - 96230, Inês Pinto
#
##############################
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData
import node_mapper
import numpy as np
import math
import time
from time import perf_counter

# Define map parameters
width = 1500
height = 1500
resolution = 0.05

# Probability for unknown cells
P_prior = 0.5


class mappingNode(object):
    """Class for subscribing to topics and publishing calculated map."""
    # Instance mapping algorithm
    mapper = node_mapper.Map(width, height, resolution, P_prior)

    def __init__(self):

        rospy.init_node('mapping_node', anonymous=True)

        # Flags for scan and pose messages
        self.Scan = False
        self.Pose = False

        # Robot pose parameters
        self.x = 0
        self.y = 0
        self.orientation = [0, 0, 0, 0]
        self.yaw = 0
        self.cov = 0

        # Laser scan parameters
        self.ranges = []
        self.angles = []
        self.z_max = None
        self.z_min = None

        # Map parameters
        self.width = width
        self.height = height
        self.resolution = resolution

        # Subscribe to scan and amcl_pose messages
        rospy.Subscriber('scan', LaserScan, self.callback_scan)
        rospy.Subscriber(
            'amcl_pose', PoseWithCovarianceStamped, self.callback_pose)

        # Map message initial information
        self.grid_map = OccupancyGrid()
        self.grid_map.header.frame_id = "map"
        self.grid_map.info.resolution = self.resolution
        self.grid_map.info.width = self.width
        self.grid_map.info.height = self.height
        self.grid_map.info.origin.position.x = -15
        self.grid_map.info.origin.position.y = -15
        self.grid_map.info.origin.position.z = 0
        self.grid_map.info.origin.orientation.x = 0
        self.grid_map.info.origin.orientation.y = 0
        self.grid_map.info.origin.orientation.z = 0
        self.grid_map.info.origin.orientation.w = 0
        self.grid_map.data = []

        # Create topic /my_map with the calculated map
        self.map_publisher = rospy.Publisher(
            '/my_map', OccupancyGrid, queue_size=1)

        # Create topic /my_map_metadata
        self.map_data_publisher = rospy.Publisher(
            '/my_map_metadata', MapMetaData, queue_size=1)

        # Publish initial map
        rospy.loginfo("Publishing initial map !")
        self.map_publisher.publish(self.grid_map)
        self.map_data_publisher.publish(self.grid_map.info)

        # OGM Algorithm
        self.occupancy_map = self.mapper.calculate_map(self.ranges,
                                                       self.angles,
                                                       self.x,
                                                       self.y,
                                                       self.yaw,
                                                       self.z_max,
                                                       self.z_min)
        # save the updated map
        self.probability_map = []
        # threshold to decide occupancy values
        self.threshold = 0.5

    def callback_pose(self, pose_msg):
        """Log listened pose data."""

        print('x = ' + str(self.x) + ', ' +
              'y = ' + str(self.y) + ', ' +
              'yaw = ' + str(self.yaw))

        self.x = pose_msg.pose.pose.position.x
        self.y = pose_msg.pose.pose.position.y
        self.orientation = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y,
                            pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        self.yaw = 2 * math.atan2(self.orientation[2],
                                  self.orientation[3])
        self.cov = pose_msg.pose.covariance
        self.pose_time = pose_msg.header.stamp.secs
        self.Pose = True

    def callback_scan(self, scan_msg):
        """Log listened laser data."""

        self.ranges = scan_msg.ranges
        # transform NaN values to 0
        self.ranges = np.where(np.isnan(self.ranges), 0, self.ranges)
        self.angles = np.linspace(scan_msg.angle_min,
                                  scan_msg.angle_max,
                                  len(self.ranges))
        self.z_max = scan_msg.range_max
        self.z_min = scan_msg.range_min
        self.scan_time = scan_msg.header.stamp.secs
        self.Scan = True

    def run_mapping(self):
        """ Calculate and convert the map data to a list and its respective occupancy probabilities 
        and publish the topic to ROS.
        The map data is a list and the Occupancy probabilities have values of -1, 0 or 100.
        """
        # OGM Algorithm
        self.occupancy_map = self.mapper.calculate_map(self.ranges,
                                                       self.angles,
                                                       self.x,
                                                       self.y,
                                                       self.yaw,
                                                       self.z_max,
                                                       self.z_min)
        # restore probability from log-odds
        self.probability_map = 100 * \
            (1 - np.divide(1, 1+np.exp(self.occupancy_map)))

        # define cell thresholds and apply occupancy probabilities
        self.probability_map[self.probability_map == 50] = -1

        # convert map to a list of occupancy values and publishes to ROS
        temp = np.reshape(self.probability_map, (1, self.width*self.height))
        self.grid_map.data = temp.tolist()[0]
        self.grid_map.data = np.int8(self.grid_map.data)
        rospy.loginfo("Publishing updated map ! ")
        self.map_publisher.publish(self.grid_map)
        self.map_data_publisher.publish(self.grid_map.info)

    def check_timestamps(self):

        difference = abs(self.pose_time - self.scan_time)
        print(difference)
        if difference <= 1:
            return True
        else:
            return False

    def master(self):
        # ROS node rate to get messages
        rate = rospy.Rate(10)  # 10 Hz
        initial_time = time.time()
        while not rospy.is_shutdown():
            if (not(self.Scan) or not(self.Pose)):
                continue
            # check if messages are synchronized
            if (self.check_timestamps()):
                self.run_mapping()
            self.Pose = False
            self.Scan = False
            rate.sleep()
        sim_time = time.time()-initial_time
        # computational time of the mapping algorithm
        self.mapper.return_times()
        # computational time for whole algorithm
        print('\nTotal simulation time outside: %.3f[s]' % sim_time)


def main():
    my_node = mappingNode()
    my_node.master()
    print('finished mapping !')


if __name__ == '__main__':
    main()
