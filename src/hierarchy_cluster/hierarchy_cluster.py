#!/usr/bin/env python3

""" MIT License

Copyright (c) 2020 Group of Electronic Technology and Communications. University of A Coruna.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. """


import string
from turtle import position
from attr import has
import numpy as np
from typing import Dict, List
import time
from regex import T
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from geometry_msgs.msg import PointStamped, Point, PoseWithCovarianceStamped, PoseWithCovariance
from std_msgs.msg import Header
import rospy
import tf2_ros
import tf2_geometry_msgs
from gtec_msgs.msg import RadarFusedPointStamped
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial import distance

class TargetPoint():
    def __init__(self, x: float, y:float, z:float) -> None:
        self.x = x
        self.y = y
        self.z = z

class HierarchyClusterOptions():
    def __init__(self, linkageMethod: string, clusterCriterion: string, clusterThreshold: float) -> None:
        self.linkageMethod = linkageMethod
        self.clusterCriterion = clusterCriterion
        self.clusterThreshold = clusterThreshold

class HierarchyClusterFuse():
    def __init__(self, options: HierarchyClusterOptions, windowTimeInMs:float, loop_time:float) -> None:

        self.options = options
        self.windowTimeInMs = windowTimeInMs
        self.loop_time = loop_time

        self.window_positions_stack = []

        self.last_window_time_ms = -1
        self.last_update_time_ms = -1

    def addTargetsCloud(self, cloud: PointCloud2)-> None:
        for p in pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
            self.addPosition(TargetPoint(p[0], p[1], p[2]))
    
    def addPosition(self, pos: TargetPoint)-> None:
        #print(f'New position: {pos.x},{pos.y}')
        self.window_positions_stack.append(pos)

    
    def getClusterCentroids(self, positions, cluster_index) -> np.array:
        centroids = []
        max_distances = []

        for i in range(cluster_index.min(), cluster_index.max()+1):
            centroid_of_cluster = positions[cluster_index == i].mean(0)
            centroids.append(centroid_of_cluster)
            max_distance = np.array([np.max(distance.cdist(positions[cluster_index == i], [centroid_of_cluster]))])
            max_distances.append(max_distance)

        max_distances = np.vstack(max_distances)
        centroid_means = np.vstack(centroids)
        # print(f'max_distances {max_distances} centroid_means {centroid_means}')
        return (centroid_means, max_distances)


    def processWindow(self, elapsed_time) -> []:
        #num_points = len(self.window_positions_stack)
        array_coords = np.empty((0,2))
        #array_est_err = np.empty((0,1))
        
        for point in self.window_positions_stack:
            array_coords = np.vstack([array_coords, [point.x, point.y]])
            #array_est_err = np.vstack([array_est_err, [point.error_est]])
            
        
        self.window_positions_stack.clear()

        if (len(array_coords)<1):
            return (np.array([]), np.array([[0]])) 
        
        
        if (len(array_coords)>1):
            linked = linkage(array_coords, method=self.options.linkageMethod)
            cluster_indexes= fcluster(linked, criterion=self.options.clusterCriterion,t=self.options.clusterThreshold)
            (cluster_centroids, max_distances) = self.getClusterCentroids(array_coords, cluster_indexes)
        else:
            #Only one point
            cluster_centroids = np.array([array_coords[0,:]])
            max_distances = np.array([[0.1]])

        return (cluster_centroids, max_distances)
            

    def loop(self) -> []:

        cluster_centroids = np.array([]) 
        max_distances = np.array([]) 
        current_time_ms = float(rospy.get_rostime().to_nsec())/1000000.0
        if (current_time_ms>0):
            if (self.last_window_time_ms<0):
                self.last_window_time_ms = current_time_ms
            
            elapsed_in_ms = current_time_ms - self.last_window_time_ms

            if elapsed_in_ms>=self.windowTimeInMs:
                (cluster_centroids, max_distances) = self.processWindow(elapsed_in_ms)
                self.last_window_time_ms = current_time_ms
            # else:
            #      self.sendFakeMeasurement(10)


        return (cluster_centroids, max_distances)


if __name__ == "__main__":

    rospy.init_node('HierarchyCluster', anonymous=True)
    rate = rospy.Rate(20)  # hz

    cloud_topic = rospy.get_param('~cloud_topic')
    publish_centroids_topic = rospy.get_param('~publish_centroids_topic')
    publish_own_centroids_topic = rospy.get_param('~publish_own_centroids_topic')



    clusterThreshold = float(rospy.get_param('~cluster_threshold',1))
    windowTimeInMs = int(rospy.get_param('~window_time_in_ms',500))

    
    pub_centroids= rospy.Publisher(publish_centroids_topic, PointCloud2, queue_size=100)
    pub_own_centroids= rospy.Publisher(publish_own_centroids_topic, PointCloud2, queue_size=100)


    hc_options = HierarchyClusterOptions(linkageMethod='centroid', clusterCriterion='distance', clusterThreshold= clusterThreshold)
    hc = HierarchyClusterFuse(options=hc_options,
    windowTimeInMs=windowTimeInMs, 
    loop_time=windowTimeInMs)

    rospy.Subscriber(str(cloud_topic), PointCloud2, hc.addTargetsCloud)  



    print("=========== GTEC mmWave Hierarchy Cluster Node ============")
    
    last_targets_count = -1
    while not rospy.is_shutdown():
        (current_targets, max_distances) = hc.loop()
        if (len(current_targets)>0):
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'map'

            centroids = []
            
            if len(current_targets)!=last_targets_count:
                last_targets_count = len(current_targets)
                print(f'Num targets {last_targets_count}')
            
            
            for index in range(len(current_targets)):
                pos = current_targets[index]
                max_distance = max_distances[index]
                #print(f'Publish {[pos[0], pos[1], 0, max_distance[0]]}')
                centroids.append([pos[0], pos[1], 0, max_distance[0]])


            centroid_fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('error_est', 12, PointField.FLOAT32, 1)
            ]

            centroids_msg = pc2.create_cloud(
                header, centroid_fields, centroids)    
            pub_centroids.publish(centroids_msg)
            pub_own_centroids.publish(centroids_msg)





        rate.sleep()