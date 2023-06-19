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


import rospy
import time
import serial
import os
import tf2_ros
import csv
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class OpticsClusterSimple(object):

    def __init__(self,
                 publisher_centroids_cloud,
                 publisher_cluster_cloud,
                 window_time_in_ms,
                 min_samples_to_be_core_point, 
                 min_samples_in_cluster,
                 max_distance_to_consider_same_cluster
                 ):
        self.publisher_centroids_cloud = publisher_centroids_cloud
        self.publisher_cluster_cloud = publisher_cluster_cloud

        self.min_samples_in_cluster = min_samples_in_cluster
        self.min_samples_to_be_core_point = min_samples_to_be_core_point
        self.max_distance_to_consider_same_cluster = max_distance_to_consider_same_cluster
        self.window_time_in_ms = window_time_in_ms
        
        self.publishers_pose = []
        self.tracked_objects = []
        self.last_window_time_ms = -1
        self.last_update_time_ms = -1
        self.windowed_points_first = True


    def loop(self):
        current_time_ms = float(rospy.get_rostime().to_nsec())/1000000.0
        if (current_time_ms>0):
            if (self.last_window_time_ms<0):
                self.last_window_time_ms = current_time_ms
            
            elapsed_in_ms = current_time_ms - self.last_window_time_ms

            if elapsed_in_ms>=self.window_time_in_ms:
                self.processWindow(elapsed_in_ms)
                self.last_window_time_ms = current_time_ms
                self.windowed_points_first = True

    def addSingleCloud(self, cloud: PointCloud2)-> None:
        for p in pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
            if (self.windowed_points_first==True):
                self.windowed_points = np.array([[p[0], p[1], p[2]]])
                self.windowed_points_first = False
            else:
                self.windowed_points = np.vstack(
                    [self.windowed_points, [p[0], p[1], p[2]]])


    def processWindow(self, elapsed_time) -> None:
        if (self.windowed_points_first == False):
            print("Len points: " + str(len(self.windowed_points)))
            if (len(self.windowed_points) >= self.min_points_cluster):
                    this_window_points = self.windowed_points
                    model = OPTICS(min_samples=self.min_samples_to_be_core_point,
                                   min_cluster_size=self.min_samples_in_cluster, 
                                   max_eps=self.max_distance_to_consider_same_cluster)
                    pred = model.fit_predict(this_window_points)

                    space = np.arange(len(this_window_points))
                    reachability = model.reachability_[model.ordering_]
                    labels = model.labels_[model.ordering_]

                    fields_cloud = [
                        PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('r', 12, PointField.FLOAT32, 1),
                        PointField('g', 16, PointField.FLOAT32, 1),
                        PointField('b', 20, PointField.FLOAT32, 1)
                    ]

                    header_cloud = Header()
                    header_cloud.stamp = rospy.Time.now()
                    header_cloud.frame_id = "map"

                    points_cluster_cloud = []
                    points_centroid_cloud = []

                    clusters_ids = np.unique(labels)
                    colors = [[30, 200, 60], [85, 255, 127], [0, 85, 255], [
                        255, 255, 0], [128, 128, 0], [0, 128, 128]]
                    centroids = np.zeros([len(clusters_ids), 3])

                    print("Num. Clusters: ", len(clusters_ids))
                    print("Labels: ", str(labels))
                    print("self.windowed_points: ", str(this_window_points))

                    for i in range(len(clusters_ids)):
                        cluster_id = clusters_ids[i]
                        
                        if float(cluster_id) > -1:
                            print("Cluster ID: ", str(cluster_id))
                            points_of_cluster = this_window_points[labels == cluster_id]
                            centroid_of_cluster = np.mean(
                                points_of_cluster, axis=0)
                            centroids[i] = centroid_of_cluster
                            rgb = [0, 0, 0]
                            if (i < len(colors)):
                                rgb = colors[i]
                            pt = np.append(centroid_of_cluster, rgb)
                            points_centroid_cloud.append(pt)   
                    
                    print("Publish centroids")
                    centroids_cloud = pc2.create_cloud(
                        header_cloud, fields_cloud, points_centroid_cloud)
                    self.publisher_centroids_cloud.publish(centroids_cloud)


                    for i in range(len(this_window_points)):
                        rgb = [0, 0, 0]
                        index_label = np.where(clusters_ids == labels[i])
                        if (index_label[0][0] < len(colors)):
                            rgb = colors[index_label[0][0]]
                        pt = np.append(this_window_points[i], rgb)
                        points_cluster_cloud.append(pt)

                    print("Publish Cluster cloud")
                    cluster_cloud = pc2.create_cloud(
                        header_cloud, fields_cloud, points_cluster_cloud)
                    self.publisher_cluster_cloud.publish(cluster_cloud)


if __name__ == "__main__":

    rospy.init_node('OpticsClusterSimple', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    # Read parameters
    cloud_topic = rospy.get_param('~cloud_topic')
    publish_centroids_topic = rospy.get_param('~publish_centroids_topic')
    publish_cluster_topic = rospy.get_param('~publish_cluster_cloud_topic')


    min_samples_in_cluster = int(rospy.get_param('~min_samples_in_cluster'))
    min_samples_to_be_core_point = int(rospy.get_param('~min_samples_in_cluster'))
    max_distance_to_consider_same_cluster = float(rospy.get_param('~min_samples_in_cluster'))
    window_time_in_ms = int(rospy.get_param('~min_samples_in_cluster'))


    pub_centroids = rospy.Publisher(publish_centroids_topic, PointCloud2, queue_size=100)
    pub_cluster = rospy.Publisher(publish_cluster_topic, PointCloud2, queue_size=100)


    opticsCluster = OpticsClusterSimple(
        pub_centroids, pub_cluster, window_time_in_ms, min_samples_to_be_core_point, 
                 min_samples_in_cluster,
                 max_distance_to_consider_same_cluster)

    rospy.Subscriber(cloud_topic, PointCloud2, opticsCluster.addSingleCloud)

    print("=========== GTEC mmWave Optics Cluster Simple ============")

    # rospy.spin()
    while not rospy.is_shutdown():
        opticsCluster.loop()
        rate.sleep()
