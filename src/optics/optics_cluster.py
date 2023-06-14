#!/usr/bin/env python

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


class TrackedObject(object):
    def __init__(self):
        self.alive_time = 0.0
        self.centroid = np.zeros(3)
        self.covariance = np.zeros(36)

    def add_time(self, time):
        self.alive_time += time

    def reset_time(self):
        self.alive_time = 0.0

    def update_position(self, centroid, covariance):
        self.centroid = centroid
        self.covariance = covariance

    def is_alive(self, max_time):
        return True if (self.alive_time <= max_time) else False

    def get_centroid(self):
        return self.centroid

    def get_pose_msg(self):
        poseMsg = PoseWithCovarianceStamped()
        poseMsg.header.stamp = rospy.Time.now()
        poseMsg.header.frame_id = "radar"
        poseMsg.pose.pose.position.x = self.centroid[0]
        poseMsg.pose.pose.position.y = self.centroid[1]
        poseMsg.pose.pose.position.z = self.centroid[2]
        poseMsg.pose.pose.orientation.w = 1.0
        poseMsg.pose.pose.orientation.x = 0.0
        poseMsg.pose.pose.orientation.y = 0.0
        poseMsg.pose.pose.orientation.z = 0.0
        cov = self.covariance
        poseMsg.pose.covariance = list(cov)
        return poseMsg


class OpticsCluster(object):

    def __init__(self,
                 publisher_centroids_cloud,
                 publisher_cluster_cloud,
                 publish_pose_topic_base,
                 window_time_in_ms,
                 min_points_cluster, 
                 max_alive_without_measurements,
                 max_distance_to_consider_same_cluster
                 ):
        self.publisher_centroids_cloud = publisher_centroids_cloud
        self.publisher_cluster_cloud = publisher_cluster_cloud
        self.min_points_cluster = 5
        self.publishers_pose = []
        self.tracked_objects = []
        self.last_window_time_ms = -1
        self.last_update_time_ms = -1
        self.max_alive = max_alive_without_measurements
        self.max_distance = max_distance_to_consider_same_cluster
        self.window_time_in_ms = window_time_in_ms

        for index in range(max_objects_tracked):
            topic = publish_pose_topic_base+'/' + str(index)
            pub = rospy.Publisher(
                topic, PoseWithCovarianceStamped, queue_size=100)
            self.publishers_pose.append(pub)

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

            for index in range(len(self.tracked_objects)):
                (self.tracked_objects[index]).add_time(elapsed_in_ms)

            self.tracked_objects = [i for i in self.tracked_objects if i.is_alive(self.max_alive)]

            for index in range(len(self.tracked_objects)):
                self.publish_pose(index, self.tracked_objects[index])



    def publish_pose(self, index_tracked, tracked_object):
        msg = tracked_object.get_pose_msg()
        self.publishers_pose[index_tracked].publish(msg)


    def addSingleCloud(self, cloud: PointCloud2)-> None:
        for p in pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
            if (self.windowed_points_first==True):
                self.windowed_points = np.array([[p[0], p[1], p[2]]])
                self.windowed_points_first = False
            else:
                self.windowed_points = np.vstack(
                    [self.windowed_points, [p[0], p[1], p[2]]])


    def processWindow(self, elapsed_time) -> None:
        print("Len points: " + str(len(self.windowed_points)))
        if (len(self.windowed_points) >= self.min_points_cluster):
                model = OPTICS(min_samples=self.min_points_cluster,
                               min_cluster_size=0.05)
                pred = model.fit_predict(self.windowed_points)

                space = np.arange(len(self.windowed_points))
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

                for i in range(len(clusters_ids)):
                    cluster_id = clusters_ids[i]
                    if cluster_id > -1:
                        points_of_cluster = self.windowed_points[labels == cluster_id]
                        centroid_of_cluster = np.mean(
                            points_of_cluster, axis=0)
                        centroids[i] = centroid_of_cluster
                        rgb = [0, 0, 0]
                        if (i < len(colors)):
                            rgb = colors[i]
                        pt = np.append(centroid_of_cluster, rgb)
                        points_centroid_cloud.append(pt)

                        if (i < self.max_poses):
                            closer = -1
                            if (len(self.tracked_objects) == 0):
                                self.tracked_objects.append(TrackedObject())
                                closer = 0
                            else:
                                min_distance = 0
                                closer = -1
                                for i in range(len(self.tracked_objects)):
                                    dist = np.linalg.norm(centroid_of_cluster-(self.tracked_objects[i]).get_centroid())
                                    if (dist<= self.max_distance):
                                        if (closer==-1):
                                            min_distance = dist
                                            closer = i
                                        else:
                                            if dist<=min_distance:
                                                closer = i
                                                min_distance= dist

                                if closer<0:
                                    #No closer point, we add a new one
                                    self.tracked_objects.append(TrackedObject())
                                    closer = len(self.tracked_objects)-1

                            cov = np.zeros(36)
                            cov[0] = np.var(points_of_cluster[:, 0])
                            cov[7] = np.var(points_of_cluster[:, 1])
                            cov[14] = np.var(points_of_cluster[:, 2])

                            (self.tracked_objects[closer]).update_position(centroid_of_cluster, cov)
                            (self.tracked_objects[closer]).reset_time()

                centroids_cloud = pc2.create_cloud(
                    header_cloud, fields_cloud, points_centroid_cloud)
                self.publisher_centroids_cloud.publish(centroids_cloud)

                for i in range(len(self.windowed_points)):
                    rgb = [0, 0, 0]
                    index_label = np.where(clusters_ids == labels[i])
                    if (index_label[0][0] < len(colors)):
                        rgb = colors[index_label[0][0]]
                    pt = np.append(self.windowed_points[i], rgb)
                    points_cluster_cloud.append(pt)

                cluster_cloud = pc2.create_cloud(
                    header_cloud, fields_cloud, points_cluster_cloud)
                self.publisher_cluster_cloud.publish(cluster_cloud)


if __name__ == "__main__":

    rospy.init_node('OpticsCluster', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    # Read parameters
    cloud_topic = rospy.get_param('~cloud_topic')
    publish_centroids_topic = rospy.get_param('~publish_centroids_topic')
    publish_cluster_topic = rospy.get_param('~publish_cluster_cloud_topic')
    publish_pose_topic = rospy.get_param('~publish_pose_topic')

    pub_centroids = rospy.Publisher(
        publish_centroids_topic, PointCloud2, queue_size=100)
    pub_cluster = rospy.Publisher(
        publish_cluster_topic, PointCloud2, queue_size=100)

    
    min_points_cluster = 10
    window_time_in_ms = 1000
    max_target_alive_time_in_ms = 3000
    max_distance_same_cluster_in_m = 0.5

    opticsCluster = OpticsCluster(
        pub_centroids, pub_cluster, publish_pose_topic, window_time_in_ms, min_points_cluster, max_target_alive_time_in_ms, max_distance_same_cluster_in_m)

    rospy.Subscriber(cloud_topic, PointCloud2,
                     opticsCluster.point_cloud_listener)

    print("=========== GTEC mmWave Optics Cluster ============")

    # rospy.spin()
    while not rospy.is_shutdown():
        opticsCluster.loop()
        rate.sleep()
