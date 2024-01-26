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
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header


class OverlapCluster(object):

    def __init__(self, window_time_in_ms:float, overlap_threshold:float, publisher):
        
        self.window_time_in_ms = window_time_in_ms
        self.overlap_threshold = overlap_threshold
        self.publisher = publisher
        
        self.last_window_time_ms = -1
        self.last_update_time_ms = -1
        self.last_clusters_count = -1
        self.circles = []
        

    

    def loop(self):
        current_time_ms = float(rospy.get_rostime().to_nsec())/1000000.0
        if (current_time_ms>0):
            if (self.last_window_time_ms<0):
                self.last_window_time_ms = current_time_ms
            
            elapsed_in_ms = current_time_ms - self.last_window_time_ms


            if elapsed_in_ms>=self.window_time_in_ms:
                self.processWindow()


    def addNewMeasurement(self, cloud: PointCloud2)-> None:
        points = pc2.read_points(cloud, field_names=("x", "y", "z", "error_est"), skip_nans=True)
        self.circles.extend(list(points))
  
    
    def remove_overlapping_circles(self, circles, overlap_threshold) -> []:
        def overlap(circle1, circle2):
            x1, y1, z1, r1 = circle1
            x2, y2, z2, r2 = circle2
            distance_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
            overlap_area = (r1 + r2) ** 2 - distance_squared
            return overlap_area > overlap_threshold
    
        # Sort the circles in descending order of radius
        circles.sort(key=lambda circle: circle[3], reverse=True)
    
        # Check for overlap and remove larger circles
        new_circles = []
        for circle in circles:
            if not any(overlap(circle, existing_circle) for existing_circle in new_circles):
                new_circles.append(circle)
            else:
                pass
                #print(f'Circle removed {circle}')

        return new_circles
    
    def publish_final_pointcloud(self, circles) -> None:
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
    
        fields = [
            pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="error_est", offset=12, datatype=pc2.PointField.FLOAT32, count=1),
        ]
        
        points = [(circle[0], circle[1], circle[2], circle[3]) for circle in circles]
        cloud = pc2.create_cloud(header, fields, points)
        self.publisher.publish(cloud)
    
       
             
    def processWindow(self) -> None:
        window_circles = self.circles.copy()
        self.circles = []
        
        clean_circles = self.remove_overlapping_circles(window_circles, self.overlap_threshold)
        
        if len(clean_circles)>0:
            if len(clean_circles)!=self.last_clusters_count:
                self.last_clusters_count = len(clean_circles)
                print(f'Num clusters {self.last_clusters_count} Removed {len(window_circles) - self.last_clusters_count}')
            self.publish_final_pointcloud(clean_circles)
        
        
        

if __name__ == "__main__":

    rospy.init_node('OverlapCluster', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    # Read parameters
    subscribe_topic = rospy.get_param('~subscribe_topic')
    publish_topic = rospy.get_param('~publish_topic')
    overlap_threshold = float(rospy.get_param('~overlap_threshold'))
    window_time_in_ms = int(rospy.get_param('~window_time_in_ms'))
    
    publisher= rospy.Publisher(publish_topic, PointCloud2, queue_size=100)


    overlap_cluster = OverlapCluster(window_time_in_ms, overlap_threshold, publisher)

    rospy.Subscriber(subscribe_topic, PointCloud2, overlap_cluster.addNewMeasurement)

    print("=========== GTEC mmWave Overlap Cluster ============")

    # rospy.spin()
    while not rospy.is_shutdown():
        overlap_cluster.loop()
        rate.sleep()