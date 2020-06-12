# Copyright 2018 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import unittest
from unittest.mock import Mock
import argparse

import builtin_interfaces.msg
import rclpy
# from python_pytest.subscriber_member_function import *
import rectify_test
from std_msgs.msg import String
from traffic_light_msgs.msg import TrafficLightStruct
from cv_bridge import CvBridge
import cv2

class TestTimeSource(unittest.TestCase):

    def setUp(self):
        # self.context = rclpy.context.Context()
        # rclpy.init(context=self.context)
        # self.node = rclpy.create_node('mynode', namespace='/rclpy', context=self.context)
        rclpy.init(args=None)
        ####################################################333333
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default='/home/mayank_s/playing_ros/c++/ros2_play_old/src/rectifier/rectifier/cfg/yolov3.cfg', help='*.cfg path')
        parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
        parser.add_argument('--weights', type=str, default='/home/mayank_s/playing_ros/c++/ros2_play_old/src/rectifier/rectifier/weights/yolov3.pt',
                            help='weights path')
        parser.add_argument('--source', type=str, default='data/samples', help='source')
        parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
        parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        opt = parser.parse_args()
        ################################################################################
        self.myobject = rectify_test.Rectifier(opt)
        # rclpy.spin(self.minimal_subscriber)
        # rclpy.spin_once(self.myobject)
        self.msg=String.data
        self.msg="mayank"

    def tearDown(self):
        # self.node.destroy_node()
        # rclpy.shutdown(context=self.context)
        
        self.myobject.destroy_node()
        rclpy.shutdown()
        
    def testrectifeir(self):
        # 333333333333333333333333333333333
        cv_image=cv2.imread("/home/mayank_s/playing_ros/python/test/ros2_python_test/src/rectifier/rectifier/rectifer_test.jpg",1)
        bridge = CvBridge()
        image_message = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough") 
        mydata=TrafficLightStruct()
        # mydata.cropped_roi=[2,5,60,20]
        mydata.raw_image=image_message
        
        # $$$$$$$$$$$$$$$$$$$
        # mydata.cropped_roi= [941,264,500,500]
        mydata.cropped_roi.x_offset=941
        mydata.cropped_roi.y_offset=264
        mydata.cropped_roi.height=500
        mydata.cropped_roi.width=500

        # (941,264,500,500)
        self.myobject.img_callback(mydata)
        # cool=subscriber_member_function.MinimalSubscriber()
        1
#        act
        # result=sum(self.k,self.j)
        # self.assertEqual(result,(self.k+self.j))
        # self.assertIs(self.k,self.j)
        

if __name__ == '__main__':
    unittest.main()
