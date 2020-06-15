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
# !/usr/bin/python3

import time
import unittest
from unittest.mock import Mock
import argparse
from sensor_msgs.msg import Image
import numpy as np
import builtin_interfaces.msg
import rclpy
# from python_pytest.subscriber_member_function import *
# from rectifier.rectify_test import *
import rectifier.rectify_test
# import rectify_test
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
        self.myobject =  rectifier.rectify_test.Rectifier(opt)
        # rclpy.spin(self.minimal_subscriber)
        # rclpy.spin_once(self.myobject)

    def tearDown(self):
        # self.node.destroy_node()
        # rclpy.shutdown(context=self.context)
        self.myobject.destroy_node()
        rclpy.shutdown()

    def CreateDMessage(self,img_depth):

                bridge = CvBridge()
                msg_d=Image()
                msg_d.encoding = "bayer_rggb8"
                msg_d.height = img_depth.shape[0]
                msg_d.width = img_depth.shape[1]
                msg_d.data = bridge.cv2_to_imgmsg(img_depth, "bgr8").data

                return msg_d
    
    def test_rectifeir(self):
        # 333333333333333333333333333333333
        cv_image=cv2.imread("/home/mayank_s/playing_ros/python/test/ros2_python_test/src/rectifier/rectifier/rectifer_test.jpg",1)
        bridge = CvBridge()
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_BG2BGR)
        # Encode to "bayer_rggb8"
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_BG2BGR)
        # msg=self.CreateDMessage(cv_image)
        image_message = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8") 
        # image_message.encoding="bayer_rggb8"
        # image_message = bridge.cv2_to_imgmsg(msg, encoding="bayer_rggb8") 
        # image_message = cv2.cvtColor(image_message, cv2.COLOR_BAYER_BG2GRAY)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
        mydata=TrafficLightStruct()
        mydata.raw_image=image_message
        # #############################3
        mydata.cropped_roi.x_offset=941
        mydata.cropped_roi.y_offset=264
        mydata.cropped_roi.height=450
        mydata.cropped_roi.width=450
        
        return_data=self.myobject.img_callback(mydata)
        print(return_data.detections)
        self.assertIsNotNone(return_data.detections[0])
        # self.assertIsNone(return_data.detections[0])
        # result=sum(self.k,self.j)
        # self.assertEqual(result,(self.k+self.j))
        # self.assertIs(self.k,self.j)
        

if __name__ == '__main__':
    unittest.main()
