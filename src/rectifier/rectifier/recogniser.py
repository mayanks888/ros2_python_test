# !/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int16, Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv_bridge
# General libs
import numpy as np
import os
import time
import cv2
import torch
import shutil
import argparse
from sys import platform
from rectifier.models import *
from rectifier.utils import *
from rectifier.torch_utils import *

# from .models import *
# from .utils import *
# from .torch_utils import *
from std_msgs.msg import String
from traffic_light_msgs.msg import TrafficLightStruct, Detection2D
import torchvision.transforms as transforms
from PIL import Image

class Recogniser(Node):
    def __init__(self, args_):
        self.opt = args_
        # print(self.opt)
        img_size = (
        320, 192) if ONNX_EXPORT else self.opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        out, source, weights, self.half, view_img, save_txt = self.opt.output, self.opt.source, self.opt.weights, self.opt.half, self.opt.view_img, self.opt.save_txt
        self.device = select_device(device='cpu' if ONNX_EXPORT else self.opt.device)
        self.model = Color_Net_CNN()
        self.model.load_state_dict(torch.load(weights))
        self.model.to(self.device).eval()
        self.traffic_light = ['black', 'green', 'red']
        # model.to(device).eval()
        #######################################################
        super().__init__('mytalker')
        self._pub = self.create_publisher(TrafficLightStruct, '/snowball/perception/traffic_light/recogniser_output')
        # self._pub = self.create_publisher(TrafficLightStruct, '/tl_bbox_info')
        # self._pub = self.create_publisher(TrafficLightStruct, '/snowball/perception/traffic_light/color_output')
        self._cv_bridge = CvBridge()
        super().__init__('recogniser')
        self._sub = self.create_subscription(TrafficLightStruct, '/snowball/perception/traffic_light/processor', self.img_callback, 10)
   
        # self._pub = self.create_publisher(TrafficLightStruct, '/tl_bbox_info')
        
        
        print("ready to process recogniser----------------------------------------------------------")

    # Camera image callback
    def img_callback(self, selected_data):
        t = time.time()
        image_msg=selected_data.raw_image
        bridge = cv_bridge.CvBridge()
        cv_img = bridge.imgmsg_to_cv2(image_msg, 'passthrough')
        # print("image shape is ", cv_img.shape)
        image_np = cv_img
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BAYER_BG2BGR, 3)
        selected_roi=selected_data.selected_box
        print('new data',selected_roi)
        # crop_img = img[y:y+h, x:x+w]
        crop_img = image_np[selected_roi.y_offset:selected_roi.y_offset+selected_roi.height, selected_roi.x_offset:selected_roi.x_offset+selected_roi.width]
        img0=Image.fromarray(crop_img)

        img0 = img0.convert("RGB")
        # img0 = img0.convert("BGR")
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        img = transform(img0)
        img = img.unsqueeze(0)
        img = img.to(self.device)

        output = self.model(img)
        print(output)
        data = torch.argmax(output, dim=1)
        # print(output)

        data=data.type(torch.cuda.FloatTensor)
        light_color = self.traffic_light[int(data)]
        c1, c2 = (int(selected_roi.x_offset), int(selected_roi.y_offset)), (int(selected_roi.x_offset+selected_roi.width), int(selected_roi.y_offset+selected_roi.height))
        # data=2
        if data==0:#black
            cv2.rectangle(image_np, c1, c2, color=(255, 255, 255), thickness=2)
        elif data==1: #green
            cv2.rectangle(image_np, c1, c2, color=(0, 255, 0), thickness=2)
        elif data==2: #red
             cv2.rectangle(image_np, c1, c2, color=(0, 0, 255), thickness=2)
        cv2.imshow('color detector', image_np)
        ch = cv2.waitKey(1)
        
        thorth = time.time()
        time_cost = round((thorth - t) * 1000)
        print("Inference Time", time_cost)
        # print("publishing bbox:- ", self.tl_bbox.data)
        # print("publishing bbox:- ", ppros_data.detections)
        another_time_msg=self.get_clock().now().to_msg()  
        selected_data.selected_box.header.stamp=another_time_msg  
        selected_data.selected_box.color=int(data)
        self._pub.publish(selected_data)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/module/src/rectifier/rectifier/cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='/home/mayank_s/playing_ros/c++/ros2_play_old/src/rectifier/rectifier/weights/color_model.pt',
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
    # print(opt)
    rclpy.init(args=args)
    # with torch.no_grad():
    node = Recogniser(opt)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()