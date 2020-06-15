# !/usr/bin/python3
import pytest
import time
import unittest
# from unittest.mock import Mock
import argparse
# from sensor_msgs.msg import Image
import numpy as np
# import builtin_interfaces.msg
import rclpy
from traffic_light_msgs.msg import TrafficLightStruct
from cv_bridge import CvBridge
import cv2

# # from python_pytest.subscriber_member_function import *
from rectifier.rectify_test import *
import rectifier.rectify_test
# # import rectify_test
# from std_msgs.msg import String



class TestClass():
    def setup_class(self):
        print("setup_class called once for the class")

    def teardown_class(self):
        print("teardown_class called once for the class")


    def setup_method(self):
        print("setup_method called for every method")

    def teardown_method(self):
        print("teardown_method called for every method")


    def test_one(self):
        print("one")
        assert True
        print("one after")

    def test_two(self):
        print("two")
        assert True
        print("two after")

    def test_three(self):
        print("three")
        assert False
        print("three after")