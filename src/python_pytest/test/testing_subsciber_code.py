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

import builtin_interfaces.msg
import rclpy
from python_pytest.subscriber_member_function import *


class TestTimeSource(unittest.TestCase):

    def setUp(self):
        self.context = rclpy.context.Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node('mynode', namespace='/rclpy', context=self.context)

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown(context=self.context)
        
    def testrectifeir(self):
        1
#        act
        # result=sum(self.k,self.j)
        # self.assertEqual(result,(self.k+self.j))
        # self.assertIs(self.k,self.j)
        

if __name__ == '__main__':
    unittest.main()
