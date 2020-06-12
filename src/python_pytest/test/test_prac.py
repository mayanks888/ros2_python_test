# Copyright 2017 Open Source Robotics Foundation, Inc.
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

# import asyncio
# import time
import unittest

import rclpy
# from rclpy.executors import MultiThreadedExecutor
# from rclpy.executors import SingleThreadedExecutor


class TestExecutor(unittest.TestCase):
    # @classmethod
    def setUp(self):
        self.context = rclpy.context.Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node('mayank_node', namespace='/rclpy', context=self.context)

    # @classmethod
    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown(context=self.context)

    # def func_execution(self, executor):
    #     got_callback = False

    #     def timer_callback():
    #         nonlocal got_callback
    #         got_callback = True

    #     tmr = self.node.create_timer(0.1, timer_callback)

    #     assert executor.add_node(self.node)
    #     executor.spin_once(timeout_sec=1.23)
    #     # TODO(sloretz) redesign test, sleeping to workaround race condition between test cleanup
    #     # and MultiThreadedExecutor thread pool
    #     time.sleep(0.1)

    #     self.node.destroy_timer(tmr)
    #     return got_callback

if __name__ == '__main__':
    unittest.main()
