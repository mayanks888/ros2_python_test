import unittest
import rclpy


def sum(a,b):
    return (a+b)

class suntest(unittest.TestCase):
    
    # this is similar to fixture
    def setUp(self):
        self.k=10
        self.j=10
        self.context = rclpy.context.Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node('mayank_node', namespace='/rclpy', context=self.context)
        
        
    # remove all the fixture setting
    def tearDown(self):
        print("teardown called")
        self.node.destroy_node()
        rclpy.shutdown(context=self.context)

    def testsum(self):
#        act
        result=sum(self.k,self.j)
        self.assertEqual(result,(self.k+self.j))
        self.assertIs(self.k,self.j)

if __name__== "__main__":
    unittest.main()