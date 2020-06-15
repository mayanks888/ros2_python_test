import unittest
# import pytest

def sum(a,b):
    return (a+b)

class suntest(unittest.TestCase):
    # this is similar to fixture
    def setUp(self):
        self.k=10
        self.j=10
    # remove all the fixture setting
    def tearDown(self):
        print("teardown called")
    def testsum(self):
#        act
        result=sum(self.k,self.j)
        self.assertEqual(result,(self.k+self.j))
        self.assertIs(self.k,self.j)

if __name__== "__main__":
    unittest.main()