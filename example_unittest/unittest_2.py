import unittest
def sum(a,b):
    return (a+b)

class suntest(unittest.TestCase):
    # this is similar to fixture
    def setUp(self) -> None:
        self.a=10
        self.b=20

    def testsum(self):
#        act
        result=sum(self.a,self.b)
        self.assertEqual(result=(self.a+self.b))

if __name__== "__main__":
    unittest.main()