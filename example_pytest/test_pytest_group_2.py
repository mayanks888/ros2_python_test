import pytest



def test_method1():
    a=5
    b=5
    assert a==b

def test_method2():
    a=5
    b=8
    assert a==b

# on terminal
#  py.test pytest_group_2.py -k method2 -v
#  pytest pytest_group_2.py -k method1 -v