import pytest


@pytest.mark.one
def test_method():
    a=5
    b=5
    assert a==b

@pytest.mark.two
def test_method2():
    a=5
    b=8
    assert a==b


# on terminal
# pytest pytest_marker_3.py -m one
# pytest pytest_marker_3.py -m two
#