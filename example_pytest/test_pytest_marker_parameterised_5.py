
import pytest

# fixture will play everytime in the code
@pytest.mark.parametrize("x,y,z",[(1,2,3),(1,2,5)])
def test1(x,y,z):
    assert x+y==z







