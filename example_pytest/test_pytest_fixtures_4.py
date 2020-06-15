
import pytest
# fixture will play everytime in the code
@pytest.fixture
def myvales():
    a=3
    b=5
    return (a,b)

def test1(myvales):
    assert myvales[0]==9

def test2(myvales):
    assert myvales[1]==5

# pytest pytest_fixtures_4.py
# pytest pytest_fixtures_4.py -k test2 -v

if __name__== "__main__":

    pytest.main()



