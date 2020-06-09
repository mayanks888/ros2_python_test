import pytest
class TestClass():
    def test_one(self):
        x = "This"
        assert 'h' in x
    def test_two(self):
        x = "hello"
        assert hassattr(x, 'hello')