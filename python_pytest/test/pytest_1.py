import pytest

def func(a):
    return a+5

def test_method():
  assert func(6)==1
  print("my custom pytest ran")

#On terminal execute:  "pytest pytesr_1.py"