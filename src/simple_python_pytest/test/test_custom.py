import pytest

def func(a):
    return a+5

def test_custom():
  print("my custom pytest ran")
  assert func(6)==11
 

#On terminal execute:  "pytexst pytesr_1.py"