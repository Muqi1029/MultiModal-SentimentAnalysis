import pytest
import os


class TestPath:
    def test_num(self):
        print(os.listdir("../data/"))
        print(len(os.listdir("../data")) // 2)
