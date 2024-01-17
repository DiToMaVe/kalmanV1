#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import Matrix
import example
import numpy as np


m3 = np.array([[1,2,3],[4,5,6]]).astype(np.float64)
print(hex(id(m3)))
print(hex(m3.__array_interface__['data'][0]))
m4 = Matrix(m3)
print(m4)

# for i in range(m4.rows()):
#     for j in range(m4.cols()):
#         print(m4[i, j], end = ' ')
#     print()
