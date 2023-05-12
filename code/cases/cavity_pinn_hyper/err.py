#!/usr/bin/env python3

import statistics
import numpy as np
import os
import sys

dtype = np.dtype("float64")
size = os.path.getsize(sys.argv[-1])
u0 = np.fromfile(sys.argv[-1], dtype)

for path in sys.argv[1:]:
    u = np.fromfile(path, dtype)
    err = np.mean((u - u0)**2)**0.5
    print("%.16e" % err)
