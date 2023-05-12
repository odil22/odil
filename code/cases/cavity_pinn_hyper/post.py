#!/usr/bin/env python3

import contourpy
import os
import numpy as np
import math
import sys
import re

u1 = 1
dtype = np.dtype("float64")

for path in sys.argv[1:]:
    dirname = os.path.dirname(path)
    suffix = re.search("\.[0-9]+\.[^\.]*$", path).group(0)
    timestep = re.search("^\.[0-9]+\.", suffix).group(0)

    upath = os.path.join(dirname, "u" + timestep + "raw")
    vpath = os.path.join(dirname, "v" + timestep + "raw")
    omegapath = os.path.join(dirname, "omega" + suffix)
    fieldpath = os.path.join(dirname, "fields" + timestep + "xdmf2")
    levelpath = os.path.join(dirname, "levels" + timestep + "xdmf2")

    size = os.path.getsize(upath)
    n = math.isqrt(size // dtype.itemsize)
    if n * n * dtype.itemsize != size:
        sys.stderr.write(
            "post.py: something wrong with the size of '%s (size: %ld)'\n" %
            (path, size))
        sys.exit(2)

    u = np.fromfile(upath, dtype)
    v = np.fromfile(vpath, dtype)
    omega = np.memmap(omegapath, dtype, "w+", 0, n * n)
    for j in range(n):
        for i in range(n):
            ox = (
                2 * v[i + n * j] if i - 1 < 0 else  #
                -2 * v[i + n * j] if i + 1 >= n else
                (v[i + 1 + n * j] - v[i - 1 + n * j]) / 2)
            oy = (
                2 * u[i + n * j] if j - 1 < 0 else  #
                2 * (u1 - u[i + n * j]) if j + 1 >= n else
                (u[i + n * (j + 1)] - u[i + n * (j - 1)]) / 2)
            omega[i + n * j] = n * (ox - oy)
    fields = ("u", "v", "p", "omega")
    h = 1 / n
    with open(fieldpath, "w") as file:
        file.write("""\
<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf
    Version="2.0">
  <Domain>
   <Grid>
     <Topology
        TopologyType="3DCORECTMesh"
        Dimensions="2 %ld %ld"/>
     <Geometry
       GeometryType="ORIGIN_DXDYDZ">
       <DataItem
          Name="Origin"
          Dimensions="3">
            0 0 0
       </DataItem>
       <DataItem
          Name="Spacing"
          Dimensions="3">
            1 %.16e %.16e
       </DataItem>
     </Geometry>
""" % (n + 1, n + 1, h, h))
        for name in fields:
            file.write("""
     <Attribute
        Name="%s"
        Center="Cell">
       <DataItem
          Dimensions="1 %ld %ld"
          Format="Binary"
          Precision="8">
          %s
       </DataItem>
     </Attribute>
""" % (name, n, n, name + timestep + "raw"))
        file.write("""\
    </Grid>
  </Domain>
</Xdmf>
""")

    levels = (-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3)
    gen = contourpy.contour_generator(z=np.reshape(omega, (n, n)))
    with open(levelpath, "w") as file:
        XY = []
        levs = []
        for lev in levels:
            lines = gen.lines(lev)
            XY.extend(lines)
            levs += [lev] * len(lines)
        file.write("""\
<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf
    Version="2.0">
  <Domain>
    <Grid>
      <Geometry
           Type="XY">
         <DataItem
            Dimensions="%ld">\n
""" % (2 * sum(len(xy) for xy in XY)))
        for xy in XY:
            for x, y in xy:
                file.write("         %.16e %.16e\n" %
                           (h / 2 + x * h, h / 2 + y * h))
        file.write("""\
         </DataItem>
      </Geometry>
      <Topology
           Dimensions="%ld"
           Type="Mixed">
         <DataItem
             DataType="Int"
             Dimensions="%ld">
""" % (len(XY), sum(len(xy) + 2 for xy in XY)))
        cnt = 0
        for xy in XY:
            file.write("          2\n")
            file.write("          %ld\n" % len(xy))
            for i in xy:
                file.write("          %ld\n" % cnt)
                cnt += 1
        file.write("""\
        </DataItem>
      </Topology>
      <Attribute
          Name="level"
          Center="Cell">
        <DataItem
            Dimensions="%ld">
""" % len(levs))
        for lvl in levs:
            file.write("          %.16e\n" % lvl)
        file.write("""\
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
""")
