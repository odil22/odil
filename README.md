# ODIL

Optimizing a DIscrete Loss (ODIL) to solve forward and inverse problems
for partial differential equations using machine learning tools

This repository contains code and configuration that can be imported at
<https://codeocean.com> to create a compute capsule.

## Requirements

Python packages

```
tensorflow
matplotlib
scipy
numpy
pyamg
```

Debian packages

```
libmpich-dev
rsync
```

## Run

To run a set of minimal examples (runtime about 5 minutes)

```
cd code
./run
```

To run in production mode (runtime 2-3 days with 128 cores)

```
cd code
minimal=0 ./run
```
