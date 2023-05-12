# Install

```
cd odil-dev/poc/34_cavity
git clone git@github.com:lululxvi/deepxde .deepxde
(cd .deepxde && git apply) < deepxde.patch
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
PYTHONPATH= sh Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -u -b
eval "`$HOME/miniconda3/bin/conda shell.bash activate`"
(base) $ conda install -c nvidia cuda
(base) $ conda install -c conda-forge cudatoolkit=11.8.0
(base) $ pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.* ./.deepxde
```

# Run

```
sbatch -C gpu -A s1160 -t 5 --wrap='set -x; . ./env.conda && python3 -u train.py -s 100 -i 0 -o adam -l 0.001 -a 32 32 32'
```
