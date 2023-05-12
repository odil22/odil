#!/usr/bin/env python3

import math
import numpy as np
import os
import sys

Initializers = (
    "Glorot normal",
    "Glorot uniform",
    "He normal",
    "He uniform",
    "LeCun normal",
    "LeCun uniform",
    "Orthogonal",
    "zeros",
)

Optimizers = (
    "L-BFGS-B",
    "adadelta",
    "adagrad",
    "adam",
    "rmsprop",
    "sgd",
    "sgdnesterov",
)


def pde(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)
    v_y = dde.grad.jacobian(y, x, i=1, j=1)
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_y = dde.grad.jacobian(y, x, i=2, j=1)
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    continuity = u_x + v_y
    x_momentum = (u * u_x + v * u_y) + p_x - 1 / Re * (u_xx + u_yy)
    y_momentum = (u * v_x + v * v_y) + p_y - 1 / Re * (v_xx + v_yy)
    return continuity, x_momentum, y_momentum


def boundary_u0(x, on_boundary):
    return on_boundary and (math.isclose(x[0], 0, abs_tol=1e-8)
                            or math.isclose(x[0], 1, abs_tol=1e-8)
                            or math.isclose(x[1], 0, abs_tol=1e-8))


def boundary_u1(x, on_boundary):
    return on_boundary and math.isclose(x[1], 1, abs_tol=1e-8)


def boundary_v(x, on_boundary):
    return on_boundary


def b1(x):
    return u1


def b0(x):
    return 0


class Callback:

    def __init__(self, model, grid, period, output):
        self.grid = grid
        self.niter = 0
        self.period = period
        self.output = output

    def __call__(self, x):
        if self.niter % self.period == 0:
            packed_var_val = x
            var_vals = [
                packed_var_val[packing_slice]
                for packing_slice in model.train_step._packing_slices
            ]
            model.sess.run(
                model.train_step._var_updates,
                feed_dict=dict(
                    zip(model.train_step._update_placeholders, var_vals)),
            )
            u, v, p = model.predict(self.grid).T
            for field, name in zip((u, v, p), ("u", "v", "p")):
                path = os.path.join(self.output,
                                    "%s.%09d.raw" % (name, self.niter))
                with open(path, "wb") as file:
                    file.write(field.tobytes())
        self.niter += 1


class AdamCallback:

    def __init__(self, model, grid, period, output):
        self.grid = grid
        self.niter = 0
        self.period = period
        self.output = output

    def set_model(self, mode):
        pass

    def on_train_begin(self):
        pass

    def on_epoch_begin(self):
        if self.niter % self.period == 0:
            u, v, p = model.predict(self.grid).T
            for field, name in zip((u, v, p), ("u", "v", "p")):
                path = os.path.join(self.output,
                                    "%s.%09d.raw" % (name, self.niter))
                with open(path, "wb") as file:
                    file.write(field.tobytes())
        self.niter += 1

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        pass


Seed = None
Initializer = None
Optimizer = None
LearningRate = None
Arch = None
Resample = None
NumDomain = None
while True:
    sys.argv.pop(0)
    if not sys.argv or sys.argv[0][0] != "-" or len(sys.argv[0]) < 2:
        break
    if sys.argv[0][1] == "h":
        sys.stderr.write(
            "usage train.py -o 'L-BFGS-B|adam|...' -a"
            " 16 16|32 32 32|..' -i '0-7' -n 'number of domain samples' [-l 'learning rate'] [-r"
            " 'resampling period'] -s seed\n")
        sys.exit(2)
    elif sys.argv[0][1] == "o":
        sys.argv.pop(0)
        if not sys.argv:
            sys.stderr.write("train.py: error: option -o needs an argument\n")
            sys.exit(2)
        Optimizer = sys.argv[0]
        if Optimizer not in Optimizers:
            sys.stderr.write(
                "train.py: error: unknown optimizer '%s', possible values are %s\n"
                % (Optimizer, Optimizers))
            sys.exit(2)
    elif sys.argv[0][1] == "a":
        Arch = []
        while len(sys.argv) > 1 and sys.argv[1][0] != "-":
            sys.argv.pop(0)
            try:
                arch = int(sys.argv[0])
            except ValueError:
                sys.stderr.write("train.py: not an integer '%s'\n" %
                                 sys.argv[0])
                sys.exit(2)
            Arch.append(arch)
        Arch = tuple(Arch)
    elif sys.argv[0][1] == "i":
        sys.argv.pop(0)
        if not sys.argv:
            sys.stderr.write("train.py: error: option -i needs an argument\n")
            sys.exit(2)
        try:
            Initializer = int(sys.argv[0])
        except ValueError:
            sys.stderr.write("train.py: not an integer '%s'\n" % sys.argv[0])
            sys.exit(2)
        if Initializer < 0 or Initializer >= len(Initializers):
            sys.stderr.write(
                "train.py: argument of -i should be between >= 0 and < %d\n" %
                len(Initializers))
            sys.exit(2)
    elif sys.argv[0][1] == "l":
        sys.argv.pop(0)
        if not sys.argv:
            sys.stderr.write("train.py: error: option -l needs an argument\n")
            sys.exit(2)
        try:
            LearningRate = float(sys.argv[0])
        except ValueError:
            sys.stderr.write("train.py: not a float '%s'\n" % sys.argv[0])
            sys.exit(2)
    elif sys.argv[0][1] == "r":
        sys.argv.pop(0)
        if not sys.argv:
            sys.stderr.write("train.py: error: option -r needs an argument\n")
            sys.exit(2)
        try:
            Resample = int(sys.argv[0])
        except ValueError:
            sys.stderr.write("train.py: not an integer '%s'\n" % sys.argv[0])
            sys.exit(2)
    elif sys.argv[0][1] == "n":
        sys.argv.pop(0)
        if not sys.argv:
            sys.stderr.write("train.py: error: option -n needs an argument\n")
            sys.exit(2)
        try:
            NumDomain = int(sys.argv[0])
        except ValueError:
            sys.stderr.write("train.py: not an integer '%s'\n" % sys.argv[0])
            sys.exit(2)
    elif sys.argv[0][1] == "s":
        sys.argv.pop(0)
        if not sys.argv:
            sys.stderr.write("train.py: error: option -s needs an argument\n")
            sys.exit(2)
        try:
            Seed = int(sys.argv[0])
        except ValueError:
            sys.stderr.write("train.py: not an integer '%s'\n" % sys.argv[0])
            sys.exit(2)
    else:
        sys.stderr.write("train.py: error: unknown option '%s'\n" %
                         sys.argv[0])
        sys.exit(2)
sys.argv.append('')

if Seed == None:
    sys.stderr.write("train.py: error: seed (-s) is not set\n")
    sys.exit(2)
if NumDomain == None:
    sys.stderr.write("train.py: error: seed (-n) is not set\n")
    sys.exit(2)
if Initializer == None:
    sys.stderr.write("train.py: error: seed (-i) is not set\n")
    sys.exit(2)
if Optimizer == None:
    sys.stderr.write("train.py: error: seed (-o) is not set\n")
    sys.exit(2)
if Optimizer != "L-BFGS-B" and LearningRate == None:
    sys.stderr.write("train.py: error: seed (-l) is not set\n")
    sys.exit(2)
if Optimizer == "L-BFGS-B" and LearningRate != None:
    sys.stderr.write("train.py: error: -l is not used for L-BFGS-B\n")
    sys.exit(2)
if Optimizer == "L-BFGS-B" and Resample != None:
    sys.stderr.write("train.py: error: -r is not used for L-BFGS-B\n")
    sys.exit(2)
if Arch == None:
    sys.stderr.write("train.py: error: network architecture (-a) is not set\n")
    sys.exit(2)
Fields = (("optimizer", Optimizer), ("arch", Arch),
          ("initializer", Initializer), ("learningrate", LearningRate),
          ("numdomain", NumDomain),
          ("resample", Resample), ("seed", Seed))
output = "_".join(a + ":" + str(b).replace(" ", "") for a, b in Fields if b != None
                  if b != None)
os.makedirs(output, exist_ok=True)
sys.stdout = open(os.path.join(output, "out.log"), "w")
sys.stderr = open(os.path.join(output, "err.log"), "w")

import deepxde as dde

dde.config.set_random_seed(Seed)
dde.config.set_default_float("float64")
n = 128
Re = 100
u1 = 1
num_boundary = 400
period = 1000
maxiter = 500000

geom = dde.geometry.Rectangle([0, 0], [1, 1])
bc = (
    dde.icbc.DirichletBC(geom, b1, boundary_u1, component=0),
    dde.icbc.DirichletBC(geom, b0, boundary_u0, component=0),
    dde.icbc.DirichletBC(geom, b0, boundary_v, component=1),
)
x0 = np.array([0, 0])
observe = dde.icbc.PointSetBC(np.array([x0]), np.array([0]), component=2)
data = dde.data.PDE(
    geom,
    pde,
    bc + (observe, ),
    num_domain=NumDomain,
    num_boundary=num_boundary,
    anchors=np.array([x0]),
    num_test=1,
    train_distribution="pseudo",
)

sys.stderr.write("train.py: initializer: '%s'\n" % Initializers[Initializer])
net = dde.nn.FNN((2, ) + Arch + (3, ), "tanh", Initializers[Initializer])
model = dde.Model(data, net)
h = 1 / n
grid = [(h / 2 + i * h, h / 2 + j * h) for j in range(n) for i in range(n)]

if Optimizer == "L-BFGS-B":
    callback = Callback(model, grid, period, output)
    model.compile(Optimizer)
    model.train_step.optimizer_kwargs = {
        'options': {
            'gtol': 0,
            'ftol': 0,
            'maxfun': math.inf,
            'maxiter': maxiter,
        },
        'callback': callback,
    }
    model.train()
else:
    callback = AdamCallback(model, grid, period, output)
    model.compile(Optimizer, lr=LearningRate)
    if Resample == None:
        model.train(iterations=maxiter, callbacks=[callback])
    else:
        resampler = dde.callbacks.PDEPointResampler(period=Resample)
        model.train(iterations=maxiter, callbacks=[callback, resampler])
