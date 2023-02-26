import numpy as np
import scipy.optimize
from argparse import Namespace
from tfwrap import tf
import math


class Optimizer:

    def __init__(self, name=None, displayname=None, dtype=None):
        self.name = name
        self.displayname = displayname if displayname is not None else name
        self.dtype = dtype
        self.last_loss = None
        self.last_residual = None
        self.evals = 0

    def run(self,
            x0,
            loss_grad,
            epochs,
            callback=None,
            epoch_start=0,
            **kwargs):
        info = Namespace()
        info.evals = 0  # Number of `loss_grad()` evaluations.
        info.epochs = 0  # Number of epochs actually done.
        return x0, info


class LbfgsbOptimizer(Optimizer):

    def __init__(self,
                 pgtol=1e-16,
                 m=50,
                 maxls=50,
                 factr=0,
                 dtype=None,
                 **kwargs):
        """
        pgtol: `float`
            Gradient convergence condition.
        m: `int`
            Maximum number of variable metric corrections used to define
            the limited memory matrix.
        maxls: `int`
            Maximum number of line search steps (per iteration).
        factr: `float`
            Convergence condition:
               1e12 (low accuracy),
               1e7 (moderate accuracy),
               10.0 (extremely high accuracy)
        """
        super().__init__(name="lbfgsb", displayname="L-BFGS-B", dtype=dtype)
        self.pgtol = pgtol
        self.m = m
        self.maxls = maxls
        self.factr = factr
        self.evals = 0

    def run(self,
            x0,
            loss_grad,
            epochs=None,
            callback=None,
            epoch_start=0,
            **kwargs):
        self.epoch = epoch_start

        def callback_wrap(x):
            self.epoch += 1
            if callback:
                callback(x, self.epoch, opt=self)

        def func_wrap(x):
            self.evals += 1
            loss, grad, last_residual = loss_grad(x, self.epoch)
            loss = loss.numpy().astype(np.float64)
            grad = grad.numpy().astype(np.float64)
            self.last_residual = [r.numpy() for r in last_residual]
            self.last_loss = loss
            return loss, grad

        x0 = x0.numpy()

        x, f, sinfo = scipy.optimize.fmin_l_bfgs_b(func=func_wrap,
                                                   x0=x0,
                                                   maxiter=epochs,
                                                   pgtol=self.pgtol,
                                                   m=self.m,
                                                   maxls=self.maxls,
                                                   factr=self.factr,
                                                   maxfun=math.inf,
                                                   callback=callback_wrap)
        info = Namespace()
        info.epochs = sinfo['nit']
        info.evals = sinfo['funcalls']
        info.task = sinfo['task']
        return x, info


class LbfgsOptimizer(Optimizer):

    def __init__(self,
                 pgtol=1e-16,
                 m=50,
                 maxls=50,
                 factr=0,
                 dtype=None,
                 **kwargs):
        """
        pgtol: `float`
            Gradient convergence condition.
        m: `int`
            Maximum number of variable metric corrections used to define
            the limited memory matrix.
        maxls: `int`
            Maximum number of line search steps (per iteration).
        factr: `float`
            Convergence condition:
               1e12 (low accuracy),
               1e7 (moderate accuracy),
               10.0 (extremely high accuracy)
        """
        super().__init__(name="lbfgs", displayname="L-BFGS_TF", dtype=dtype)
        self.pgtol = pgtol
        self.m = m
        self.maxls = maxls
        self.factr = factr
        self.evals = 0
        self.last_x = None

    def run(self,
            x0,
            loss_grad,
            epochs=None,
            callback=None,
            epoch_start=0,
            **kwargs):
        self.epoch = epoch_start

        def callback_wrap(x):
            self.epoch += 1
            if callback:
                callback(self.last_x, self.epoch, opt=self)

        def func_wrap(x):
            self.last_x = x
            self.evals += 1
            loss, grad, last_residual = loss_grad(x, self.epoch)
            self.last_loss = loss.numpy()
            self.last_residual = [r.numpy() for r in last_residual]
            return loss, grad

        def stopping_condition(converged, failed):
            callback_wrap(self.last_x)
            if self.epoch > self.epochs:
                return tf.ones(converged.shape, dtype=bool)
            return tfp.optimizer.converged_all(converged, failed)

        import tensorflow_probability as tfp
        self.last_x = x0
        self.epochs = epochs
        res = tfp.optimizer.lbfgs_minimize(
            func_wrap,
            initial_position=x0,
            max_iterations=epochs,
            num_correction_pairs=self.m,
            max_line_search_iterations=self.maxls,
            tolerance=-1,
            x_tolerance=-1,
            f_relative_tolerance=-1,
            stopping_condition=stopping_condition,
        )

        x = res.position
        info = Namespace()
        info.epochs = 0
        info.evals = self.evals
        return x, info


class AdamOptimizer(Optimizer):

    def __init__(self, dtype=None, **kwargs):
        super().__init__(name="adam", displayname="Adam", dtype=dtype)

    def run(self,
            x0,
            loss_grad,
            epochs=None,
            callback=None,
            lr=1e-3,
            epoch_start=0,
            **kwargs):

        class CustomModel(tf.keras.Model):

            def __init__(self, x):
                super().__init__()
                self.x = x
                self.evals = 0
                self.epoch = epoch_start

            def __call__(self):
                self.evals += 1
                return loss_grad(self.x, self.epoch)

            def train_step(self, _):
                loss, grad, last_residual = self()
                self.optimizer.apply_gradients(zip([grad], [self.x]))
                self.last_loss = loss.numpy()
                self.last_residual = [r.numpy() for r in last_residual]
                return {'loss': loss}

        x = tf.Variable(x0)
        model = CustomModel(x)

        class CustomCallback(tf.keras.callbacks.Callback):

            def on_epoch_end(self, epoch, logs=None):
                model.epoch = epoch_start + epoch
                self.last_residual = model.last_residual
                self.last_loss = model.last_loss
                if epoch > 0 and callback:
                    callback(model.x, epoch=epoch, opt=self)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      run_eagerly=True)
        dummy = [1]  # Unused input.
        model.fit(dummy,
                  epochs=epochs + 1,
                  callbacks=[CustomCallback()],
                  verbose=0)
        info = Namespace()
        info.epochs = epochs
        info.evals = model.evals
        return x, info


class GdOptimizer(Optimizer):

    def __init__(self, dtype=None, **kwargs):
        super().__init__(name="gd", displayname="GD", dtype=dtype)

    def run(self,
            x0,
            loss_grad,
            epochs=None,
            callback=None,
            lr=1e-3,
            epoch_start=0,
            **kwargs):

        x = x0 + 0
        for epoch in range(epoch_start + 1, epoch_start + epochs + 1):
            loss, grad, last_residual = loss_grad(x, epoch)
            x -= grad * lr
            self.evals += 1
            self.last_residual = [r.numpy() for r in last_residual]
            self.last_loss = loss.numpy()
            if epoch > 0 and callback is not None:
                callback(x, epoch=epoch, opt=self)

        info = Namespace()
        info.epochs = epochs
        info.evals = self.evals
        return x, info


def make_optimizer(name, dtype=None, **kwargs):
    if name == "lbfgsb":
        optimizer = LbfgsbOptimizer(dtype=dtype, **kwargs)
    elif name == "lbfgs":
        optimizer = LbfgsOptimizer(dtype=dtype, **kwargs)
    elif name == "adam":
        optimizer = AdamOptimizer(dtype=dtype, **kwargs)
    elif name == "gd":
        optimizer = GdOptimizer(dtype=dtype, **kwargs)
    else:
        raise ValueError("Unknown optimizer '{}'".format(name))
    return optimizer
