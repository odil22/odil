import numpy as np
import scipy.optimize
from argparse import Namespace
from tfwrap import tf


class Optimizer:
    def __init__(self, name=None, displayname=None, dtype=np.float64):
        self.name = name
        self.displayname = displayname if displayname is not None else name
        self.dtype = dtype

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
                 dtype=np.float64,
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
            epochs=100,
            callback=None,
            epoch_start=0,
            **kwargs):
        self.epoch = epoch_start

        def callback_wrap(x):
            self.epoch += 1
            if callback:
                callback(x, self.epoch, opt=self, loss_grad=loss_grad)

        def func_wrap(x):
            self.evals += 1
            return loss_grad(x, self.epoch)

        x, f, sinfo = scipy.optimize.fmin_l_bfgs_b(func=func_wrap,
                                                   x0=x0,
                                                   maxiter=epochs + 1,
                                                   pgtol=self.pgtol,
                                                   m=self.m,
                                                   maxls=self.maxls,
                                                   factr=self.factr,
                                                   maxfun=1000000,
                                                   callback=callback_wrap)
        info = Namespace()
        info.epochs = sinfo['nit']
        info.evals = sinfo['funcalls']
        info.task = sinfo['task']
        return x, info


class AdamOptimizer(Optimizer):
    def __init__(self, dtype=np.float64, **kwargs):
        super().__init__(name="adam", displayname="Adam", dtype=dtype)

    def run(self,
            x0,
            loss_grad,
            epochs=100,
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
                return loss_grad(self.x.numpy(), self.epoch)

            def train_step(self, _):
                loss, grad = self()
                self.optimizer.apply_gradients(zip([grad], [self.x]))
                return {'loss': loss}

        x = tf.Variable(x0)
        model = CustomModel(x)

        class CustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                model.epoch = epoch_start + epoch
                if callback:
                    callback(model.x.numpy(),
                             epoch,
                             opt=self,
                             loss_grad=loss_grad)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      run_eagerly=True)
        dummy = [1]
        model.fit(dummy,
                  epochs=epochs + 1,
                  callbacks=[CustomCallback()],
                  verbose=0)
        x = x.numpy()
        info = Namespace()
        info.epochs = epochs
        info.evals = model.evals
        return x, info


def make_optimizer(name, loss_grad, dtype=np.float64, **kwargs):
    if name == "lbfgsb":
        optimizer = LbfgsbOptimizer(dtype=dtype, **kwargs)
    elif name == "adam":
        optimizer = AdamOptimizer(dtype=dtype, **kwargs)
    else:
        raise ValueError("Unknown optimizer '{}'".format(name))
    return optimizer
