import sys
import os
import matplotlib

#matplotlib.use("pgf")  # Produces smaller files than Agg
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import inspect
import time
import optimizer
import linsolver
import numpy as np

g_log_file = sys.stderr  # File used by printlog()
g_log_echo = False  # True if printlog() should print to stderr.

geodata_qualitative = [
    "#FF1F5B", "#00CD6C", "#009ADE", "#AF58BA", "#FFC61E", "#F28522",
    "#A0B1BA", "#A6761D", "#E9002D", "#FFAA00", "#00B000"
]

g_colorscheme = geodata_qualitative[:7]

if os.environ.get("NOFONT", False):
    g_fontfamily = "DejaVu Sans"
else:
    g_fontfamily = "Myriad Pro"
plt.rcParams.update({'font.family': g_fontfamily})


def set_log_file(f, echo=False):
    global g_log_file, g_log_echo
    g_log_file = f
    g_log_echo = echo


def printlog(*msg):
    m = ' '.join(map(str, msg)) + '\n'
    if g_log_echo and g_log_file != sys.stderr:
        sys.stderr.write(m)
        sys.stderr.flush()
    g_log_file.write(m)
    g_log_file.flush()


def TIMECLEAR():
    global time_prev
    time_prev = time.time()


def TIME(pattern="{file}:{line}"):
    global time_prev
    if 'time_prev' not in globals():
        time_prev = time.time()
    file = os.path.basename(inspect.stack()[1][1])
    line = inspect.stack()[1][2]
    t = "{:.3f}s".format(time.time() - time_prev)
    d = {'file': file, 'line': line}
    printlog((pattern + ' ' + t).format(**d))
    time_prev = time.time()


class Timer():
    def __init__(self):
        self._starts = []  # Stack of pairs (key, time).
        self.counters = dict()  # Key to time.

    def push(self, key=None):
        self._starts.append((key, time.time()))

    def pop(self, key=None):
        start = self._starts.pop()
        assert start[0] is None or key is None or start[0] == key, \
                "Inconsistent keys passed to push() and pop(): "\
                "{:} and {:}".format(start[0], key)
        if key is None:
            key = start[0]
        self.counters[key] = self.counters.get(key, 0.) + time.time() - start[1]

    def append(self, timer):
        for k in timer.counters:
            self.counters[k] = self.counters.get(k, 0.) + timer.counters[k]


def get_error(u, v):
    e1 = np.mean(abs(u - v))
    e2 = np.mean((u - v)**2)**0.5
    einf = np.max(abs(u - v))
    return e1, e2, einf


def plot(domain,
         exact_u,
         pred_u,
         path,
         cellsel=None,
         title=None,
         umin=None,
         umax=None,
         slice_lim=0.1,
         transpose=False,
         nslices=6,
         dpi=300,
         x_imp=None,
         y_imp=None,
         marker_size=1,
         marker_color='k',
         transparent=True,
         figsize=(3, 2.5),
         cmap=None):
    if transpose:
        # Index zero drawn as vertical.
        ix = 1
        iy = 0
        exact_u = exact_u.T
        pred_u = pred_u.T
    else:
        # Index zero drawn as horizontal.
        ix = 0
        iy = 1
    extent = [
        domain.lower[ix], domain.upper[ix], domain.lower[iy], domain.upper[iy]
    ]
    cred = g_colorscheme[0]
    cblue = g_colorscheme[2]
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0, wspace=0)
    spec = fig.add_gridspec(2 * nslices, 3)
    x = domain.cell_center_1d(ix)
    y = domain.cell_center_1d(iy)
    dx = domain.step_by_dim(ix)
    dy = domain.step_by_dim(iy)
    xlim = (domain.lower[ix], domain.upper[ix])
    if umin is None:
        umin = exact_u.min()
    if umax is None:
        umax = exact_u.max()
    if cmap is None:
        cmap = 'viridis'
    ulim = (umin, umax)
    ptp = umax - umin
    slim = (umin - ptp * slice_lim, umax + ptp * slice_lim)
    if title is not None:
        fig.suptitle(title, fontsize=8)
    axes = [None, None]
    for data, i in (pred_u, 0), (exact_u, 1):
        axes[i] = fig.add_subplot(spec[1:-1, i])
        ax = axes[i]
        ax.imshow(data.T,
                  interpolation='nearest',
                  cmap=cmap,
                  vmin=ulim[0],
                  vmax=ulim[1],
                  extent=extent,
                  origin='lower',
                  aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
    if cellsel is not None:
        ax = axes[1]
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (y[cellsel.y0] - 0.5 * dy, x[cellsel.x0] - 0.5 * dx),
                y[cellsel.y1 - 1] - y[cellsel.y0] + dy,
                x[cellsel.x1 - 1] - x[cellsel.x0] + dy,
                fc='none',
                color='black',
                linewidth=1,
                linestyle="solid"))

    if x_imp is not None:
        if type(x_imp) != list:
            x_imp = [x_imp]
            y_imp = [y_imp]
            marker_color = [marker_color]
        ax = axes[1]
        for xi, yi, mc in zip(x_imp, y_imp, marker_color):
            ax.scatter(xi,
                       yi,
                       s=marker_size,
                       alpha=1,
                       edgecolor='none',
                       facecolor=mc,
                       zorder=100)

    Ny = pred_u.shape[1]
    shift = 0.22
    spec = fig.add_gridspec(2 * nslices, 3, left=shift)
    for i in range(nslices):
        yslice = i * (Ny - 1) // max(1, nslices - 1)
        ns = nslices - 1 - i
        ax = fig.add_subplot(spec[2 * ns:2 * ns + 2, 2])
        l0, = ax.plot(x,
                      exact_u[:, yslice],
                      c=cblue,
                      ls='-',
                      label="reference",
                      linewidth=0.9)
        l1, = ax.plot(x,
                      pred_u[:, yslice],
                      c=cred,
                      ls='-',
                      label="inferred",
                      linewidth=0.6)
        # ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(slim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.arrow(-0.025,
                 0.5,
                 -0.05,
                 0,
                 overhang=0,
                 head_width=0.05,
                 head_length=0.03,
                 linewidth=0.5,
                 transform=ax.transAxes,
                 facecolor='black',
                 clip_on=False)
    leg0 = plt.legend(handles=[l0],
                      loc=(-1.0 - shift, 0.5),
                      frameon=False,
                      handletextpad=0.5,
                      fontsize=7)
    leg1 = plt.legend(handles=[l1],
                      loc=(-2.15 - shift, 0.5),
                      frameon=False,
                      handletextpad=0.5,
                      fontsize=7)
    ax.add_artist(leg0)
    ax.add_artist(leg1)

    fig.savefig(path,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.01,
                transparent=transparent)
    plt.close(fig)


def plot_2d(
    domain,
    exact_uu,
    pred_uu,
    slices_it,
    path,
    title=None,
    umin=None,
    umax=None,
    dpi=300,
    figsizey=3.,
    hspace=0.05,
    cmap=None,
    callback=None,
    xlabel='{:.2f}',
    ylabel_exact='reference',
    ylabel_pred='inferred',
    transparent=False,
):
    '''
    exact_uu: `array`
        3D fields (t, x, y) with reference solution.
    pred_uu: `array`
        3D fields (t, x, y) with inferred solution.
    slices_it: `list`
        List of indices along direction t
    '''
    nslices = len(slices_it)
    figsize = (figsizey * nslices * 0.5, figsizey)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=hspace, wspace=hspace)
    spec = fig.add_gridspec(2, nslices)
    extent = [
        domain.lower[1], domain.upper[1], domain.lower[2], domain.upper[2]
    ]
    tt = domain.cell_center_1d(0)
    axes = [[None] * nslices for i in range(2)]
    for j in range(nslices):
        it = slices_it[j]
        for i, data in enumerate((exact_uu[it], pred_uu[it])):
            ax = fig.add_subplot(spec[i, j])
            axes[i][j] = ax
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(data.T,
                      interpolation='nearest',
                      cmap=cmap,
                      vmin=umin,
                      vmax=umax,
                      extent=extent,
                      origin='lower',
                      aspect='equal')
            if i == 1:
                if xlabel:
                    ax.set_xlabel(xlabel.format(tt[it]))
            if j == 0 and i == 0:
                if ylabel_exact:
                    ax.set_ylabel(ylabel_exact)
            if j == 0 and i == 1:
                if ylabel_pred:
                    ax.set_ylabel(ylabel_pred)

            if callback:
                callback(i, j, ax, fig)

    fig.savefig(path,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.01,
                transparent=transparent)
    plt.close(fig)

def add_arguments(parser):
    parser.add_argument('--epochs',
                        type=int,
                        default=None,
                        help="Maximum epochs, "
                        "defaults to product of plot_every and frames")
    parser.add_argument('--every_factor',
                        type=int,
                        default=1,
                        help="Multiplier for all *_every options")
    parser.add_argument('--plot_every',
                        type=int,
                        default=5,
                        help="Epochs between plots")
    parser.add_argument('--report_every',
                        type=int,
                        default=10,
                        help="Epochs between reports to stdout")
    parser.add_argument('--history_every',
                        type=int,
                        default=1,
                        help="Epochs between entries of training history")
    parser.add_argument('--checkpoint_every',
                        type=int,
                        default=5,
                        help="Epochs between checkpoints")
    parser.add_argument('--frames',
                        type=int,
                        default=10,
                        help="Frames to plot. Zero disables first frame.")
    parser.add_argument('--outdir',
                        type=str,
                        default='.',
                        help='Output directory')
    parser.add_argument('--optimizer',
                        type=str,
                        default='newton',
                        help="Optimizer")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="Seed for numpy.random and tensorflow.random")
    parser.add_argument('--plot_title',
                        type=int,
                        default=0,
                        help="Enable title in plots")
    parser.add_argument('--history_full',
                        type=int,
                        default=0,
                        help="Number of epochs to write "
                        "history at every point")
    parser.add_argument('--montage',
                        type=int,
                        default=0,
                        help="Run montage after plotting")

def optimize_newton(args, problem, state, callback):
    opt = optimizer.Optimizer(name='newton', displayname='Newton')
    opt.evals = 0
    printlog("Running Newton optimizer")
    packed = problem.pack_state(state)
    dhistory = dict()
    for epoch in range(args.epochs + 1):
        s = problem.unpack_state(packed)
        problem.domain.assign_active_state(state, s)
        callback(packed, epoch, dhistory=dhistory, opt=opt)
        if epoch == args.epochs:
            break
        const, m = problem.linearize(state, epoch=epoch)
        opt.evals += 1
        const = np.array(const).flatten()
        problem.timer_total.push('tt_linsolver')
        timer = Timer()
        timer.push('linsolver')
        dpacked = linsolver.solve(m, -const, args, dhistory, args.linsolver)
        timer.pop()
        problem.timer_total.append(timer)
        problem.timer_last.append(timer)
        packed += dpacked


def optimize_opt(args, name, problem, state, callback, **kwargs):
    def func(packed, epoch):
        s = problem.unpack_state(packed)
        problem.domain.assign_active_state(state, s)
        loss, grads, wgrads = problem.eval_loss_grad(state, epoch=epoch)
        g = problem.pack_fields(grads)
        wg = problem.pack_weights(wgrads)
        g = np.hstack([g, wg])
        return loss, g

    packed = problem.pack_state(state)
    opt = optimizer.make_optimizer(name, func, **kwargs)
    printlog("Running {:} optimizer".format(opt.displayname))
    callback(packed, epoch=0, opt=opt)
    packed, info = opt.run(packed,
                           func,
                           epochs=args.epochs,
                           callback=callback,
                           lr=args.lr,
                           **kwargs)
    printlog(info)

g_colormap_names = [
    "rainbow",
    "coolwarm",
    "yellow",
    "geo",
]

def get_colormap_data(name):
    data = None
    assert name in g_colormap_names
    if name == "rainbow":
        data = [
            0.0, 0.278431372549, 0.278431372549, 0.858823529412, 0.143, 0.0,
            0.0, 0.360784313725, 0.285, 0.0, 1.0, 1.0, 0.429, 0.0,
            0.501960784314, 0.0, 0.571, 1.0, 1.0, 0.0, 0.714, 1.0,
            0.380392156863, 0.0, 0.857, 0.419607843137, 0.0, 0.0, 1.0,
            0.878431372549, 0.301960784314, 0.301960784314
        ]
    elif name == "coolwarm":
        data = [
            0.0, 0.0, 0.0, 0.34902, 0.03125000000000003, 0.039216, 0.062745,
            0.380392, 0.06250000000000006, 0.062745, 0.117647, 0.411765,
            0.09374999999999994, 0.090196, 0.184314, 0.45098,
            0.12499999999999997, 0.12549, 0.262745, 0.501961, 0.15625,
            0.160784, 0.337255, 0.541176, 0.18750000000000003, 0.2, 0.396078,
            0.568627, 0.21875000000000006, 0.239216, 0.454902, 0.6,
            0.24999999999999994, 0.286275, 0.521569, 0.65098,
            0.28124999999999994, 0.337255, 0.592157, 0.701961, 0.3125,
            0.388235, 0.654902, 0.74902, 0.34375, 0.466667, 0.737255, 0.819608,
            0.37500000000000006, 0.572549, 0.819608, 0.878431, 0.40625,
            0.654902, 0.866667, 0.909804, 0.4375, 0.752941, 0.917647, 0.941176,
            0.46875, 0.823529, 0.956863, 0.968627, 0.5, 0.988235, 0.960784,
            0.901961, 0.5, 0.941176, 0.984314, 0.988235, 0.52, 0.988235,
            0.945098, 0.85098, 0.54, 0.980392, 0.898039, 0.784314, 0.5625,
            0.968627, 0.835294, 0.698039, 0.59375, 0.94902, 0.733333, 0.588235,
            0.625, 0.929412, 0.65098, 0.509804, 0.65625, 0.909804, 0.564706,
            0.435294, 0.6875, 0.878431, 0.458824, 0.352941, 0.71875, 0.839216,
            0.388235, 0.286275, 0.7500000000000001, 0.760784, 0.294118,
            0.211765, 0.78125, 0.701961, 0.211765, 0.168627, 0.8125, 0.65098,
            0.156863, 0.129412, 0.84375, 0.6, 0.094118, 0.094118, 0.875,
            0.54902, 0.066667, 0.098039, 0.9062500000000001, 0.501961, 0.05098,
            0.12549, 0.9375, 0.45098, 0.054902, 0.172549, 0.96875, 0.4,
            0.054902, 0.192157, 1.0, 0.34902, 0.070588, 0.211765
        ]
    elif name == "geo":
        data = [
            0.0, 0.0, 0.6039215686274509, 0.8705882352941177, 0.5, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.12156862745098039, 0.3568627450980392
        ]
    elif name == "yellow":
        data = [
            0.0, 1.0, 1.0, 0.988235, 0.002, 1.0, 1.0, 0.988235,
            0.05000000000000001, 0.984314, 0.988235, 0.843137,
            0.10000000000000002, 0.988235, 0.988235, 0.741176, 0.15, 0.980392,
            0.968627, 0.654902, 0.20000000000000004, 0.980392, 0.945098,
            0.576471, 0.25, 0.968627, 0.905882, 0.486275, 0.3, 0.968627,
            0.862745, 0.388235, 0.3499999999999999, 0.960784, 0.803922,
            0.286275, 0.4000000000000001, 0.94902, 0.741176, 0.219608, 0.45,
            0.941176, 0.678431, 0.14902, 0.5, 0.929412, 0.607843, 0.094118,
            0.55, 0.921569, 0.545098, 0.054902, 0.6, 0.909804, 0.486275,
            0.035294, 0.65, 0.890196, 0.411765, 0.019608, 0.6999999999999998,
            0.8, 0.305882, 0.0, 0.7500000000000001, 0.760784, 0.239216, 0.0,
            0.8000000000000002, 0.678431, 0.180392, 0.011765, 0.85, 0.6,
            0.121569, 0.023529, 0.9, 0.501961, 0.054902, 0.031373, 0.95, 0.4,
            0.039216, 0.058824, 1.0, 0.301961, 0.047059, 0.090196
        ]
    else:
        assert False, "Unknown colormap=" + name
    data = np.reshape(data, (-1, 4))
    return data


def get_cmap(name):
    data = get_colormap_data(name)
    nodes = data[:, 0]
    colors = data[:, 1:]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        name, list(zip(nodes, colors)))
    return cmap
