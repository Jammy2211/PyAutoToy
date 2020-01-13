#!/usr/bin/env python
# coding: utf-8

# # Time series Toy Model
# 
# ## Competitive Lotka-Volterra equations
# 
# We are going to define a simple time series model based on the [competive Lotka-Volterra equations](https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations),
# 
# $$
# {\frac  {d x_{i}}{dt}} = 
# r_{i}x_{i} \left(1 - {\frac{\sum _{{j=1}}^{N} A _{{ij}}x_{j}}{K}}\right)
# $$,
# 
# where $x_i$ is the number of species $i$ present, $r_i$ is the inherent growth rate, $K$ is the carrying capacity, $A_{ij}$ is the interaction matrix between the different species.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import odeint


def lotka_volterra(x, t, r, A, K):
    return x * r * (1 - A.dot(x) / K)


# A classic instance of the equations are shown below, demonstrating chaotic behaviour

# In[ ]:


np.random.seed(1)

colors = plt.rcParams["axes.prop_cycle"].by_key()['color']
species_colors = colors[:4]

r = np.array([1, 0.72, 1.53, 1.27])
A = np.array([
    [1., 1.09, 1.52, 0.],
    [0., 1., 0.44, 1.36],
    [2.33, 0., 1., 0.47],
    [1.21, 0.51, 0.35, 1.]])
K = 1e6

x0 = (np.random.exponential(0.1, 4) + 0.005) * K
ts = np.linspace(0, 100, 2 ** 10)
xt = odeint(lotka_volterra, x0, ts, args=(r, A, K))

for i, color in enumerate(species_colors):
    plt.plot(ts, xt[:, i], color=color)

# ## Stochastic  model

# We can further generalise these Lotka-Volterra equations to incorporate a stochastic element, where we split the equation above into a [birth-death process](https://en.wikipedia.org/wiki/Birth%E2%80%93death_process), where we define the growth rate of the population,
# $$ 
# \lambda_{x_i \rightarrow x_i + 1} = r_{i}x_{i},
# $$
# and death rate,
# $$ 
# \lambda_{x_i \rightarrow x_i - 1} = 
# r_{i}x_{i}\frac{\sum _{{j=1}}^{N} A _{{ij}}x_{j}}{K}.
# $$
# These birth-death processes can be simulated in a variety ways, in this case we will use the [tau-leaping algorithm](https://en.wikipedia.org/wiki/Tau-leaping) which assumes that over a short time step, $\Delta t$, the magnitude of the populations $x_i$ does not change significantly, so the number of birth and death processes during that period can be modelled as as Poisson process,
# 
# $$
# \Delta_{x_i +} \sim \text{Pois}(\lambda_{x_i \rightarrow x_i + 1} \Delta t),$$
# $$
# \Delta_{x_i -} \sim \text{Pois}(\lambda_{x_i \rightarrow x_i - 1} \Delta t),$$
# $$
# \Delta_{x_i} = \Delta_{x_i +} - \Delta_{x_i -},
# $$

# In[ ]:


from numpy.random import poisson


def LV_birthdeath(x, t, r, A, K):
    xr = x * r
    return xr, xr * A.dot(x) / K


def tau_leaping(func, x0, ts, args=()):
    """
    Simulate a birth death process using the tau leaping algorithm
    
    Parameters
    ----------
    func: callable(x, ts, ...)
        Computes the birth and death rate of the process, must return a 2-tuple
        where the 1st element is the birth rate and the 2nd element is the death rate
        
    x0: array
        The initial value of x
        
    t: array
        A sequence of time points for which to solve for y. The initial value point 
        should be the first element of this sequence. This sequence must be monotonically 
        increasing or monotonically decreasing; repeated values are allowed.

    args: tuple, optional
        Extra arguments to pass to function.
    """
    xs = np.zeros(np.shape(ts) + np.shape(x0))
    xs[0, :] = x0
    delta_t = np.diff(ts)
    for i, (t, dt) in enumerate(zip(ts[1:], delta_t)):
        x = xs[i]
        birthrate, deathrate = func(x, t, *args)
        deltax = poisson(birthrate * dt) - poisson(deathrate * dt)
        xs[i + 1] = x + deltax

    return xs


# In[ ]:


n_repeats = 30
xs_stoch = np.empty((n_repeats,) + ts.shape + x0.shape)

for i in range(n_repeats):
    x0_stoch = np.random.poisson(x0)
    xs_stoch[i] = tau_leaping(LV_birthdeath, x0_stoch, ts, args=(r, A, K))

xt_lo, xt_med, xt_hi = np.quantile(xs_stoch, [0.16, 0.5, 0.84], axis=0)

for i, color in enumerate(species_colors):
    plt.plot(ts, xt[:, i], color=color)
    plt.plot(ts, xt_med[:, i], ls=':', color=color)
    plt.fill_between(ts, xt_lo[:, i], xt_hi[:, i], color=color, alpha=0.5)

plt.ylim(0, xs_stoch.max() * 1.05)
plt.xlim(ts[0], ts[-1]);


# ## Central limit approximation 
# 
# Performing inference on a birth-death process will be difficult as it is not possible to calculate the probability of observing a given state for a set of initial conditions except for when the maximum population size is very small, as the probabilistic master equation must be integrated, which has an $n_{species}^{\max(\sum x_i)}$ components.
# 
# Likelihood free methods such as [approximate Bayesian computation (ABC)](https://en.wikipedia.org/wiki/Approximate_Bayesian_computation) could be used, however these methods are computationally expensive.
# 
# The most computational efficient approach is to consider the ODE solution, 
# $\hat{\mathbf{x}}(t) = (\hat{x}_1(t), ..., \hat{x}_n(t))$, to the process,
# 
# $$K \frac{d \hat{x}_i(t)}{d t} = \lambda_{x_i \rightarrow x_i + 1} - \lambda_{x_i \rightarrow x_i - 1}$$,
# 
# $$\frac{d \hat{x}_i(t)}{d t} = f(\hat{x}_i(t))$$
# 
# where $\mathbf{f} = (f_1, ..., f_n)$ is the rate function of the ODE. In general there can be multiple types of birth-death process leading to larger or smaller  jumps.
# 
# $$ K \frac{d \hat{x}_i(t)}{d t} = \sum_{v} v \lambda_{x_i \rightarrow x_i + v}.$$
# 
# 
# This approach neglects stochastic fluctuations, and also cannot directly calculate the probability of observing a given state at a specific time.
# 
# In this section we consider the [central limit approximation (CLA)](http://arxiv.org/abs/1804.08744) which models the distribution over number of species present at time $t_0$ as a normal distribution,
# 
# $$ x_i(t) \approx \tilde{x}_i(t) \sim K \hat{x}_i(t) + K^{1/2} g_i(t),$$
# 
# $$ \mathbf{G}(t_0) = \{g_1(t_0), ..., g_n(t_0)\} \sim \mathcal{N}(\mathbf{m}(t_0), \mathbf{\Sigma}(t_0)),$$
# 
# which in turn makes $\mathbf{G}$ a Gaussian process, 
# 
# $$ \mathbf{g}(t) = \{g_1(t), ..., g_n(t)\} \sim \mathcal{GP}(\mathbf{m}(t), \mathbf{\Sigma}(t)),$$
# 
# The probability distribution of $\mathbf{g}(t)$ is given by by the solution to the following differential equations,
# 
# $$\frac{d \mathbf{m}(t)}{d t} = J_f(t) \cdot \mathbf{m}(t)
# $$
# 
# $$\frac{d \mathbf{\Sigma}(t)}{d t} =  
# J_f(t) \cdot \mathbf{\Sigma}(t) + \mathbf{\Sigma}(t) \cdot J^\top_f(t) + W(t)
# $$
# where 
# $$J_f(t) = 
# \left. \nabla_{\hat{\mathbf{x}}} \mathbf{f} \right|_{\hat{\mathbf{x}} = \hat{\mathbf{x}}(t)}$$
# is the Jacobian of the ODE rate function, and
# $$W(t)_{i, i} = 
# \left(\lambda_{x_i \rightarrow x_i + 1} + \lambda_{x_i \rightarrow x_i - 1}\right) \frac{\delta_{ij}}{K}.$$

# In[ ]:


def CLAint(func, x0, cov0, ts, args=()):
    """
    Simulate a birth death process using the tau leaping algorithm
    
    Parameters
    ----------
    func: callable(x, ts, ...)
        Computes the rate function and the Jacobian of the rate function
        must return a tuple containing the value of the rate function as its first value
        and the Jacobian as the second value
        
    x0: array
        The initial value of x
        
    cov0: array
        The initial covariance of x
        
    t: array
        A sequence of time points for which to solve for y. The initial value point 
        should be the first element of this sequence. This sequence must be monotonically 
        increasing or monotonically decreasing; repeated values are allowed.

    args: tuple, optional
        Extra arguments to pass to function.
    """
    xs = np.empty(np.shape(ts) + np.shape(x0))
    xs[0, :] = x0
    if np.ndim(cov0) == 1:
        cov0 = np.diag(cov0)

    covs = np.empty(np.shape(ts) + np.shape(cov0))
    covs[0, :, :] = cov0

    delta_t = np.diff(ts)
    for i, (t, dt) in enumerate(zip(ts[1:], delta_t)):
        x = xs[i]
        cov = covs[i]
        f, jac, W = func(x, t, *args)
        xs[i + 1, :] = x + f * dt
        covs[i + 1, :, :] = cov + jac.dot(cov) + cov.dot(jac.T) + W

    return xs, covs


# ### Lotka - Volterra CLA model
# 
# Defining the Lotka-Volterra equations in this formalism we find,
# 
# $$
# f_i(\hat{\mathbf{x}}) = 
# {r_{i}\hat{x}_{i}} \left(1 - {{\sum _{{j=1}}^{N} A _{{ij}}\hat{x}_{j}}}\right),
# $$
# with Jacobian,
# $$
# J_f(\hat{\mathbf{x}})_{i, j} = 
# \frac{r_{i}}{K}\left(1 - {{\sum _{{k=1}}^{N} A _{{ik}}\hat{x}_{k}}} \right)  \delta_{i, j}
# - \frac{r_{i} \hat{x}_i A_{{i, j}}}{K},
# $$
# and 
# $$
# W(\hat{\mathbf{x}})_{i, j} = \frac{r_{i}\hat{x}_{i}}{K}
# \left(1 + {{\sum _{{j=1}}^{N} A _{{ij}}\hat{x}_{j}}}\right).
# $$

# In[ ]:


def LV_CLA(x, t, r, A, K):
    rx = x * r
    Ax = A.dot(x)
    d = r * (1 - Ax)
    f = d * x
    jac = - rx[:, None] * A
    jac[np.diag_indices(4)] += d
    W = rx * (1 + Ax)
    return f, jac / K, W / K


# In[ ]:


cov0 = np.diag(x0 * K ** -2)
x_cla, cov_cla = CLAint(LV_CLA, x0 / K, cov0, ts, args=(r, A, K))
x_cla_std = cov_cla[(slice(None),) + np.diag_indices(4)] ** 0.5

for i, color in enumerate(species_colors):
    plt.plot(ts, x_cla[:, i], ls=':', color=color)
    plt.fill_between(ts, x_cla[:, i] - x_cla_std[:, i],
                     x_cla[:, i] + x_cla_std[:, i], color=color, alpha=0.5)

# In[ ]:


for i, color in enumerate(species_colors):
    plt.plot(ts, xt_med[:, i], ls=':', color=color)
    plt.fill_between(ts, xt_lo[:, i], xt_hi[:, i], color=color, alpha=0.5)

plt.ylim(0, xs_stoch.max() * 1.05)
plt.xlim(ts[0], ts[-1]);

# # Hierarchical Model
# 
# Frequently we do not have direct access to the size of the populations and when we sample from the population we do not even from which species our sample came from. Instead we observe the properties of a set of individuals sampled from the distribution.

# In[ ]:


from matplotlib import animation
import matplotlib.patches as patches
import matplotlib.path as path
from IPython.display import HTML


class Histogram(object):
    """
    updateable histogram for matplotlib animations
    """

    def __init__(self, a, bins=10, quantiles=None, range=None, density=None, weights=None,
                 plot=True, **kwargs):
        self.patch = None
        self.bins = 10

        self._hist_kwargs = dict(range=range, density=density, weights=weights)
        self.hist, self.bins = self.histogram(a, bins=bins, quantiles=quantiles)
        self.ax = None

        self._initialize_verts()
        if plot:
            self.plot(**kwargs)

    def update_lims(self):
        ax = self.ax
        xlim = ax.get_xlim()
        ax.set_xlim(min(xlim[0], self.bins[0]), max(xlim[1], self.bins[-1]))
        ylim = ax.get_ylim()
        ax.set_ylim(min(xlim[0], 0), max(ylim[1], self.hist.max()))

    def plot(self, ax=None, **kwargs):
        patch = self._make_patch(**kwargs)
        if ax is None:
            ax = plt.gca()

        ax.add_patch(patch)
        self.ax = ax
        self.update_lims()
        return [patch]

    def histogram(self, a, bins=None, quantiles=None, **kwargs):
        if bins is None:
            bins = self.bins

        if quantiles is not None:
            if isinstance(quantiles, int):
                bins = np.quantile(a, np.linspace(0, 1, quantiles))
            else:
                bins = np.quantile(a, quantiles)

        kwargs.update(self._hist_kwargs)
        return np.histogram(a, bins=bins, **kwargs)

    def update_data(self, a, update_lims=True, **kwargs):
        self.hist, _ = self.histogram(a, **kwargs)
        self._set_verts()
        if update_lims:
            self.update_lims()
        return [self.patch]

    def _initialize_verts(self):
        # get the corners of the rectangles for the histogram
        nrects = self.bins.size - 1
        nverts = nrects * (1 + 3 + 1)
        verts = np.zeros((nverts, 2))
        codes = np.ones(nverts, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY

        self.codes = codes
        self.verts = verts
        self._set_verts()

    def _set_verts(self, top=None, bottom=None, left=None, right=None):
        if top is None:
            top = self.hist
        if left is None:
            left = self.bins[:-1]
        if right is None:
            right = self.bins[1:]
        if bottom is None:
            bottom = np.zeros(len(left))

        verts = self.verts
        verts[0::5, 0] = left
        verts[0::5, 1] = bottom
        verts[1::5, 0] = left
        verts[1::5, 1] = top
        verts[2::5, 0] = right
        verts[2::5, 1] = top
        verts[3::5, 0] = right
        verts[3::5, 1] = bottom

    def _make_patch(self, **kwargs):
        self.patch = patches.PathPatch(path.Path(self.verts, self.codes), **kwargs)
        return self.patch


# Let's assume that we can measure 3 properties associated with each species. In this case we assume that the properties of each species are distributed normally,
# 
# $$y_{i,j} \sim \mathcal{N}(\mu_{i, j}, \sigma_{i, j}),$$
# 
# see below,

# In[ ]:


np.random.seed(1)

n_prop = 3
y_mean = stats.uniform.rvs(size=(4, n_prop)) * 2 - 1
y_std = stats.invgamma.rvs(70., size=(4, n_prop)) ** 0.5
y_dists = [stats.norm(loc=mu, scale=std)
           for mu, std in zip(y_mean, y_std)]

f, axes = plt.subplots(3, sharex=True)

ys = np.linspace(-2, 2, 1000)
pdf = np.zeros(ys.shape + (n_prop,))
for dist, color in zip(y_dists, species_colors):
    species_pdf = dist.pdf(ys[:, None])
    pdf += species_pdf
    for i, ax in enumerate(axes):
        ax.plot(ys, species_pdf[:, i], color=color, zorder=10)

for i, ax in enumerate(axes):
    ax.plot(ys, pdf[:, i], color='k', ls=':', zorder=-10)

# When we sample from the population we only observe the relative proportions of the species

# In[ ]:


rel_fracs = xt / xt.sum(1, keepdims=True)

plt.plot(ts, rel_fracs)
plt.ylabel("relative proportions");

# So if we consider time point t=50 and draw a sample from that time point we can observe the distribution of property values from our sample,

# In[ ]:


fig, axes = plt.subplots(3, 3, figsize=(6, 6), sharex=True)

lim = (-1.5, 1.5)

n_samples = 10000

rel_frac = rel_fracs[ts.searchsorted(50)]

n_species_sampled = np.random.multinomial(n_samples, rel_frac)
sample = np.concatenate(
    [dist.rvs(size=(n, n_prop)) for dist, n in zip(y_dists, n_species_sampled)])

hist2d_kws = dict(bins=30, range=[lim] * 2, cmap='Purples')
for i in range(n_prop):
    Histogram(sample[:, i], bins=30, density=True,
              color=colors[4], alpha=0.5, ax=axes[i, i])
    axes[i, i].set_xlim(*lim)
    axes[i, i].set_ylim(0, 2)

    axes[-1, i].set_xlabel("property {:d}".format(i))
    axes[i, 0].set_ylabel("property {:d}".format(i))

    for j in range(i):
        Z, x, y = np.histogram2d(sample[:, j], sample[:, i], bins=30, range=[lim] * 2)
        X, Y = np.meshgrid(x[:-1], y[:-1])
        axes[j, i].contour(X, Y, Z)  # , cmap='Purples')
        axes[i, j].scatter(sample[:, j], sample[:, i],
                           marker='+', color=colors[6], alpha=0.1)
        axes[i, j].set_xlim(*lim)
        axes[i, j].set_ylim(*lim)

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
fig.subplots_adjust(hspace=0., wspace=0.)

# In[ ]:


n_samples = 5000
bins = np.linspace(-1.5, 1.5, 100)

fig, axes = plt.subplots(3, sharex=True)

ys = bins
pdf = np.zeros(ys.shape + (n_prop,))

lines = [[] for _ in y_dists]
totals = []

for dist, f, color, ls in zip(y_dists, rel_frac, species_colors, lines):
    species_pdf = f * dist.pdf(ys[:, None])
    pdf += species_pdf
    for i, ax in enumerate(axes):
        ls.extend(ax.plot(ys, species_pdf[:, i], color=color, zorder=10))

for i, ax in enumerate(axes):
    totals.extend(ax.plot(ys, pdf[:, i], color='k', ls=':', zorder=-10))

histograms = [
    Histogram(sample[:, i], bins=bins, density=True,
              color=colors[4], alpha=0.5, ax=ax) for i, ax in enumerate(axes)]

for i, ax in enumerate(axes):
    ax.set_ylim(0, 2)
    ax.set_ylabel("property {:d}".format(i))


# We can see how this changes in time,

# In[ ]:


def animate(t):
    i = ts.searchsorted(t)
    rel_frac = rel_fracs[i]

    pdf = np.zeros(ys.shape + (n_prop,))
    for dist, f, color, ls in zip(y_dists, rel_frac, species_colors, lines):
        species_pdf = f * dist.pdf(ys[:, None])
        pdf += species_pdf
        for i, ax in enumerate(axes):
            ls[i].set_data(ys, species_pdf[:, i])

    for i, ax in enumerate(axes):
        totals[i].set_data(ys, pdf[:, i])

    n_species_sampled = np.random.multinomial(n_samples, rel_frac)
    sample = np.concatenate(
        [dist.rvs(size=(n, n_prop)) for dist, n in zip(y_dists, n_species_sampled)])

    artists = []
    for i, hist in enumerate(histograms):
        artists.extend(hist.update_data(sample[:, i]))

    for ax in axes:
        ax.set_ylim(0, 2)

    return artists


ani = animation.FuncAnimation(
    fig, animate, np.linspace(ts[0], ts[-1], 100), repeat=False, blit=True)
HTML(ani.to_html5_video())


# # Hierarchical model

# In[ ]:


class LotkaVolterra(object):

    def __init__(self, r, A, K, dists):
        self.r = r
        self.A = A
        self.K = K
        self.args = (r, A, K)
        self.dists = dists
        self.prop_shape = self.dists[0].mean().shape

    def odeint(self, x0, ts):
        return odeint(lotka_volterra, x0, ts, args=self.args)

    def tau_leaping(self, x0, ts):
        return tau_leaping(LV_birthdeath, x0, ts, args=self.args)

    def CLAint(self, x0, ts, cov0=None):
        if cov0 is None:
            cov0 = np.zeros_like(x0)

        K = self.K
        x_hat, covs = CLAint(LV_CLA, x0 / K, cov0, ts, args=self.args)
        return x_hat * K, covs * K ** 2

    def sample_properties(self, xs, nsamples):
        rel_fracs = xs / xs.sum(1, keepdims=True)

        samples = np.empty((len(xs), nsamples,) + self.prop_shape)
        for i, rel_frac in enumerate(rel_fracs):
            samples[i] = np.concatenate(
                [dist.rvs(size=(n,) + self.prop_shape) for dist, n in
                 zip(self.dists, np.random.multinomial(n_samples, rel_frac))])

        return samples


# In[ ]:


# lv = LotkaVolterra(r, A, K, y_dists)
#
# x_tau = lv.tau_leaping(x0, ts)
# plt.plot(ts, x_tau)
#
# # In[ ]:
#
#
# t_sample = np.linspace(10, 100, 5)
# samples = lv.sample_properties(xs[ts.searchsorted(t_sample)], nsamples=5000)
#
# f, axes = plt.subplots(5, figsize=(8, 10), sharex=True)
#
# for i, ax in enumerate(axes):
#     for j in range(n_prop):
#         Histogram(samples[i, :, j], bins=100, density=True, ax=ax,
#                   facecolor=colors[4 + j], lw=0, alpha=0.3)
#
#     ax.set_ylabel("time = {:.1f}".format(t_sample[i]))
#     ax.set_ylim(0, 3)
#
# ax.set_xlabel("properties")
# f.tight_layout()
# f.subplots_adjust(hspace=0.03)
