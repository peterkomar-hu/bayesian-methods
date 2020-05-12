import numpy as np
from scipy.stats import chi2


def equalize_xy(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plot_max = max(*xlim, *ylim)
    plot_min = min(*xlim, *ylim)
    plot_lim = (plot_min, plot_max)
    ax.set_xlim(plot_lim)
    ax.set_ylim(plot_lim)
    

def add_margin(ax, single_margin_ratio=0.2):
    width_ratio = 2 * single_margin_ratio
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xwidth = xlim[1] - xlim[0]
    xcenter = (xlim[1] + xlim[0]) / 2
    ywidth = ylim[1] - ylim[0]
    ycenter = (ylim[1] + ylim[0]) / 2
    
    xradius = (1 + width_ratio) * xwidth / 2
    yradius = (1 + width_ratio) * ywidth / 2
    ax.set_xlim([xcenter - xradius, xcenter + xradius])
    ax.set_ylim([ycenter - yradius, ycenter + yradius])
    
    
def plot_gaussian_contour(ax, mu, Sigma, color='k', linestyle = '-', q_levels=(0.95,), pixels=200):
    
    # plot contours of true model
    levels = []
    for q in q_levels:
        # relying on that y = (x.T) (Sigma^{-1}) (x) is chi^2 distributed with dof=2
        Z_level = chi2.ppf(q, df=2) 
        levels.append(Z_level)
    levels.sort()
    
    # generate grid
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x = np.arange(xmin, xmax, (xmax - xmin)/float(pixels))
    y = np.arange(ymin, ymax, (ymax - ymin)/float(pixels))
    X, Y = np.meshgrid(x, y)
    XY = np.array([X - mu[0], Y - mu[1]])
    
    cov_inv = np.linalg.inv(Sigma)
    Z = np.einsum(XY, [2, Ellipsis], cov_inv, [2,3], XY, [3, Ellipsis])
    
    CS = ax.contour(X, Y, Z, levels=levels, colors=color, linewidths=1, linestyles=linestyle)
