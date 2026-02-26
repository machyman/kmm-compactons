"""
Publication-quality plotting utilities.

Matplotlib configuration and helper functions for generating
figures matching the Lax Centennial paper style.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Publication style
PAPER_STYLE = {
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'axes.grid': False,
    'lines.linewidth': 1.5,
}


def set_paper_style():
    """Apply publication style to matplotlib."""
    mpl.rcParams.update(PAPER_STYLE)


def plot_collision_snapshot(x, u_pre, u_post, debris, m, ax=None,
                            title=None):
    """Plot pre/post collision and debris."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x, u_pre, 'b--', alpha=0.5, label='Pre-collision')
    ax.plot(x, u_post, 'r-', label='Post-collision')
    ax.plot(x, debris * 50, 'g-', alpha=0.7, label='Debris (×50)')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    return ax


def plot_debris_spectrum(k, spec, k_cut_phys, m, ax=None, color='C0',
                         label=None, noise_k=None, noise_spec=None):
    """Plot debris Fourier spectrum with cutoff and k=1 marker."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    lbl = label or f'$K({m},{m})$'
    ax.semilogy(k, spec, color=color, label=lbl)
    if noise_k is not None and noise_spec is not None:
        ax.semilogy(noise_k, noise_spec, '--', color='gray',
                     alpha=0.5, label='Noise floor')
    ax.axvline(k_cut_phys, color='red', linestyle=':', alpha=0.5,
               label=f'$\\kappa_{{cut}}={k_cut_phys:.1f}$')
    ax.axvline(1.0, color='orange', linestyle='--', alpha=0.5,
               label='$k=1$')
    ax.set_xlabel(r'Wavenumber $k$')
    ax.set_ylabel(r'$|\hat{w}(k)|$')
    ax.legend(frameon=False, fontsize=9)
    return ax


def plot_multi_m_spectra(results_dict, k_cut_phys, figsize=(11, 9)):
    """
    Generate the 4-panel debris spectrum figure (Figure 2 of paper).

    Parameters
    ----------
    results_dict : dict
        {m: {'k_d': array, 's_d': array, 'noise_data': {...}}}
    k_cut_phys : float
        Filter cutoff wavenumber.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.flatten()
    labels = ['(a)', '(b)', '(c)', '(d)']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']

    for idx, m_val in enumerate(sorted(results_dict.keys())):
        ax = axes_flat[idx]
        r = results_dict[m_val]
        ax.semilogy(r['k_d'], r['s_d'], color=colors[idx], linewidth=1.2)
        if r.get('noise_data'):
            nd = r['noise_data']
            ax.semilogy(nd['k_n'], nd['s_n'], '--', color='gray',
                         alpha=0.4, linewidth=0.8)
        ax.axvline(k_cut_phys, color='red', linestyle=':', alpha=0.4)
        ax.axvline(1.0, color='orange', linestyle='--', alpha=0.4)
        alpha = 2.0 / (m_val - 1.0)
        ax.set_title(f'{labels[idx]} $K({m_val},{m_val})$, '
                     f'$\\alpha={alpha:.2f}$', fontsize=11)
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$|\hat{w}(k)|$')
        ax.set_xlim(0, k_cut_phys * 1.2)

    plt.tight_layout()
    return fig
