#!/usr/bin/env python3
"""make_figures_v2_3_0.py  (hyman2026structure v2_3_0, Session 21, 2026-10-08)

Generates the three figures new or regenerated in v2_3_0:

  figure1_collision.pdf     Figure 3 (fig:collision-k22), panels (a)-(c) regenerated from the
                            rerun and new panel (d), the space-time diagram (review item 13b).
                            Input: spacetime_k22.npz from run_spacetime_collision.py.
  fig_phase-portraits.pdf   New main-text figure (fig:phase-portraits), K(2,2) phase portraits at
                            a = 0 and a = 0.5, c = 4 (review item 19).
  supp_phase-portraits.pdf  Supplement Figure S1 recomputed from the potential, a = 1 and
                            c = 1.5, 2, 3 (finding S21-1: the schematic panel (b) showed orbits on
                            the inaccessible side of the degenerate equilibrium).

Phase portraits are level sets of E = (1/2) v'^2 + V(v), with the paper's Section 7.1 potential
V(v) = v^2/2 - (c m/(m+1)) v^{(m+1)/m} + (c a - a^m) v, here m = 2.

Usage: python make_figures_v2_3_0.py <spacetime_k22.npz> <out-dir>
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

npz, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})

# ----------------------------------------------------------------------------- Figure 3
d = np.load(npz)
x, t, u, L = d['x'], d['t'], d['u'], float(d['L'])
A1, A2, X1, X2 = float(d['A1']), float(d['A2']), float(d['X1']), float(d['X2'])
c1, c2 = 0.75 * A1, 0.75 * A2                     # K(2,2): c = (m+1)/(2m) A^{m-1} = 3A/4
it = {0.0: 0, 72.0: int(np.argmin(np.abs(t - 72.0))), 145.0: len(t) - 1}

with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9}):
    fig = plt.figure(figsize=(7.5, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.35], hspace=0.42, wspace=0.12)
    titles = {0.0: '(a) Initial, $t = 0$', 72.0: '(b) Collision, $t = 72$',
              145.0: '(c) Post-collision, $t = 145$'}
    axes = []
    for j, tt in enumerate((0.0, 72.0, 145.0)):
        ax = fig.add_subplot(gs[0, j], sharey=axes[0] if axes else None)
        ui = u[it[tt]]
        ax.plot(x, ui, color='blue', lw=1.2)
        ax.set_xlim(0, L); ax.set_ylim(-0.1, 1.15)
        ax.set_title(titles[tt], fontsize=9.5)
        ax.set_xlabel('$x$'); ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel('$u(x,t)$')
            ax.text(0.96, 0.93, f'$A_1 = {A1:.1f}$\n$A_2 = {A2:.1f}$', transform=ax.transAxes,
                    va='top', ha='right', fontsize=8.5)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if tt == 145.0:
            # peak heights on the grid, as in the original panel (c)
            big = ui[np.abs(((x - 37.7 + L/2) % L) - L/2) < 6].max()
            small = ui[np.abs(((x - 2.75 + L/2) % L) - L/2) < 6].max()
            ax.text(0.96, 0.93, f"$A_1' = {big:.3f}$\n$A_2' = {small:.3f}$",
                    transform=ax.transAxes, va='top', ha='right', fontsize=8.5)
        axes.append(ax)

    ax = fig.add_subplot(gs[1, :])
    im = ax.pcolormesh(x, t, u, shading='auto', cmap='viridis', vmin=-0.03, vmax=1.0,
                       rasterized=True)
    tt = np.linspace(0, t[-1], 1200)
    for X, c in ((X1, c1), (X2, c2)):
        xx = (X + c * tt) % L
        brk = np.where(np.diff(xx) < 0)[0] + 1        # split at the periodic wrap
        for seg in np.split(np.arange(len(tt)), brk):
            ax.plot(xx[seg], tt[seg], color='white', ls='--', lw=0.9, alpha=0.9)
    ax.set_xlim(0, L); ax.set_ylim(0, t[-1])
    ax.set_xlabel('$x$'); ax.set_ylabel('$t$')
    ax.set_title('(d) Space-time diagram; dashed: free-flight paths $x_j + c_j t$ (mod 100)',
                 fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.035)
    cb.set_label('$u$')
    fig.savefig(out / 'figure1_collision.pdf', dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------------- phase portraits
def V(v, a, c, m=2):
    return v**2 / 2 - (c * m / (m + 1)) * v**((m + 1) / m) + (c * a - a**m) * v


def portrait(ax, a, c, vmax, ymax, levels, special, title, inset=None):
    vv = np.linspace(0, vmax, 1400)
    yy = np.linspace(-ymax, ymax, 1000)
    Vg, Yg = np.meshgrid(vv, yy)
    E = 0.5 * Yg**2 + V(Vg, a, c)
    ax.contour(Vg, Yg, E, levels=sorted(levels), colors='0.55', linewidths=0.6,
               linestyles='solid')
    ax.contour(Vg, Yg, E, levels=[special], colors='crimson', linewidths=1.6,
               linestyles='solid')
    ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)
    ax.set_xlim(0, vmax); ax.set_ylim(-ymax, ymax)
    ax.set_xlabel('$v$'); ax.set_title(title, fontsize=10)
    return Vg, Yg, E


with plt.rc_context({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
                     'font.size': 9}):
    # -- main-text figure: K(2,2), c = 4, a = 0 versus a = 0.5
    c = 4.0
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.75), sharey=True)
    # (a) a = 0: degenerate origin, center at v = c^2, compacton orbit on E = 0
    vc0 = c**2
    lv = [V(vc0, 0, c) * f for f in (0.85, 0.6, 0.35, 0.12)]
    portrait(axs[0], 0.0, c, 44, 10.0, lv, 0.0, '(a) $a$ = 0: compacton')
    axs[0].plot([0], [0], 'ko', ms=3.5); axs[0].plot([vc0], [0], 'k+', ms=6)
    axs[0].set_ylabel("$v'$")
    # (b) a = 0.5: saddle at a^2, center at (c-a)^2, homoclinic on E = V(a^2)
    a = 0.5; vs, vcen = a**2, (c - a)**2
    Es = V(vs, a, c); lv = [V(vcen, a, c) + (Es - V(vcen, a, c)) * f for f in (0.15, 0.4, 0.65, 0.88)]
    portrait(axs[1], a, c, 44, 10.0, lv, Es, r'(b) $a$ = 0.5: solitary wave on the shelf')
    axs[1].plot([vs], [0], 'ko', ms=3.5); axs[1].plot([vcen], [0], 'k+', ms=6)
    # insets: the neighbourhood of the shelf equilibrium
    for ax, aa in ((axs[0], 0.0), (axs[1], 0.5)):
        ins = ax.inset_axes([0.70, 0.05, 0.28, 0.40])
        vv = np.linspace(0, 1.2, 600); yy = np.linspace(-1.6, 1.6, 600)
        Vg, Yg = np.meshgrid(vv, yy)
        Eg = 0.5 * Yg**2 + V(Vg, aa, c)
        ins.contour(Vg, Yg, Eg, levels=[V(aa**2, aa, c)], colors='crimson', linewidths=1.2)
        ins.plot([aa**2], [0], 'ko', ms=2.5)
        ins.axhline(0, color='k', lw=0.3)
        ins.set_xlim(0, 1.2); ins.set_ylim(-1.6, 1.6)
        ins.tick_params(labelsize=6, length=2, pad=1)
        ins.set_xticks([0, 0.5, 1.0]); ins.set_yticks([-1, 0, 1])
        ins.set_facecolor('white')
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / 'fig_phase-portraits.pdf')
    plt.close(fig)

    # -- Supplement Figure S1: a = 1, c* = 2; c = 1.5, 2, 3
    a = 1.0
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    # (a) c = 1.5 < c*: center at a^2 = 1, saddle at (c-a)^2 = 0.25
    c = 1.5; vcen, vsad = a**2, (c - a)**2
    Ec, Es = V(vcen, a, c), V(vsad, a, c)
    portrait(axs[0], a, c, 2.6, 0.9, [Ec + (Es - Ec) * f for f in (0.15, 0.4, 0.7, 0.92)], Es,
             r'(a) $c = 1.5 < c^*$: center')
    axs[0].plot([vcen], [0], 'k+', ms=6); axs[0].plot([vsad], [0], 'ko', ms=3)
    axs[0].set_ylabel("$v'$")
    # (b) c = c* = 2: degenerate (inflection) equilibrium at a^2 = 1, no closed orbits
    c = 2.0; E0 = V(a**2, a, c)
    portrait(axs[1], a, c, 2.6, 0.9, [E0 + dE for dE in (-0.3, -0.12, 0.08, 0.2)], E0,
             r'(b) $c = c^*$: degenerate')
    axs[1].plot([a**2], [0], 'ko', ms=3)
    # (c) c = 3 > c*: saddle at a^2 = 1, center at (c-a)^2 = 4, homoclinic loop
    c = 3.0; vsad, vcen = a**2, (c - a)**2
    Es, Ec = V(vsad, a, c), V(vcen, a, c)
    portrait(axs[2], a, c, 8.2, 2.2, [Ec + (Es - Ec) * f for f in (0.2, 0.5, 0.8)], Es,
             r'(c) $c = 3 > c^*$: saddle')
    axs[2].plot([vsad], [0], 'ko', ms=3); axs[2].plot([vcen], [0], 'k+', ms=6)
    axs[2].set_ylim(-2.2, 2.2)
    for ax in axs[:2]:
        ax.set_ylim(-0.9, 0.9)
    fig.tight_layout(w_pad=0.5)
    fig.savefig(out / 'supp_phase-portraits.pdf')
    plt.close(fig)

print('wrote', ', '.join(p.name for p in sorted(out.glob('*.pdf'))))
