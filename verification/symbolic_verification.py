#!/usr/bin/env python3
"""
symbolic_verification.py: SymPy verification of Appendix C of Hyman (2026), "What Remains When
Integrability Fails? Mathematical Structure in the K(m,m) Compacton Equations", DCDS-A
(Peter D. Lax Centennial Memorial Volume).

Appendix C ("Symbolic verification of Section 7 identities") lists the identities checked in its
table (Table 8 of the paper). This script re-derives every row of that table, the residual
polynomials R_4 and R_5 displayed after it (kappa_4 = 80, kappa_5 = 75), the statements about
R_2 and R_3 in the proof of Proposition 7.3 (period monotonicity), and the numerical
confirmations the appendix reports at (a, c) = (0.01, 4), (0.1, 4), (0.5, 4), (1.0, 4).

Checks marked "numerical" sample parameter values rather than prove a statement; the paper
marks the same statements as numerical (m >= 4, and the monotonicity of R_m on the annulus).

Run from anywhere:   python verification/symbolic_verification.py
Requirements: SymPy >= 1.12, NumPy, SciPy.  Exit status 0 when every check passes.
"""
import sys

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

RESULTS = []


def check(label, ok, detail=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""), flush=True)


def zero(expr):
    return sp.simplify(sp.expand(expr)) == 0


a, c, v, w, U, xi, eps, s, D = sp.symbols('a c v w U xi epsilon s D', positive=True)
m = sp.symbols('m', positive=True)
W = sp.Function('W')(xi)


def V_of(mm, aa, cc, var):
    """Potential (eq. potential): V(v) = v^2/2 - c m/(m+1) v^((m+1)/m) + (c a - a^m) v."""
    return sp.Rational(1, 2) * var**2 - cc * mm / (mm + 1) * var**(sp.Integer(mm + 1) / mm if isinstance(mm, int)
                                                   else (mm + 1) / mm) + (cc * aa - aa**mm) * var


# ---------------------------------------------------------------- Table 8 rows 1-3
print("Table 8, rows 1-3: equilibria of the potential")
Vg = V_of(m, a, c, v)
check("row 1: V'(a^m) = 0 for general m", zero(sp.diff(Vg, v).subs(v, a**m)))
check("        V'''(v) = c(m-1)/m^2 v^(1/m-2) (used in Prop. 7.3)",
      zero(sp.diff(Vg, v, 3) - c * (m - 1) / m**2 * v**(1 / m - 2)))
V2 = V_of(2, a, c, v)
check("row 2: V''(a^2) = 1 - c/(2a) for K(2,2)", zero(sp.diff(V2, v, 2).subs(v, a**2) - (1 - c / (2 * a))))
V2s = V2.subs(c, a + s)                              # c = a + s > a, so sqrt((c-a)^2) = c - a = s
check("row 3: V'((c-a)^2) = 0 (second equilibrium)", zero(sp.diff(V2s, v).subs(v, s**2)))
check("        V''((c-a)^2) = (c-2a)/(2(c-a)) > 0 for c > 2a (a center)",
      zero(sp.diff(V2s, v, 2).subs(v, s**2) - (s - a) / (2 * s)))

# ---------------------------------------------------------------- row 4: first integral
print("Table 8, row 4: first integral")
tw = -c * W + W**2 + sp.diff(W**2, xi, 2) + (c * a - a**2)
check("eq. (w-ode) is the K(2,2) traveling-wave ODE with C0 = a^2 - c a",
      zero(tw - (-c * W + W**2 + 2 * W * sp.diff(W, xi, 2) + 2 * sp.diff(W, xi)**2 + (c * a - a**2))))
F = c * w / 3 - w**2 / 4 - (c * a - a**2) / 2 + D / w**2      # (w')^2 = F(w)
check("(w')^2 = F(w) solves eq. (w-ode) identically in w and D",
      zero(-c * w + w**2 + 2 * w * sp.diff(F, w) / 2 + 2 * F + (c * a - a**2)))
Dbc = sp.solve(sp.Eq(F.subs(w, a), 0), D)[0]
check("row 4: w -> a, w' -> 0 fixes D = c a^3/6 - a^4/4", zero(Dbc - (c * a**3 / 6 - a**4 / 4)))
w0 = sp.Rational(4, 3) * c * sp.cos(xi / 4)**2
check("a = 0: w0 = (4c/3) cos^2(xi/4) satisfies (w')^2 = c w/3 - w^2/4 (eq. compacton-recovery)",
      sp.simplify((sp.diff(w0, xi)**2 - (c * w0 / 3 - w0**2 / 4)).rewrite(sp.exp)) == 0)
check("        and w0 = w0' = 0 at |xi| = 2 pi", w0.subs(xi, 2 * sp.pi) == 0 and sp.diff(w0, xi).subs(xi, 2 * sp.pi) == 0)

# ---------------------------------------------------------------- rows 5-9: factored form, roots, limits
print("Table 8, rows 5-9: factored form, roots, decay rate, limits")
P = sp.expand(4 * w**2 * F.subs(D, Dbc))
Q = w**2 + (2 * a - sp.Rational(4, 3) * c) * w + (a**2 - sp.Rational(2, 3) * c * a)
check("row 5: 4 w^2 (w')^2 = (w-a)^2 (w+ - w)(w - w-), i.e. -(w-a)^2 Q(w)", zero(P + (w - a)**2 * Q))
check("        w = a is a double root", zero(P.subs(w, a)) and zero(sp.diff(P, w).subs(w, a)))
disc = sp.expand((2 * a - sp.Rational(4, 3) * c)**2 - 4 * (a**2 - sp.Rational(2, 3) * c * a))
check("eq. (disc): discriminant of Q = (8c/9)(2c - 3a)", zero(disc - sp.Rational(8, 9) * c * (2 * c - 3 * a)))
r = sp.sqrt(2 * c * (2 * c - 3 * a))
wp, wm = 2 * c / 3 - a + r / 3, 2 * c / 3 - a - r / 3
check("row 6: w+- = 2c/3 - a +- sqrt(2c(2c-3a))/3 are the roots of Q (eq. wpm)",
      zero(Q.subs(w, wp)) and zero(Q.subs(w, wm)))
check("row 7: (w+ - a)(a - w-) = 2a(c - 2a)", zero((wp - a) * (a - wm) - 2 * a * (c - 2 * a)))
check("        so the linearization at w = a gives kappa^2 = (c-2a)/(2a) (eq. decay-rate)",
      zero((wp - a) * (a - wm) / (4 * a**2) - (c - 2 * a) / (2 * a)))
check("row 8: w+ -> 4c/3 as a -> 0+", sp.simplify(sp.limit(wp, a, 0, '+') - 4 * c / 3) == 0)
fmax = wp - a
check("eq. (amplitude-speed): f_max = w+ - a = 2c/3 - 2a + sqrt(2c(2c-3a))/3",
      zero(fmax - (2 * c / 3 - 2 * a + r / 3)))
ser = sp.series(fmax.subs(c, 2 * a + eps), eps, 0, 2).removeO()
check("row 9: f_max ~ (3/2)(c - 2a) near onset c = 2a", zero(ser - sp.Rational(3, 2) * eps))
grid = [(aa, k * aa) for aa in (0.01, 0.1, 0.5, 1.0, 2.0) for k in np.linspace(2.001, 60, 400)]
wpf = sp.lambdify((a, c), wp, 'numpy'); wmf = sp.lambdify((a, c), wm, 'numpy')
ok = all(wmf(aa, cc) < 0 < aa < wpf(aa, cc) and wpf(aa, cc) > cc - aa for aa, cc in grid)
check("w- < 0 < a < w+, and crest w+ above the center c - a, for c > 2a (numerical, 2000 points)", ok)

# ---------------------------------------------------------------- row 10: Chicone numerator and R_m
print("Table 8, row 10 and Proposition 7.3: Chicone numerator, R_m")
t = sp.symbols('t'); Wt = sp.Function('Wt')(t)
chi2 = sp.diff(Wt / sp.diff(Wt, t)**2, t, 2)
Nw = 6 * Wt * sp.diff(Wt, t, 2)**2 - 3 * sp.diff(Wt, t)**2 * sp.diff(Wt, t, 2) - 2 * Wt * sp.diff(Wt, t) * sp.diff(Wt, t, 3)
check("chi = W/(W')^2 has chi'' = N/(W')^4 with N = 6W(W'')^2 - 3(W')^2 W'' - 2W W' W'''",
      sp.simplify(chi2 - Nw / sp.diff(Wt, t)**4) == 0)


def residual_poly(mm, aa):
    """P_m(U) with N = c (U-a)^4 P_m(U) / U^(2m-1), from V in the physical field U = v^(1/m)."""
    Vt = V_of(mm, aa, c, U**mm) - V_of(mm, aa, c, aa**mm)
    V1 = U**mm - c * U + (c * aa - aa**mm)
    V2_ = 1 - c / mm * U**(1 - mm)
    V3 = c * (mm - 1) / sp.Integer(mm)**2 * U**(1 - 2 * mm)
    N = 6 * Vt * V2_**2 - 3 * V1**2 * V2_ - 2 * Vt * V1 * V3
    Pm = sp.cancel(sp.together(N * U**(2 * mm - 1) / (c * (U - aa)**4)))
    return sp.expand(Pm)


def is_poly_in_U(expr):
    try:
        sp.Poly(expr, U); return sp.denom(sp.together(expr)).free_symbols.isdisjoint({U})
    except sp.PolynomialError:
        return False


P2 = residual_poly(2, a)
R2 = 3 * U**2 + (12 * a - 8 * c) * U + (3 * a - 2 * c) * (a - c)
k2 = sp.cancel(R2 / P2)
check("m = 2: N = c (U-a)^4 R_2 / (kappa_2 U^3) with R_2 as printed", is_poly_in_U(P2) and k2.is_number and k2 > 0,
      f"kappa_2 = {k2}")
check("        R_2(0) = (3a-2c)(a-c) and R_2(c-a) = 3(c-a)(2a-c)",
      zero(R2.subs(U, 0) - (3 * a - 2 * c) * (a - c)) and zero(R2.subs(U, c - a) - 3 * (c - a) * (2 * a - c)))
check("        inner (saddle) edge U = c - a: U^2 - cU + (ca - a^2) = (U-a)(U-(c-a))",
      zero(U**2 - c * U + (c * a - a**2) - (U - a) * (U - (c - a))))
check("        vertex (4c-6a)/3 lies left of c - a exactly when c < 3a", zero((c - a) - (4 * c - 6 * a) / 3 - (3 * a - c) / 3))
P3 = residual_poly(3, a)
ct = c / a**2
R3_0 = 2 * a**5 * (ct - 2) * (ct - 1)
k3 = sp.cancel(R3_0 / P3.subs(U, 0))
check("m = 3: N = c (U-a)^4 R_3 / (kappa_3 U^5), R_3(0) = 2a^5 (c~-2)(c~-1)", is_poly_in_U(P3) and k3.is_number and k3 > 0,
      f"kappa_3 = {k3}")
R3 = k3 * P3
cs = a**2 * (s**2 + 3) / 4                            # s = sqrt(4 c~ - 3), c~ = c/a^2 in (1, 3) <=> s in (1, 3)
Us = (s - 1) * a / 2
check("        U_s = (sqrt(4c/a^2 - 3) - 1) a / 2 is the saddle: U^3 - cU + (ca - a^3) = 0",
      zero((Us**3 - c * Us + (c * a - a**3)).subs(c, cs)))
check("        R_3(U_s) = -(3/2) a^5 (c~-1)(4c~-3)(sqrt(4c~-3) - 3)",
      zero(R3.subs(U, Us).subs(c, cs) + sp.Rational(3, 2) * a**5 * ((s**2 - 1) / 4) * s**2 * (s - 3)))
R4 = (21 * U**8 + 84 * U**7 + 210 * U**6 + 420 * U**5 - 114 * U**5 * c + 420 * U**4 - 141 * U**4 * c + 300 * U**3
      - 120 * U**3 * c + 150 * U**2 - 90 * U**2 * c + 12 * U**2 * c**2 + 60 * U - 54 * U * c + 12 * U * c**2
      + 15 - 21 * c + 6 * c**2)
R5 = (18 * U**11 + 72 * U**10 + 180 * U**9 + 360 * U**8 + 630 * U**7 - 124 * U**7 * c + 684 * U**6 - 172 * U**6 * c
      + 591 * U**5 - 169 * U**5 * c + 420 * U**4 - 140 * U**4 * c + 240 * U**3 - 110 * U**3 * c + 10 * U**3 * c**2
      + 120 * U**2 - 76 * U**2 * c + 12 * U**2 * c**2 + 48 * U - 43 * U * c + 9 * U * c**2 + 12 - 16 * c + 4 * c**2)
ONE = sp.Integer(1)                                   # exact arithmetic throughout
P4, P5 = residual_poly(4, ONE), residual_poly(5, ONE)
check("m = 4 (a = 1): R_4 as printed equals kappa_4 P_4 with kappa_4 = 80", zero(80 * P4 - R4))
check("m = 5 (a = 1): R_5 as printed equals kappa_5 P_5 with kappa_5 = 75", zero(75 * P5 - R5))
check("        R_4(0) = 3(c-1)(2c-5), R_5(0) = 4(c-3)(c-1); sign changes at 5/2 in (1,4) and 3 in (1,5)",
      zero(R4.subs(U, 0) - 3 * (c - 1) * (2 * c - 5)) and zero(R5.subs(U, 0) - 4 * (c - 3) * (c - 1)))
for mm in (2, 3, 4, 5):                               # the a = 1 display covers general a by scaling
    Pa, P1 = residual_poly(mm, a), residual_poly(mm, ONE)
    ratio = sp.simplify(Pa / P1.subs({U: U / a, c: c / a**(mm - 1)}))
    check(f"m = {mm}: scaling U -> U/a, c -> c/a^(m-1) holds up to a positive power of a",
          ratio.free_symbols <= {a} and sp.simplify(sp.log(ratio) / sp.log(a)).is_number, f"factor {ratio}")


def annulus(Pf, mm, cc):
    """Interior sample of the period annulus (a = 1) and its inner edge."""
    V = lambda u: 0.5 * u**(2 * mm) - cc * mm / (mm + 1) * u**(mm + 1) + (cc - 1) * u**mm
    if cc > 1:
        g = lambda u: (u**mm - cc * u + (cc - 1)) / (u - 1)
        u_in = brentq(g, 1e-12, 1 - 1e-12, xtol=1e-15)
    else:
        u_in = 0.0
    E = V(u_in)
    u_out = brentq(lambda u: V(u) - E, 1 + 1e-12, 50.0, xtol=1e-14)
    uu = np.linspace(u_in, u_out, 4003)[1:-1]
    return u_in, uu


print("Proposition 7.3 and Appendix C: R_m on the period annulus (numerical, a = 1)")
for mm in range(2, 13):
    Pm = {2: P2.subs(a, 1), 3: P3.subs(a, 1), 4: P4, 5: P5}.get(mm) or residual_poly(mm, ONE)
    Pf = sp.lambdify((U, c), Pm, 'numpy'); dPf = sp.lambdify((U, c), sp.diff(Pm, U), 'numpy')
    worst, inc, edge = np.inf, True, True
    for cc in np.linspace(0.01, mm - 0.01, 300):
        u_in, uu = annulus(Pf, mm, cc)
        vals = Pf(uu, cc); worst = min(worst, vals.min() / np.abs(vals).max())
        if mm <= 5:
            inc &= bool(np.all(dPf(uu, cc) > 0))
            if cc > 1: edge &= bool(Pf(u_in, cc) > 0)
    extra = ""
    if mm <= 5:
        extra = f"; increasing across the annulus: {inc}; positive at the saddle edge: {edge}"
    check(f"m = {mm:2d}: R_m > 0 on the annulus for 0 < c < c* (300 speeds x 4001 points)", worst > 0,
          f"min R_m / max|R_m| = {worst:.3g}{extra}")
    if mm <= 5:
        RESULTS.append((f"m = {mm}: R_m increasing on the annulus", inc))
        RESULTS.append((f"m = {mm}: R_m positive at the saddle edge", edge))

# ---------------------------------------------------------------- numerical confirmations stated in Appendix C
print("Appendix C numerical confirmations: K(2,2) traveling-wave ODE at c = 4 (tolerance 1e-10)")
for aa, cc in [(0.01, 4.0), (0.1, 4.0), (0.5, 4.0), (1.0, 4.0)]:
    wpv, wmv = float(wpf(aa, cc)), float(wmf(aa, cc))
    kap = np.sqrt((cc - 2 * aa) / (2 * aa))
    G = lambda ww, p: (cc * ww - ww**2 - 2 * p**2 - (cc * aa - aa**2)) / (2 * ww)
    h = 1e-30
    kap_num = np.sqrt(np.imag(G(aa + 1j * h, 0.0)) / h)       # linearization at the shelf, complex step
    d = 1e-9 * (wpv - aa)
    ev = lambda x, y: y[1]; ev.terminal, ev.direction = True, -1
    sol = solve_ivp(lambda x, y: [y[1], G(y[0], y[1])], [0, 400], [aa + d, kap * d], method='DOP853',
                    rtol=1e-13, atol=1e-16, events=ev, dense_output=True)
    crest = sol.y_events[0][0][0]
    wm_num = (aa**2 - 2 * cc * aa / 3) / crest                 # product of the roots of Q
    xs = np.linspace(0, sol.t_events[0][0], 20001); ww, pp = sol.sol(xs)
    rhs = (ww - aa)**2 * (wpv - ww) * (ww - wmv)
    res = np.max(np.abs(4 * ww**2 * pp**2 - rhs)) / np.max(np.abs(rhs))
    e1 = abs(kap_num / kap - 1); e2 = abs(crest / wpv - 1)
    e3 = abs((crest - aa) * (aa - wm_num) / (2 * aa * (cc - 2 * aa)) - 1)
    check(f"(a, c) = ({aa}, {cc}): factored form along the integrated orbit", res < 1e-10, f"max rel. residual {res:.1e}")
    check(f"(a, c) = ({aa}, {cc}): decay rate from the linearized ODE", e1 < 1e-10, f"rel. error {e1:.1e}")
    check(f"(a, c) = ({aa}, {cc}): crest from integration equals w+", e2 < 1e-10, f"rel. error {e2:.1e}")
    check(f"(a, c) = ({aa}, {cc}): (w+ - a)(a - w-) = 2a(c - 2a) with the integrated crest", e3 < 1e-10,
          f"rel. error {e3:.1e}")

n_fail = sum(1 for _, ok in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_fail} of {len(RESULTS)} checks pass" + ("" if not n_fail else f"; {n_fail} FAIL"))
sys.exit(1 if n_fail else 0)
