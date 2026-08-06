"""Yield curve: bootstrapping, interpolation, discounting.

Two decisions here are load-bearing.

**Monotone cubic (Fritsch-Carlson / PCHIP) interpolation on ZERO rates.**
Linear-on-par is the obvious shortcut and it is wrong in a way that matters:
it produces kinked forward rates, and roll-down — which is a Rates gate — is
computed by sliding a bond down the curve. A kink in the forwards shows up as
a bond that appears to roll down sharply purely because it sits next to a
quoted tenor. PCHIP is monotonicity-preserving, so it cannot invent a
non-monotone wiggle between two monotone knots, and it needs no dependency.

**Flat extrapolation, never a slope.** Beyond 30Y and below 1M the curve is
held flat. Extrapolating the long-end slope produces absurd 40-year discount
factors, and there is nothing to calibrate an extrapolation against.
"""

from models.solver import brent

# Standard Treasury quote tenors. 1.5M appears in the par XML feed.
TENOR_YEARS = {
    '1M': 1 / 12, '1.5M': 0.125, '2M': 2 / 12, '3M': 0.25, '4M': 4 / 12,
    '6M': 0.5, '1Y': 1.0, '2Y': 2.0, '3Y': 3.0, '5Y': 5.0, '7Y': 7.0,
    '10Y': 10.0, '20Y': 20.0, '30Y': 30.0,
}


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _pchip_tangents(xs, ys):
    """Fritsch-Carlson tangents: monotone by construction.

    Where consecutive secants have opposite signs (a local extremum) the
    tangent is pinned to zero, which is what prevents overshoot. Elsewhere the
    weighted harmonic mean of the neighbouring secants is used — the harmonic
    mean is the part that guarantees the result stays inside the data's range.
    """
    n = len(xs)
    if n == 2:
        s = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return [s, s]

    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    delta = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]

    m = [0.0] * n
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0                      # local extremum: flatten
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    # One-sided three-point endpoints, clamped so they cannot overshoot.
    def _endpoint(d0, d1, h0, h1):
        d = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if d * d0 <= 0:
            return 0.0
        if abs(d) > 3 * abs(d0):
            return 3 * d0
        return d

    m[0] = _endpoint(delta[0], delta[1], h[0], h[1])
    m[-1] = _endpoint(delta[-1], delta[-2], h[-1], h[-2])
    return m


def interpolate(x, xs, ys, method='monotone_cubic'):
    """Interpolate y at x over knots (xs, ys). Flat outside the knot range."""
    if not xs:
        return None
    if len(xs) == 1:
        return ys[0]
    if x <= xs[0]:
        return ys[0]                        # flat extrapolation
    if x >= xs[-1]:
        return ys[-1]

    # Locate the bracketing interval.
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid

    h = xs[hi] - xs[lo]
    t = (x - xs[lo]) / h

    if method == 'linear':
        return ys[lo] + t * (ys[hi] - ys[lo])

    m = _pchip_tangents(xs, ys)
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * ys[lo] + h10 * h * m[lo] + h01 * ys[hi] + h11 * h * m[hi]


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------

def _discount_from_zero(z, t, frequency=2):
    base = 1.0 + z / frequency
    if base <= 0:
        return None
    return base ** (-t * frequency)


def bootstrap_zero_curve(tenors_years, par_yields, frequency=2,
                         method='monotone_cubic', max_sweeps=60, tol=1e-15):
    """Bootstrap semiannually-compounded zero rates from a par curve.

    Each quoted tenor's zero rate is the one that reprices that tenor's par
    bond to exactly 100, with intermediate coupon dates supplied by the
    interpolator. Solved with a root-finder rather than the closed-form
    rearrangement, because the closed form assumes every intermediate coupon
    date is already a knot — false for a curve quoted at 1/2/3/5/7/10/20/30
    years, where a 10-year bond has 19 coupon dates and only a handful of them
    land on knots.

    ITERATED TO A FIXED POINT, not solved in one forward pass. A single pass
    is what you would write first and it is subtly wrong here: monotone-cubic
    tangents depend on the neighbouring knots, so adding the 20Y and 30Y
    points retroactively changes the interpolated zeros in the 7-10Y region
    that the 10Y bond was bootstrapped against. The 10Y then no longer
    reprices to par. One pass leaves ~0.1 cents of error per 100 face —
    small enough to look like rounding, large enough to pollute every
    Z-spread. Gauss-Seidel sweeps until no knot moves, which takes only a few
    iterations because the system is strongly diagonally dominant.

    Returns [(t_years, zero_rate), ...] sorted by tenor.
    """
    pairs = sorted((t, c) for t, c in zip(tenors_years, par_yields) if t > 0)
    if not pairs:
        return []
    ts = [t for t, _ in pairs]
    cs = [c for _, c in pairs]
    m = frequency

    # Seed with the par yields: the right order of magnitude, and exact on a
    # flat curve, so the sweeps usually converge in two or three passes.
    zs = list(cs)

    def solve_one(i):
        t, c = ts[i], cs[i]
        n = int(round(t * m))
        if n <= 1:
            # One payment or less: par and zero coincide under the same
            # compounding, so there is nothing to bootstrap.
            return c
        cpn = c / m
        coupon_times = [k / m for k in range(1, n)]

        def price_error(z):
            trial = list(zs)
            trial[i] = z
            total = 0.0
            for ct in coupon_times:
                zc = interpolate(ct, ts, trial, method=method)
                df = _discount_from_zero(zc, ct, m)
                if df is None:
                    return None
                total += cpn * df
            df_t = _discount_from_zero(z, t, m)
            if df_t is None:
                return None
            total += (1.0 + cpn) * df_t
            return total - 1.0

        z = brent(price_error, -0.50, 1.00, tol=1e-15)
        # Do not fabricate a knot on failure: fall back to the par yield,
        # which keeps the curve usable and shows up in repricing_error().
        return z if z is not None else c

    for _ in range(max_sweeps):
        max_change = 0.0
        for i in range(len(ts)):
            new_z = solve_one(i)
            max_change = max(max_change, abs(new_z - zs[i]))
            zs[i] = new_z
        if max_change < tol:
            break

    return list(zip(ts, zs))


# ---------------------------------------------------------------------------
# Curve object
# ---------------------------------------------------------------------------

class YieldCurve:
    """A dated par curve plus its bootstrapped zeros.

    All rates are decimals (0.0425, not 4.25) and all zeros are quoted on the
    same compounding basis as `frequency`.
    """

    def __init__(self, as_of, tenors_years, par_yields, frequency=2,
                 method='monotone_cubic'):
        pairs = sorted(zip(tenors_years, par_yields))
        self.as_of = as_of
        self.frequency = frequency
        self.method = method
        self.par_t = [t for t, _ in pairs]
        self.par_y = [y for _, y in pairs]
        zeros = bootstrap_zero_curve(self.par_t, self.par_y,
                                     frequency=frequency, method=method)
        self.zero_t = [t for t, _ in zeros]
        self.zero_z = [z for _, z in zeros]

    @classmethod
    def from_par_dict(cls, as_of, par_dict, frequency=2,
                      method='monotone_cubic'):
        """Build from {'10Y': 0.0510, ...}. Unknown tenor labels are ignored."""
        items = [(TENOR_YEARS[k], v) for k, v in par_dict.items()
                 if k in TENOR_YEARS and v is not None]
        if not items:
            raise ValueError("No recognisable tenors in par_dict")
        items.sort()
        return cls(as_of, [t for t, _ in items], [v for _, v in items],
                   frequency=frequency, method=method)

    # -- lookups ------------------------------------------------------------

    def par(self, t):
        """Par yield at tenor t (years)."""
        return interpolate(t, self.par_t, self.par_y, method=self.method)

    def zero(self, t):
        """Zero rate at tenor t (years), same compounding as `frequency`."""
        return interpolate(t, self.zero_t, self.zero_z, method=self.method)

    def discount(self, t, spread=0.0):
        """Discount factor to t, optionally with a parallel spread added."""
        z = self.zero(t)
        if z is None:
            return None
        return _discount_from_zero(z + spread, t, self.frequency)

    def forward(self, t1, t2):
        """Forward rate between t1 and t2, same compounding as `frequency`."""
        if t2 <= t1:
            return None
        d1, d2 = self.discount(t1), self.discount(t2)
        if not d1 or not d2 or d2 <= 0:
            return None
        m = self.frequency
        return m * ((d1 / d2) ** (1.0 / (m * (t2 - t1))) - 1.0)

    def shift(self, bp):
        """A parallel-shifted copy. bp is in basis points."""
        d = bp / 10000.0
        return YieldCurve(self.as_of, self.par_t, [y + d for y in self.par_y],
                          frequency=self.frequency, method=self.method)

    # -- diagnostics --------------------------------------------------------

    def repricing_error(self):
        """Max absolute error (per 100 face) repricing each quoted par bond
        off the bootstrapped zeros. Should be ~1e-10; a large value means the
        bootstrap or the interpolator is broken."""
        worst = 0.0
        m = self.frequency
        for t, c in zip(self.par_t, self.par_y):
            n = int(round(t * m))
            if n <= 1:
                continue
            cpn = c / m
            total = sum(cpn * self.discount(k / m) for k in range(1, n))
            total += (1.0 + cpn) * self.discount(t)
            worst = max(worst, abs(total * 100.0 - 100.0))
        return worst

    def __repr__(self):
        return (f"YieldCurve(as_of={self.as_of}, "
                f"tenors={len(self.par_t)}, "
                f"10y_par={self.par(10.0):.4%})")
