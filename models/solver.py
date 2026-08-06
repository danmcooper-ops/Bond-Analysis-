"""Root finders.

Deliberately hand-rolled rather than pulled from scipy. Bond root-finding is
one-dimensional and well-behaved, the implementations are short enough to test
directly, and — the actual reason — the pipeline needs a "return None, never
raise" contract. A solver that throws on a pathological bond would take down a
30,000-row run; one that returns None lets the caller set `ytm_solver_failed`
and a rating cap demote that single row to HOLD.

Newton with an analytic derivative is the fast path (for yield-from-price the
derivative is just duration). Brent is the fallback that always terminates on
a bracketing interval.
"""


def brent(f, lo, hi, tol=1e-12, max_iter=200):
    """Brent's method on [lo, hi]. Returns None rather than raising.

    Returns None when the interval does not bracket a root (same sign at both
    ends), when f is undefined somewhere it needs it, or when the iteration
    budget runs out.
    """
    try:
        fa, fb = f(lo), f(hi)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if fa is None or fb is None:
        return None
    if fa == 0:
        return lo
    if fb == 0:
        return hi
    if fa * fb > 0:
        return None                     # no bracket, no answer

    a, b = lo, hi
    c, fc = a, fa
    d = e = b - a

    for _ in range(max_iter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2.0 * 1e-16 * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol_act or fb == 0.0:
            return b
        if abs(e) >= tol_act and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p, q = 2.0 * xm * s, 1.0 - s          # secant
            else:
                q, r = fa / fc, fb / fc               # inverse quadratic
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol_act * q), abs(e * q)):
                e, d = d, p / q                       # accept interpolation
            else:
                d = e = xm                            # fall back to bisection
        else:
            d = e = xm
        a, fa = b, fb
        b += d if abs(d) > tol_act else (tol_act if xm > 0 else -tol_act)
        try:
            fb = f(b)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        if fb is None:
            return None
    return None


def newton(f, fprime, x0, tol=1e-12, max_iter=100, bracket=None):
    """Newton's method with an analytic derivative. Returns None on failure.

    `bracket` is a (lo, hi) sanity box: an iterate that escapes it aborts, so
    a bad derivative cannot send the search off to a spurious root in a
    region that makes no financial sense (a yield of 400%, say).
    """
    x = x0
    for _ in range(max_iter):
        try:
            fx = f(x)
            if fx is None:
                return None
            if abs(fx) < tol:
                return x
            dfx = fprime(x)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        if dfx is None or dfx == 0:
            return None
        step = fx / dfx
        x_new = x - step
        if bracket is not None and not (bracket[0] <= x_new <= bracket[1]):
            return None
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return None


def solve(f, fprime=None, guess=None, bracket=(-0.99, 5.0), tol=1e-12):
    """Try Newton, fall back to Brent. Returns None if both fail.

    This is the entry point callers should use. The default bracket spans
    yields from -99% to +500%: wide enough for distressed paper and for the
    negative-yield world, narrow enough that 1 + y/m stays positive for any
    frequency >= 1.
    """
    if fprime is not None and guess is not None:
        root = newton(f, fprime, guess, tol=tol, bracket=bracket)
        if root is not None:
            return root
    return brent(f, bracket[0], bracket[1], tol=tol)
