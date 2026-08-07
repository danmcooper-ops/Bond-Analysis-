# scripts/scoring_kernel.py
#
# VENDORED from stock-analysis-model @ 168c17a792f28ec56af874fa2de1be5c3ae92db7
# on 2026-08-06, extracted from that repo's scripts/scoring.py.
# Kernel module — keep structurally in sync; run `python tools/kernel_diff.py`.
#
# Local divergences from upstream, all deliberate:
#   1. Split. Upstream is one 1124-line scripts/scoring.py mixing generic
#      machinery with the equity GATES list. Here the machinery lives in this
#      file and every asset-class-specific decision arrives via a ScoringSpec.
#      That keeps the vendored surface small and makes drift diffable
#      function-by-function (which is how tools/kernel_diff.py compares).
#   2. PEER_KEY / MIN_PEER_SCORING replace upstream's hardcoded r.get('sector')
#      and MIN_SECTOR_SCORING, and relative_mode 'sector' becomes 'peer'.
#      Bonds peer on "{TSY|IG|HY}|{maturity bucket}", not on sector.
#   3. rating_from_composite takes an asset_class and consults per-class
#      thresholds before the base ones. Upstream never needed this because
#      every equity row scores the same 26 gates; here a Treasury scores 10
#      gates across 3 categories and a corporate 25 across 5, so their
#      composites are not on one scale.
#   4. The Monte-Carlo confidence penalty is generalised to an optional
#      (field, tiers) pair on the spec. Upstream hardcodes mc_cv > 0.40 -> x0.85
#      and > 0.30 -> x0.93; bonds have no MC valuation, so it defaults off.
#
# Everything else — the Gate namedtuple, the applicable-vs-missing distinction,
# _score_linear, _ranked_percentiles, the category renormalisation, and the
# cat_avgs[cat] is None drop-out — is upstream's design, unchanged. That
# drop-out is what makes Treasuries work: see compute_continuous_scores.
"""Asset-class-agnostic screening and scoring machinery.

The whole model is driven by one list of Gate specs. Each Gate carries both a
pass/fail test and a continuous 0-100 score, so a gate's threshold and its
scoring range live side by side and cannot drift apart.

The load-bearing idea, inherited from the equity model, is the difference
between two kinds of absent:

  * STRUCTURALLY INAPPLICABLE — gate.applicable(row) is False. The metric
    cannot describe this instrument at all (leverage for a Treasury; a
    yield-to-maturity for a floater). Excluded from numerator AND denominator,
    so the instrument is not mechanically failed on questions that do not
    apply to it.
  * MISSING DATA — the field is None but the gate does apply. Scores 0
    (worst) against its full weight, so sparse rows stay penalised.

Getting that distinction right is the difference between a Treasury rating as
a Treasury and a Treasury rating as a corporate bond with no financials.
"""

from collections import namedtuple

from scripts.config import (
    MIN_PEER_SCORING,
    RATING_THRESHOLD_BUY, RATING_THRESHOLD_LEAN, RATING_THRESHOLD_PASS,
    RATING_THRESHOLDS_BY_CLASS,
)

# Row key used to pool rows for relative_mode='peer'. Equity used 'sector'.
PEER_KEY = 'peer_group'


# ---------------------------------------------------------------------------
# Gate spec
# ---------------------------------------------------------------------------

class Gate(namedtuple('Gate',
                      ['name', 'field', 'test_fn', 'score_fn',
                       'relative_mode', 'higher_better', 'weight',
                       'applicable'],
                      defaults=(False, True, 1.0, None))):
    """One metric, driving both the pass/fail matrix cell and the 0-100 score.

    test_fn:        (value, row) -> bool | None   (None = missing, renders N/A)
    score_fn:       (value, row, percentile_or_None) -> 0-100 | None
    relative_mode:  False = absolute; 'global' or 'peer' = percentile-ranked
    higher_better:  ranking direction (relative gates only)
    weight:         contribution within its category (default 1.0)
    applicable:     optional row-predicate. False => structurally inapplicable
                    for this row: excluded from numerator AND denominator.
                    None => always applicable.

    The category is the name's prefix: 'Valuation: Spread vs Fair' -> 'Valuation'.
    """
    __slots__ = ()

    @property
    def category(self):
        return self.name.split(': ')[0]


class ScoringSpec(namedtuple('ScoringSpec',
                             ['gates', 'category_weights', 'category_order',
                              'gate_display', 'category_display',
                              'prepare_fn', 'cap_fn',
                              'uncertainty_field', 'uncertainty_tiers'],
                             defaults=(None, None, None, None, None, None))):
    """Everything asset-class-specific the kernel needs, in one bundle.

    gates:            list[Gate]
    category_weights: {category_name: (param_key, default_weight)}
    category_order:   display order for gate_metadata (defaults to gates order)
    gate_display:     {gate_short: {label, threshold, fmt}}
    category_display: {category_name: {dark, light}}
    prepare_fn:       optional (results) -> None, derives shared fields before
                      gates run (the equivalent of upstream prepare_scoring_fields)
    cap_fn:           optional (row, params) -> (cap_rating | None, [reasons])
    uncertainty_field/uncertainty_tiers: optional composite haircut, as
                      [(threshold, multiplier), ...] highest threshold first.
    """
    __slots__ = ()


RATING_RANK = {'PASS': 0, 'HOLD': 1, 'LEAN BUY': 2, 'BUY': 3}
RATING_BY_RANK = {v: k for k, v in RATING_RANK.items()}


# ---------------------------------------------------------------------------
# Field-key helpers
# ---------------------------------------------------------------------------

def _gate_short(gate_name):
    """Return the stable suffix used by _gate_* / _gp_* / _score_* fields."""
    return gate_name.split(': ')[1].lower().replace(' ', '_').replace('/', '_')


def _gate_key(gate_name):
    return '_gate_' + _gate_short(gate_name)


def _gp_key(gate_name):
    return '_gp_' + _gate_short(gate_name)


def _score_key(gate_name):
    return '_score_' + _gate_short(gate_name)


def _gate_applicable(gate, row):
    """True unless the gate declares an applicability predicate that fails."""
    return gate.applicable is None or gate.applicable(row)


def _cap_rating(rating, cap):
    """Return rating capped at cap, preserving None/unknown inputs."""
    if rating not in RATING_RANK or cap not in RATING_RANK:
        return rating
    return RATING_BY_RANK[min(RATING_RANK[rating], RATING_RANK[cap])]


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def _score_linear(value, worst, best):
    """Map value linearly from [worst, best] to [0, 100], clamped.

    `worst` may exceed `best` for lower-is-better metrics; the clamp handles
    both directions. Returns None for a missing input, so a caller can tell
    "missing" from "scored zero".

    NaN IS TREATED AS MISSING, and this is load-bearing rather than tidiness.
    Missing values read out of a parquet column arrive as float('nan'), not
    None — `nan is None` is False, so a NaN sails past every guard. It then
    reaches the clamp, and because every NaN comparison is False,
    `min(100.0, nan)` returns 100.0: a missing metric scores PERFECT.

    That is not hypothetical. Bonds whose issuer could not be identified — no
    coverage, no leverage, no Altman-Z, nothing — came back from the credit
    scorecard as AAA with full confidence, because all six missing factors
    scored 100. Roughly 6,000 of them. A silent inversion like this is far
    worse than a crash: the model was most confident exactly where it knew
    least.
    """
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:                      # NaN
        return None
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100
    return max(0.0, min(100.0, score))


def _ranked_percentiles(items, higher_better=True):
    """Assign average-rank percentiles so equal values receive equal scores.

    Args:
        items: list of (row_index, value) pairs.
        higher_better: whether higher raw values earn higher percentiles.

    Returns:
        dict: {row_index: percentile_0_to_100}
    """
    if not items:
        return {}
    sorted_items = sorted(items, key=lambda x: x[1])
    n = len(sorted_items)
    if n == 1:
        return {sorted_items[0][0]: 50.0}

    out = {}
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_items[j][1] == sorted_items[i][1]:
            j += 1
        avg_rank = (i + (j - 1)) / 2.0
        pctile = (avg_rank / (n - 1)) * 100
        if not higher_better:
            pctile = 100 - pctile
        for k in range(i, j):
            out[sorted_items[k][0]] = pctile
        i = j
    return out


# ---------------------------------------------------------------------------
# Pass/fail screening matrix
# ---------------------------------------------------------------------------

def apply_screening_matrix(results, spec):
    """Evaluate each row against the pass/fail gates.

    Writes per-gate raw values (_gate_*) and pass/fail booleans (_gp_*, with
    None rendering N/A), plus the _gates_passed diagnostic.

    The denominator counts APPLICABLE gates only. Structurally inapplicable
    gates count nowhere; missing data counts as a fail but stays in the
    denominator.
    """
    if spec.prepare_fn is not None:
        spec.prepare_fn(results)

    for r in results:
        passed = 0
        applicable_total = 0
        inapplicable = 0
        for gate in spec.gates:
            gate_key = _gate_key(gate.name)
            gp_key = _gp_key(gate.name)
            if not _gate_applicable(gate, r):
                r[gate_key] = None
                r[gp_key] = None
                inapplicable += 1
                continue
            applicable_total += 1
            val = r.get(gate.field)
            result = gate.test_fn(val, r)
            if result is None:
                # Missing data — renders N/A but counts as a fail
                # (passed not incremented; still in the denominator).
                r[gate_key] = None
                r[gp_key] = None
            else:
                r[gate_key] = val
                r[gp_key] = bool(result)
                if result:
                    passed += 1

        r['_gates_passed'] = f'{passed}/{applicable_total}'
        r['_gates_passed_num'] = passed
        r['_gates_applicable'] = applicable_total
        r['_gates_inapplicable'] = inapplicable


# ---------------------------------------------------------------------------
# Continuous scoring
# ---------------------------------------------------------------------------

def compute_continuous_scores(results, spec, params=None):
    """Score every row on all gates, then roll up to categories and composite.

    Three scoring modes: absolute, 'global' percentile, and 'peer' percentile
    (pooled on PEER_KEY, falling back to the global pool below
    MIN_PEER_SCORING members).

    The category roll-up is where the applicable/missing distinction pays off,
    and it is why a Treasury can share a rating scale with a corporate bond at
    all: when every gate in a category is structurally inapplicable, that
    category's applicable weight is 0, its average is None, and it drops out
    of the composite entirely — the surviving categories renormalise over
    their own weights. Upstream marks this path "unreachable with current
    masks — defensive only". Here it is the normal path for every Treasury.
    """
    # Derived fields must exist before any gate reads them. Both this and
    # apply_screening_matrix call prepare_fn rather than one depending on the
    # other having run: an applicability predicate reading a field that
    # prepare_fn had not yet computed would not error, it would silently
    # return False and mask the gate — a whole category quietly vanishing
    # because two functions were called in the wrong order. prepare_fn is
    # required to be idempotent (tested), so the extra pass is safe.
    if spec.prepare_fn is not None:
        spec.prepare_fn(results)

    # Step 1: percentile ranks for relative gates.
    for gate in spec.gates:
        if not gate.relative_mode:
            continue
        gate_name, field, higher_better = gate.name, gate.field, gate.higher_better
        relative_mode = gate.relative_mode

        # Rows where the gate is structurally inapplicable stay out of the
        # ranking pools — otherwise Treasuries would drag corporate spread
        # percentiles around.
        all_vals = [(i, r.get(field), r.get(PEER_KEY) or '_unknown')
                    for i, r in enumerate(results)
                    if r.get(field) is not None and _gate_applicable(gate, r)]
        if len(all_vals) < 2:
            continue

        pctile_key = f'{gate_name}_{field}'

        if relative_mode == 'peer':
            peer_groups = {}
            for i, val, peer in all_vals:
                peer_groups.setdefault(peer, []).append((i, val))

            for _peer, group in peer_groups.items():
                pool = (group if len(group) >= MIN_PEER_SCORING
                        else [(i, v) for i, v, _ in all_vals])
                pctiles = _ranked_percentiles(pool, higher_better=higher_better)
                for orig_idx, _ in group:
                    results[orig_idx].setdefault('_pctile', {})[pctile_key] = \
                        pctiles[orig_idx]

        else:  # 'global'
            pctiles = _ranked_percentiles([(i, v) for i, v, _ in all_vals],
                                          higher_better=higher_better)
            for orig_idx, pctile in pctiles.items():
                results[orig_idx].setdefault('_pctile', {})[pctile_key] = pctile

    # Step 2: gate scores, category averages, composite.
    p = params or {}
    cat_weights = {
        cat: p.get(param_key, default)
        for cat, (param_key, default) in spec.category_weights.items()
    }
    all_categories = []
    for gate in spec.gates:
        if gate.category not in all_categories:
            all_categories.append(gate.category)

    for r in results:
        cat_score_sums = {cat: 0.0 for cat in all_categories}
        cat_weight_sums = {cat: 0.0 for cat in all_categories}
        applicable_gates = 0
        covered_gates = 0
        for gate in spec.gates:
            if not _gate_applicable(gate, r):
                # Score stays None so matrix cells render N/A rather than a
                # misleading 0.0.
                r[_score_key(gate.name)] = None
                continue
            applicable_gates += 1
            val = r.get(gate.field)
            if val is None:
                score = 0.0          # counts as worst, stays in denominator
            else:
                covered_gates += 1
                pct = (r.get('_pctile', {}).get(f'{gate.name}_{gate.field}', 50)
                       if gate.relative_mode else None)
                s = gate.score_fn(val, r, pct)
                score = s if s is not None else 0.0
            r[_score_key(gate.name)] = round(score, 1)
            cat_score_sums[gate.category] += score * gate.weight
            cat_weight_sums[gate.category] += gate.weight

        # A category with zero applicable weight scores None and drops out.
        cat_avgs = {
            cat: (cat_score_sums[cat] / cat_weight_sums[cat]
                  if cat_weight_sums[cat] > 0 else None)
            for cat in all_categories
        }
        for cat in all_categories:
            key = '_score_' + cat.lower()
            r[key] = round(cat_avgs[cat], 1) if cat_avgs[cat] is not None else None

        weighted_sum = 0.0
        weight_total = 0.0
        for cat in all_categories:
            if cat_avgs[cat] is None:
                continue
            w = cat_weights.get(cat, 0)
            weighted_sum += cat_avgs[cat] * w
            weight_total += w
        composite = weighted_sum / weight_total if weight_total > 0 else None

        r['_composite_score_raw'] = round(composite, 1) if composite is not None else None
        r['_composite_categories'] = [c for c in all_categories
                                      if cat_avgs[c] is not None]

        # Optional uncertainty haircut (equity used Monte-Carlo CV here).
        if composite is not None and spec.uncertainty_field:
            cv = r.get(spec.uncertainty_field)
            if cv is not None:
                for threshold, mult in (spec.uncertainty_tiers or ()):
                    if cv > threshold:
                        composite *= mult
                        break

        r['_composite_score'] = round(composite, 1) if composite is not None else None
        # Coverage over applicable gates only: a Treasury must not read as
        # low-coverage because credit gates that cannot describe it are absent.
        r['_data_coverage_score'] = (
            round(covered_gates / applicable_gates * 100, 1)
            if applicable_gates > 0 else None)

        r.pop('_pctile', None)


# ---------------------------------------------------------------------------
# Rating
# ---------------------------------------------------------------------------

def rating_from_composite(composite, params=None, asset_class=None):
    """Map a 0-100 composite to a rating bucket.

      BUY       composite >= buy threshold
      LEAN BUY  composite >= lean threshold
      HOLD      composite >= pass threshold
      PASS      below that

    Per-asset-class thresholds win over the base ones when present, because
    Treasury and corporate composites are computed over different category
    sets and are not directly comparable. Order of precedence:
      params['rating_threshold_buy_<class>']
      -> config.RATING_THRESHOLDS_BY_CLASS[<class>]['buy']
      -> params['rating_threshold_buy']
      -> config.RATING_THRESHOLD_BUY

    Returns None when composite is None.
    """
    if composite is None:
        return None
    p = params or {}
    cls_defaults = RATING_THRESHOLDS_BY_CLASS.get(asset_class or '', {})

    def threshold(name, base_default):
        # A present-but-None override counts as UNSET, not as a threshold of
        # None. default_params() seeds every per-class key to None so the full
        # keyspace is visible to a calibration sweep; keying on `in` here would
        # read those as real thresholds and blow up on the comparison.
        if asset_class:
            key = f'rating_threshold_{name}_{asset_class.lower()}'
            if p.get(key) is not None:
                return p[key]
            if cls_defaults.get(name) is not None:
                return cls_defaults[name]
        base = p.get(f'rating_threshold_{name}')
        return base if base is not None else base_default

    if composite >= threshold('buy', RATING_THRESHOLD_BUY):
        return 'BUY'
    if composite >= threshold('lean', RATING_THRESHOLD_LEAN):
        return 'LEAN BUY'
    if composite >= threshold('pass', RATING_THRESHOLD_PASS):
        return 'HOLD'
    return 'PASS'


def apply_rating_caps(results, spec, params=None):
    """Apply investability caps and expose raw/final rating diagnostics.

    rating_raw is the uncapped composite-derived rating; rating is the final
    (possibly capped) one. rating != rating_raw <=> a cap fired — the report
    keys its warning badge off that comparison plus _rating_cap_reasons.
    """
    for r in results:
        raw = rating_from_composite(r.get('_composite_score'), params,
                                    asset_class=r.get('asset_class'))
        r['rating_raw'] = raw
        if spec.cap_fn is not None:
            cap, reasons = spec.cap_fn(r, params)
        else:
            cap, reasons = None, []
        r['_rating_cap'] = cap
        r['_rating_cap_reasons'] = reasons
        r['rating'] = _cap_rating(raw, cap) if cap and raw else raw


# ---------------------------------------------------------------------------
# Report metadata + snapshot hygiene
# ---------------------------------------------------------------------------

def gate_metadata(spec, params=None):
    """Return report metadata derived from the active gate definitions.

    The HTML report's "why the rating changed" bullets are driven entirely by
    this, so a new GATES list produces correct explanations for free.
    """
    p = params or {}
    gate_display = spec.gate_display or {}
    gates = []
    for gate in spec.gates:
        short = _gate_short(gate.name)
        display = gate_display.get(short, {})
        gates.append({
            'key': _gate_key(gate.name),
            'label': display.get('label', gate.name.split(': ')[1]),
            'gpKey': _gp_key(gate.name),
            'scoreKey': _score_key(gate.name),
            'threshold': display.get('threshold', ''),
            'category': gate.category,
            'fmt': display.get('fmt', 'ratio'),
            'weight': gate.weight,
        })

    order = spec.category_order or list(spec.category_weights.keys())
    cat_display = spec.category_display or {}
    categories = []
    for name in order:
        param_key, default = spec.category_weights[name]
        display = cat_display.get(name, {})
        categories.append({
            'name': name,
            'weight': p.get(param_key, default),
            'dark': display.get('dark', '#555555'),
            'light': display.get('light', '#DDDDDD'),
            'scoreKey': '_score_' + name.lower(),
        })
    return {'gates': gates, 'categories': categories}


def _purge_stale_gate_fields(results, spec):
    """Drop _gate_*/_gp_*/_score_* keys for gates that no longer exist.

    Snapshot rows round-trip through re-scoring: without this, fields from
    retired gates persist forever and leak into the report payloads.
    """
    valid = set()
    for g in spec.gates:
        valid.add(_gate_key(g.name))
        valid.add(_gp_key(g.name))
        valid.add(_score_key(g.name))
        valid.add('_score_' + g.category.lower())   # category averages
    for r in results:
        stale = [k for k in r
                 if (k.startswith('_gate_') or k.startswith('_gp_') or
                     k.startswith('_score_')) and k not in valid]
        for k in stale:
            del r[k]


def score_and_rate(results, spec, params=None):
    """The canonical scoring workflow, shared by live, replay and rescore."""
    _purge_stale_gate_fields(results, spec)
    apply_screening_matrix(results, spec)
    compute_continuous_scores(results, spec, params=params)
    apply_rating_caps(results, spec, params=params)
    return results
