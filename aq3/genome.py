"""
aq3.genome — typed expression-tree representation of a BRAIN alpha.

The core bet of aq3: an alpha is NOT a free-form string. It is a typed tree
built only from validated catalog primitives. Every well-typed tree serializes
to syntactically- and semantically-valid FastExpr, so the entire class of
failures that dominated v1/v2 — hallucinated fields, wrong operator arity,
ranking a VECTOR field, missing window args — is *structurally impossible to
represent*. You cannot build a broken genome.

Type system
-----------
    SIGNAL  — a per-stock numeric vector (the thing alphas are made of)
    WINDOW  — an integer day-count (e.g. 5, 21, 126); only valid as a ts_* arg
    GROUP   — a grouping field (subindustry / sector / ...); only a group-op arg

Leaves are catalog fields (SIGNAL) drawn from a curated, quality-filtered pool
(aq3/primitives.json — MATRIX-typed, real coverage, proven by other quants).
Internal nodes are operators whose arg *shapes* are parsed from operators.json.

Public API
----------
    Genome.random(rng, max_depth)        -> Genome
    Genome.from_fastexpr(expr)           -> Genome | None   (seed from a winner)
    g.to_fastexpr()                      -> str
    g.mutate(rng)                        -> Genome          (returns a NEW genome)
    Genome.crossover(a, b, rng)          -> Genome
    g.depth(), g.size(), g.fields()      -> structural stats
    g.signature()                        -> str             (dedup key)
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_PRIM_PATH = _HERE / "primitives.json"


# --------------------------------------------------------------------------- #
# Primitive tables (loaded once)
# --------------------------------------------------------------------------- #
def _load_primitives() -> dict:
    return json.loads(_PRIM_PATH.read_text())


_PRIM = _load_primitives()
OP_SHAPES: dict[str, list[str]] = _PRIM["op_shapes"]      # op -> ['S','W',...]
FIELDS: dict[str, dict] = _PRIM["fields"]                 # field -> {cat, cov}
GROUPS: list[str] = _PRIM["groups"]
WINDOWS: list[int] = _PRIM["windows"]
CONSTS: list[float] = _PRIM.get("consts", [0.5, 2.0])

# signed_power's 2nd argument is a numeric exponent, not a signal. Re-shape it
# so the grammar treats that slot as a CONST, never a SIGNAL subtree.
if "signed_power" in OP_SHAPES:
    OP_SHAPES["signed_power"] = ["S", "C"]

# `negate` is a synthetic UNARY signal op (sign flip of an inner signal). It is
# NOT a BRAIN operator — it serializes to `subtract(0, x)`. Modeling negation as
# a first-class unary node (rather than a literal subtract(0,x) subtree) keeps a
# stray `const 0` out of the tree, so op-swap can never turn `-x` into
# `multiply(0, x)` (= a dead leg). Shape [S] means it only ever swaps with other
# unary signal ops (rank/zscore/scale/normalize), never producing a zero leg.
OP_SHAPES["negate"] = ["S"]

FIELD_NAMES: list[str] = list(FIELDS.keys())

# Operators partitioned by what they RETURN-shape lets us nest. All of ours
# return SIGNAL, so any op can wrap any signal. We split by arg pattern for
# generation convenience.
_UNARY_S = [op for op, sh in OP_SHAPES.items() if sh == ["S"]]
_TS_WINDOW = [op for op, sh in OP_SHAPES.items() if sh == ["S", "W"]]
_BINARY_S = [op for op, sh in OP_SHAPES.items() if sh == ["S", "S"]]
_TS_BINARY_W = [op for op, sh in OP_SHAPES.items() if sh == ["S", "S", "W"]]
_GROUP_OPS = [op for op, sh in OP_SHAPES.items() if sh == ["S", "G"]]
_POWER_OPS = [op for op, sh in OP_SHAPES.items() if sh == ["S", "C"]]

# Operators that bound/normalize a signal cross-sectionally. A finished alpha
# should be wrapped in one of these so weights are sane.
_BOUNDING_OPS = [op for op in ("rank", "zscore", "scale") if op in OP_SHAPES]


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    """One node in the typed tree. kind ∈ {field, window, group, op}."""
    kind: str
    value: Any                       # field name / int / group name / op name
    children: list["Node"] = dc_field(default_factory=list)

    # ---- serialization ----
    def to_fastexpr(self) -> str:
        if self.kind == "field":
            return str(self.value)
        if self.kind == "window":
            return str(self.value)
        if self.kind == "group":
            return str(self.value)
        if self.kind == "const":
            return str(self.value)
        if self.kind == "op":
            if self.value == "negate":          # synthetic unary → valid FastExpr
                return f"subtract(0, {self.children[0].to_fastexpr()})"
            args = ", ".join(c.to_fastexpr() for c in self.children)
            return f"{self.value}({args})"
        raise ValueError(f"bad node kind {self.kind!r}")

    # ---- structural ----
    def depth(self) -> int:
        if self.kind != "op" or not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def iter_signal_nodes(self):
        """Yield every node that represents a SIGNAL subtree (fields + signal-ops)."""
        if self.kind == "field" or (self.kind == "op" and _returns_signal(self)):
            yield self
        for c in self.children:
            yield from c.iter_signal_nodes()

    def fields(self) -> set[str]:
        out: set[str] = set()
        if self.kind == "field":
            out.add(str(self.value))
        for c in self.children:
            out |= c.fields()
        return out

    def clone(self) -> "Node":
        return Node(self.kind, self.value, [c.clone() for c in self.children])


def _returns_signal(node: Node) -> bool:
    # All ops in our pool return SIGNAL.
    return node.kind == "op"


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _rand_field(rng: random.Random, cat: str | None = None) -> Node:
    if cat:
        pool = [f for f, m in FIELDS.items() if m["cat"] == cat] or FIELD_NAMES
    else:
        pool = FIELD_NAMES
    return Node("field", rng.choice(pool))


def _rand_window(rng: random.Random) -> Node:
    return Node("window", rng.choice(WINDOWS))


def _rand_group(rng: random.Random) -> Node:
    # Bias toward subindustry (tightest neutralization, best in our data).
    weighted = ["subindustry", "subindustry", "sector"]
    return Node("group", rng.choice(weighted))


def _rand_const(rng: random.Random) -> Node:
    return Node("const", rng.choice(CONSTS))


def _rand_signal(rng: random.Random, max_depth: int) -> Node:
    """Build a random SIGNAL subtree of at most `max_depth`."""
    if max_depth <= 1:
        return _rand_field(rng)

    # Choose an operator class. Probabilities favor the structures that recur
    # in real winners (ts smoothing, arithmetic combination).
    roll = rng.random()
    if roll < 0.30 and _TS_WINDOW:
        op = rng.choice(_TS_WINDOW)
        return Node("op", op, [_rand_signal(rng, max_depth - 1), _rand_window(rng)])
    if roll < 0.55 and _BINARY_S:
        op = rng.choice(_BINARY_S)
        return Node("op", op, [_rand_signal(rng, max_depth - 1),
                               _rand_signal(rng, max_depth - 1)])
    if roll < 0.70 and _UNARY_S:
        op = rng.choice(_UNARY_S)
        return Node("op", op, [_rand_signal(rng, max_depth - 1)])
    if roll < 0.80 and _TS_BINARY_W:
        op = rng.choice(_TS_BINARY_W)
        return Node("op", op, [_rand_signal(rng, max_depth - 1),
                               _rand_signal(rng, max_depth - 1), _rand_window(rng)])
    if roll < 0.85 and _POWER_OPS:
        op = rng.choice(_POWER_OPS)
        return Node("op", op, [_rand_signal(rng, max_depth - 1), _rand_const(rng)])
    # default: a bare field leaf
    return _rand_field(rng)


def _finalize(inner: Node, rng: random.Random) -> Node:
    """Wrap a raw signal into a bounded, neutralized alpha:
        group_neutralize(<bound>(inner), group)
    This mirrors the canonical winning shape and guarantees sane weights."""
    bound_op = rng.choice(_BOUNDING_OPS) if _BOUNDING_OPS else "rank"
    bounded = Node("op", bound_op, [inner])
    if _GROUP_OPS:
        return Node("op", "group_neutralize", [bounded, _rand_group(rng)])
    return bounded


# --------------------------------------------------------------------------- #
# Genome — a PROTECTED-ENVELOPE alpha.
#
# Every genome serializes to:   [-1 *] group_neutralize( bound_op( inner ), group )
#
# The envelope (sign, bound_op, group) and the requirement that the signal is
# always bounded + group-neutralized are INVARIANT. Evolution mutates only the
# `inner` signal subtree plus the three envelope knobs — it can never produce a
# raw-field-outside-neutralization malformation (the failure the first GP run
# surfaced). This guarantees every individual is a sane alpha, not just valid
# syntax, so the surrogate is never scored on garbage.
# --------------------------------------------------------------------------- #
_ENVELOPE_GROUPS = ["subindustry", "subindustry", "sector"]


@dataclass
class Genome:
    inner: Node                 # the raw signal subtree (no envelope)
    bound_op: str = "rank"      # rank | zscore | scale
    group: str = "subindustry"  # subindustry | sector
    sign: int = 1               # +1 or -1

    # ---- construction ----
    @classmethod
    def random(cls, rng: random.Random, max_depth: int = 4) -> "Genome":
        return cls(
            inner=_rand_signal(rng, max_depth),
            bound_op=rng.choice(_BOUNDING_OPS) if _BOUNDING_OPS else "rank",
            group=rng.choice(_ENVELOPE_GROUPS),
            sign=rng.choice([1, 1, 1, -1]),     # mostly long-form; sign is a cheap mutation
        )

    # ---- serialization ----
    def to_fastexpr(self) -> str:
        bounded = Node("op", self.bound_op, [self.inner])
        core = Node("op", "group_neutralize", [bounded, Node("group", self.group)])
        s = core.to_fastexpr()
        return f"-1 * {s}" if self.sign < 0 else s

    # ---- stats ----
    def depth(self) -> int:
        return self.inner.depth() + 2

    def size(self) -> int:
        return self.inner.size() + 2

    def fields(self) -> set[str]:
        return self.inner.fields()

    def signature(self) -> str:
        """Structural signature for dedup. Includes the envelope knobs + the
        inner skeleton with field categories abstracted, so two alphas differing
        only in which fundamental field they use collapse together."""
        def sig(n: Node) -> str:
            if n.kind == "field":
                return FIELDS.get(str(n.value), {}).get("cat", "fld")
            if n.kind == "window":
                return "w"
            if n.kind == "const":
                return "c"
            if n.kind in ("group",):
                return "g"
            return f"{n.value}(" + ",".join(sig(c) for c in n.children) + ")"
        return f"{self.sign}|{self.bound_op}|{self.group}|{sig(self.inner)}"

    def clone(self) -> "Genome":
        return Genome(self.inner.clone(), self.bound_op, self.group, self.sign)

    # ---- mutation (envelope-preserving) ----
    def mutate(self, rng: random.Random, max_depth: int = 5) -> "Genome":
        """Return a NEW genome with one random mutation. The envelope is always
        preserved — mutations touch the sign, bound_op, group, or the inner tree."""
        g = self.clone()
        kind = rng.choices(
            ["sign", "bound", "group", "inner"], weights=[1, 1, 1, 6], k=1
        )[0]

        if kind == "sign":
            g.sign = -g.sign
            return g
        if kind == "bound" and _BOUNDING_OPS:
            g.bound_op = rng.choice([o for o in _BOUNDING_OPS if o != g.bound_op] or _BOUNDING_OPS)
            return g
        if kind == "group":
            g.group = rng.choice([gr for gr in ("subindustry", "sector") if gr != g.group])
            return g

        # inner-tree mutation: window jitter / field swap / op swap / subtree regrow
        nodes: list[Node] = []
        _collect(g.inner, nodes)
        sub = rng.choice(["window", "field", "op", "subtree"])

        if sub == "window":
            wins = [n for n in nodes if n.kind == "window"]
            if wins:
                n = rng.choice(wins)
                i = WINDOWS.index(n.value) if n.value in WINDOWS else rng.randrange(len(WINDOWS))
                j = min(max(i + rng.choice([-1, 1]), 0), len(WINDOWS) - 1)
                n.value = WINDOWS[j]
                return g
        if sub == "field":
            flds = [n for n in nodes if n.kind == "field"]
            if flds:
                n = rng.choice(flds)
                cat = FIELDS.get(str(n.value), {}).get("cat")
                pool = [f for f, m in FIELDS.items() if m["cat"] == cat] or FIELD_NAMES
                n.value = rng.choice(pool)
                return g
        if sub == "op":
            ops = [n for n in nodes if n.kind == "op"]
            if ops:
                n = rng.choice(ops)
                shape = OP_SHAPES.get(n.value)
                sibs = [o for o, s in OP_SHAPES.items() if s == shape and o != n.value]
                if sibs:
                    n.value = rng.choice(sibs)
                    return g
        # subtree regrow: replace a child of an inner op, or regrow whole inner
        op_nodes = [n for n in nodes if n.kind == "op" and n.children]
        if op_nodes:
            target = rng.choice(op_nodes)
            ci = rng.randrange(len(target.children))
            if target.children[ci].kind in ("field", "op"):
                target.children[ci] = _rand_signal(rng, max(2, max_depth - 2))
        else:
            g.inner = _rand_signal(rng, max_depth)
        return g

    # ---- crossover (envelope-preserving) ----
    @classmethod
    def crossover(cls, a: "Genome", b: "Genome", rng: random.Random) -> "Genome":
        """Child inherits a's envelope and a copy of a's inner, with a random
        signal subtree replaced by one donated from b's inner. Signals only ever
        replace signals; the envelope is never touched."""
        child = a.clone()
        a_nodes: list[Node] = []
        _collect(child.inner, a_nodes)
        a_signals = [n for n in a_nodes if n.kind == "op" and n.children]
        b_signals = list(b.inner.iter_signal_nodes())
        if not b_signals:
            return child
        if not a_signals:
            # a's inner is a bare field — donate b's whole inner subtree
            child.inner = rng.choice(b_signals).clone()
            return child
        target = rng.choice(a_signals)
        donor = rng.choice(b_signals).clone()
        ci = rng.randrange(len(target.children))
        if target.children[ci].kind in ("field", "op"):
            target.children[ci] = donor
        return child

    # ---- seeding from an existing expression ----
    @classmethod
    def from_fastexpr(cls, expr: str) -> "Genome | None":
        """Parse a known-good FastExpr string into a protected-envelope genome.
        Strips any existing [-1*]/group_neutralize/bound wrapper to recover the
        inner signal; returns None if the expression is outside the typed grammar."""
        try:
            node, rest = _parse(expr.strip())
            if rest.strip():
                return None
        except (_ParseError, RecursionError):
            return None
        return cls._from_node(node)

    @classmethod
    def _from_node(cls, node: Node) -> "Genome | None":
        sign = 1
        # peel a leading  -1 * X   (parsed as subtract(0,1)*X or multiply(subtract(0,1),X))
        node, sign = _peel_sign(node, sign)
        bound_op, group = "rank", "subindustry"
        # peel group_neutralize(X, g)
        if node.kind == "op" and node.value in _GROUP_OPS and len(node.children) == 2:
            grp = node.children[1]
            if grp.kind in ("group", "field") and str(grp.value) in ("subindustry", "sector", "industry"):
                group = "sector" if str(grp.value) == "sector" else "subindustry"
            node = node.children[0]
        # peel a bounding op
        if node.kind == "op" and node.value in _BOUNDING_OPS and len(node.children) == 1:
            bound_op = node.value
            node = node.children[0]
        # whatever remains is the inner signal; it must be a signal node
        if node.kind not in ("field", "op"):
            return None
        return cls(inner=node, bound_op=bound_op, group=group, sign=sign)


def _peel_sign(node: Node, sign: int) -> tuple[Node, int]:
    """Strip a leading unary minus: negate(x), multiply(-1, x), or subtract(0,x)."""
    if node.kind == "op" and node.value == "negate" and len(node.children) == 1:
        return _peel_sign(node.children[0], -sign)
    if node.kind == "op" and node.value == "subtract" and len(node.children) == 2:
        a, b = node.children
        if a.kind in ("const", "window") and float(a.value) == 0:
            return _peel_sign(b, -sign)
    if node.kind == "op" and node.value == "multiply" and len(node.children) == 2:
        a, b = node.children
        # multiply(-1, X) == -1 * X
        if a.kind in ("const", "window") and float(a.value) == -1:
            return _peel_sign(b, -sign)
        if b.kind in ("const", "window") and float(b.value) == -1:
            return _peel_sign(a, -sign)
    return node, sign


def _collect(node: Node, out: list[Node]) -> None:
    out.append(node)
    for c in node.children:
        _collect(c, out)


# --------------------------------------------------------------------------- #
# Minimal recursive-descent parser for the typed FastExpr subset
# --------------------------------------------------------------------------- #
class _ParseError(Exception):
    pass


_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_NUMTOK = re.compile(r"\d+(?:\.\d+)?")

# Infix arithmetic mapped to the function-form operators in our grammar.
_INFIX = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}


def _parse(s: str) -> tuple[Node, str]:
    """Parse a full signal expression, including infix +,-,*,/ with precedence."""
    return _parse_addsub(s)


def _parse_addsub(s: str) -> tuple[Node, str]:
    node, s = _parse_muldiv(s)
    s = s.lstrip()
    while s[:1] in ("+", "-"):
        op = _INFIX[s[0]]
        rhs, s = _parse_muldiv(s[1:])
        node = Node("op", op, [node, rhs])
        s = s.lstrip()
    return node, s


def _parse_muldiv(s: str) -> tuple[Node, str]:
    node, s = _parse_atom(s)
    s = s.lstrip()
    while s[:1] in ("*", "/"):
        op = _INFIX[s[0]]
        rhs, s = _parse_atom(s[1:])
        node = Node("op", op, [node, rhs])
        s = s.lstrip()
    return node, s


def _parse_atom(s: str) -> tuple[Node, str]:
    s = s.lstrip()
    # unary minus: -x  ->  negate(x)  (first-class unary, no stray const 0)
    if s.startswith("-"):
        inner, rest = _parse_atom(s[1:])
        # -<number> stays a numeric literal (e.g. signed_power exponent); else negate
        if inner.kind in ("window", "const"):
            return Node("const", -float(inner.value)), rest
        return Node("op", "negate", [inner]), rest
    if s.startswith("("):
        node, rest = _parse_addsub(s[1:])
        rest = rest.lstrip()
        if not rest.startswith(")"):
            raise _ParseError("expected )")
        return node, rest[1:]
    mnum = _NUMTOK.match(s)
    if mnum:
        tok = mnum.group(0)
        rest = s[mnum.end():]
        val = float(tok) if "." in tok else int(tok)
        # tag as window if integer; parent _coerce reinterprets as const where needed
        return Node("window" if isinstance(val, int) else "const", val), rest
    m = _IDENT.match(s)
    if not m:
        raise _ParseError(f"expected atom at {s[:20]!r}")
    name = m.group(0)
    rest = s[m.end():].lstrip()
    if rest.startswith("("):
        if name not in OP_SHAPES:
            raise _ParseError(f"unknown op {name}")
        args, rest = _parse_args(rest[1:])
        # negation idiom: subtract(0, x) -> negate(x)  (collapse before coercion,
        # so the 0 never has to coerce to a signal)
        if (name == "subtract" and len(args) == 2
                and args[0].kind in ("window", "const") and float(args[0].value) == 0):
            return Node("op", "negate", [_coerce(args[1], "S")]), rest
        shape = OP_SHAPES[name]
        if len(args) != len(shape):
            raise _ParseError(f"{name} arity {len(args)} != {len(shape)}")
        coerced = [_coerce(a, k) for a, k in zip(args, shape)]
        return Node("op", name, coerced), rest
    if name in GROUPS:
        return Node("group", name), rest
    return Node("field", name), rest


def _parse_args(s: str) -> tuple[list[Node], str]:
    args: list[Node] = []
    s = s.lstrip()
    if s.startswith(")"):
        return args, s[1:]
    while True:
        node, s = _parse_addsub(s)
        s = s.lstrip()
        if s.startswith(","):
            args.append(node)
            s = s[1:].lstrip()
            continue
        if s.startswith(")"):
            args.append(node)
            return args, s[1:]
        raise _ParseError(f"expected , or ) at {s[:20]!r}")


def _coerce(node: Node, kind: str) -> Node:
    if kind == "W":
        if node.kind == "window":
            return node
        if node.kind in ("const",) and float(node.value).is_integer():
            return Node("window", int(node.value))
        raise _ParseError("expected window")
    if kind == "C":
        if node.kind in ("const", "window"):
            return Node("const", node.value)
        raise _ParseError("expected const exponent")
    if kind == "G":
        if node.kind in ("group", "field") and str(node.value) in GROUPS:
            return Node("group", str(node.value))
        raise _ParseError("expected group")
    # S: a signal — field or op
    if node.kind in ("field", "op"):
        return node
    raise _ParseError("expected signal")


# --------------------------------------------------------------------------- #
# Self-test: the headline claim — random genomes are ALWAYS valid FastExpr
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    import sys
    sys.path.insert(0, str(_HERE.parent))
    from math_engine import MathEngine

    me = MathEngine()
    rng = random.Random(0)

    n = 1000
    valid = 0
    fail_examples = []
    sigs = set()
    for _ in range(n):
        g = Genome.random(rng, max_depth=rng.choice([3, 4, 5]))
        expr = g.to_fastexpr()
        sigs.add(g.signature())
        crit = me.critique(expr, {"neutralization": "Subindustry"})
        if crit.get("verdict") != "FAIL":
            valid += 1
        elif len(fail_examples) < 5:
            fail_examples.append((expr, crit))

    print(f"[genome] {valid}/{n} random genomes pass math_engine.critique (non-FAIL)")
    print(f"[genome] {len(sigs)} distinct structural signatures in {n} samples")
    if fail_examples:
        print("[genome] FAIL examples (structural grammar gaps to close):")
        for expr, crit in fail_examples:
            issues = [i.get('code') if isinstance(i, dict) else getattr(i, 'code', '?')
                      for i in crit.get('issues', [])]
            print(f"    {expr[:90]}")
            print(f"      -> {issues}")

    # Mutation + crossover smoke test
    a = Genome.random(rng, 4)
    b = Genome.random(rng, 4)
    m = a.mutate(rng)
    x = Genome.crossover(a, b, rng)
    assert m.to_fastexpr() and x.to_fastexpr()
    print(f"[genome] mutate/crossover OK")
    print(f"    parent: {a.to_fastexpr()[:80]}")
    print(f"    mutant: {m.to_fastexpr()[:80]}")
    print(f"    cross : {x.to_fastexpr()[:80]}")

    # Round-trip: parse a real winner back into a genome, envelope recovered
    winner = "group_neutralize(rank(ts_decay_linear(rank(sales) - rank(assets), 10)), subindustry)"
    g2 = Genome.from_fastexpr(winner)
    print(f"[genome] seed-from-winner: {'OK' if g2 else 'FAILED'} "
          f"-> {g2.to_fastexpr() if g2 else None}")
    # Verify the envelope is enforced on a -1* near-miss seed
    nm = "-1 * group_neutralize(rank(ts_corr(returns, volume, 10)), subindustry)"
    g3 = Genome.from_fastexpr(nm)
    print(f"[genome] seed-from-near-miss (sign peeled): sign={g3.sign if g3 else '?'} "
          f"-> {g3.to_fastexpr() if g3 else None}")
    # Every random genome must now be enveloped — verify outer op
    bad = 0
    for _ in range(500):
        g = Genome.random(rng, 4)
        e = g.to_fastexpr().lstrip("-1 *").strip()
        if not e.startswith("group_neutralize("):
            bad += 1
    print(f"[genome] envelope invariant: {500 - bad}/500 random genomes are group-neutralized")


if __name__ == "__main__":
    _self_test()
