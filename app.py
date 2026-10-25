"""
Root Finder & Numerical Methods Visualizer
Foundation of Mathematics - Unit 2: Numerical Methods

Every question runs 10 steps of Bisection, Regula-Falsi, Newton-Raphson and Secant,
then shows the answers side by side, how close each one is, and why it matters in computing.
"""
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

st.set_page_config(page_title="Root Finder Visualizer", page_icon="📐", layout="wide",
                   initial_sidebar_state="collapsed")

STEPS = 10          # every question runs exactly 10 steps
GOAL = 1e-4         # "4 decimal places" benchmark used to compare the methods
STOP_AT = 1e-13     # internal: stop early only if a method has hit computer precision

COLORS = {
    "Bisection": "#2B6CB0",
    "Regula-Falsi": "#C98A12",
    "Newton-Raphson": "#C0392B",
    "Secant": "#7B4FD0",
}
SYMBOLS = {
    "Bisection": "circle",
    "Regula-Falsi": "square",
    "Newton-Raphson": "triangle-up",
    "Secant": "diamond",
}
SPEED = {
    "Bisection": "Slow but always safe",
    "Regula-Falsi": "Medium, can stall",
    "Newton-Raphson": "Very fast, can fail",
    "Secant": "Fast, no derivative",
}
NEEDS = {
    "Bisection": "interval a, b",
    "Regula-Falsi": "interval a, b",
    "Newton-Raphson": "guess x₀",
    "Secant": "two guesses a, b",
}

# --------------------------------------------------------------------------
# Look and feel
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px;}
    .hero {background: linear-gradient(120deg,#16493A 0%,#1F6F58 60%,#2C8A6D 100%);
           color:#fff; border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:1.1rem;}
    .hero h1 {margin:0; font-size:2rem; letter-spacing:-0.5px; color:#fff; padding:0;}
    .hero p {margin:.35rem 0 .7rem 0; color:#D8EDE3; font-size:1.02rem;}
    .chip {display:inline-block; padding:2px 11px; margin:4px 6px 0 0; border-radius:999px;
           background:rgba(255,255,255,.16); color:#fff; font-size:.8rem; font-weight:600;}
    .stitle {font-size:1.05rem; font-weight:700; color:#16493A; margin:.1rem 0 .5rem 0;}
    .cap {font-size:.82rem; color:#5b6b63; margin-top:-.4rem;}
    .mcard {background:#fff; border:1px solid #DCE5DF; border-radius:12px; padding:.8rem .95rem; height:100%;}
    .mname {font-weight:700; font-size:.95rem;}
    .mroot {font-size:1.45rem; font-weight:700; color:#1B2420; margin:.15rem 0; font-variant-numeric:tabular-nums;}
    .msub {font-size:.85rem; color:#4b5a52;}
    .good {color:#1F7A4D; font-weight:700;}
    .slow {color:#B5651D; font-weight:700;}
    .note {background:#EEF3EF; border-radius:10px; padding:.7rem .95rem; font-size:.95rem; color:#1B2420;}
    </style>
    <div class="hero">
      <h1>Root Finder &amp; Numerical Methods Visualizer</h1>
      <p>Pick an equation and watch four methods race to its answer in 10 steps.</p>
      <span class="chip">Unit 2 · Numerical Methods</span>
      <span class="chip">Bisection</span><span class="chip">Regula-Falsi</span>
      <span class="chip">Newton-Raphson</span><span class="chip">Secant</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# preset: (expression, a, b, x0, note)
PRESETS = {
    "x³ − x − 2 = 0": ("x^3 - x - 2", 1.0, 2.0, 1.5, ""),
    "x² − 2 = 0   (finds √2)": ("x^2 - 2", 1.0, 2.0, 1.5, ""),
    "cos(x) − x = 0": ("cos(x) - x", 0.0, 1.0, 0.5, ""),
    "eˣ − 3x = 0": ("exp(x) - 3*x", 0.0, 1.0, 0.0, ""),
    "x·sin(x) − 1 = 0": ("x*sin(x) - 1", 0.0, 2.0, 1.0, ""),
    "x³ − 2x + 2 = 0   (Newton fails here)": (
        "x^3 - 2*x + 2", -2.0, -1.0, 0.0,
        "Starting Newton-Raphson at x₀ = 0 bounces 0 → 1 → 0 → 1 forever. "
        "The other methods still find the root, so failures teach as much as successes.",
    ),
    "Type my own equation": ("x^3 - 6*x^2 + 11*x - 6.1", 2.5, 3.5, 3.0, ""),
}

# --------------------------------------------------------------------------
# Reading the equation
# --------------------------------------------------------------------------
X = sp.Symbol("x")
ALLOWED = re.compile(r"^[0-9x+\-*/^().,\sa-z]*$")


def compile_expr(text):
    s = text.strip().lower()
    if not s:
        raise ValueError("Type an equation, for example x^3 - x - 2")
    if not ALLOWED.match(s):
        raise ValueError("Use only x, numbers, + - * / ^ ( ) and sin, cos, exp, log, sqrt")
    expr = parse_expr(
        s,
        local_dict={"x": X, "e": sp.E, "pi": sp.pi, "ln": sp.log},
        transformations=standard_transformations + (implicit_multiplication_application, convert_xor),
    )
    if expr.free_symbols - {X}:
        raise ValueError("Only the letter x is allowed")
    dexpr = sp.diff(expr, X)
    f_raw = sp.lambdify(X, expr, "numpy")
    df_raw = sp.lambdify(X, dexpr, "numpy")

    def wrap(fn):
        def g(v):
            try:
                with np.errstate(all="ignore"):
                    return float(fn(v))
            except Exception:
                return float("nan")
        return g

    return expr, wrap(f_raw), wrap(df_raw), dexpr


# --------------------------------------------------------------------------
# The four methods (each runs up to `mx` = 10 steps)
# step = dict(i, x, fx, bracket, frm)
# --------------------------------------------------------------------------
def bisection(f, a, b, mx):
    fa, fb = f(a), f(b)
    if not fa * fb < 0:
        return {"err": "needs f(a) and f(b) with opposite signs"}
    a0, b0, steps = a, b, []
    for i in range(1, mx + 1):
        c = (a + b) / 2
        fc = f(c)
        steps.append(dict(i=i, x=c, fx=fc, bracket=(a, b), frm=None))
        if fc == 0 or (b - a) / 2 < STOP_AT:
            break
        if fa * fc < 0:
            b = c
        else:
            a, fa = c, fc
    return dict(steps=steps, pre=[], start=dict(bracket=(a0, b0)))


def regula_falsi(f, a, b, mx):
    fa, fb = f(a), f(b)
    if not fa * fb < 0:
        return {"err": "needs f(a) and f(b) with opposite signs"}
    a0, b0, steps, prev = a, b, [], None
    for i in range(1, mx + 1):
        if fb == fa:
            break
        c = b - fb * (b - a) / (fb - fa)
        if not math.isfinite(c):
            break
        fc = f(c)
        frm = (a, fa) if abs(a - c) > abs(b - c) else (b, fb)
        steps.append(dict(i=i, x=c, fx=fc, bracket=(a, b), frm=frm))
        if fc == 0 or (prev is not None and abs(c - prev) < STOP_AT):
            break
        prev = c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return dict(steps=steps, pre=[], start=dict(bracket=(a0, b0)))


def newton(f, df, x0, mx):
    x, steps, note = x0, [], ""
    for i in range(1, mx + 1):
        fx, d = f(x), df(x)
        if not math.isfinite(d) or abs(d) < 1e-14:
            note = "the slope f′(x) became 0, so the tangent never reaches the x-axis"
            break
        xn = x - fx / d
        if not math.isfinite(xn) or abs(xn) > 1e8:
            note = "the guess flew off to infinity"
            break
        steps.append(dict(i=i, x=xn, fx=f(xn), bracket=None, frm=(x, fx)))
        if abs(xn - x) < STOP_AT:
            break
        x = xn
    return dict(steps=steps, pre=[x0], start=dict(points=[x0]), note=note)


def secant(f, a, b, mx):
    x0, x1, steps, note = a, b, [], ""
    for i in range(1, mx + 1):
        f0, f1 = f(x0), f(x1)
        if f1 == f0:
            break
        xn = x1 - f1 * (x1 - x0) / (f1 - f0)
        if not math.isfinite(xn) or abs(xn) > 1e8:
            note = "the guess flew off to infinity"
            break
        frm = (x0, f0) if abs(x0 - xn) > abs(x1 - xn) else (x1, f1)
        steps.append(dict(i=i, x=xn, fx=f(xn), bracket=None, frm=frm))
        if abs(xn - x1) < STOP_AT:
            break
        x0, x1 = x1, xn
    return dict(steps=steps, pre=[], start=dict(points=[a, b]), note=note)


def reference_root(f, df, a, b, x0):
    """Very accurate root, used only to measure how far each guess is from the truth."""
    if f(a) * f(b) < 0:
        lo, hi, flo = a, b, f(a)
        for _ in range(200):
            mid = (lo + hi) / 2
            fm = f(mid)
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        return (lo + hi) / 2
    x = x0
    for _ in range(200):
        d = df(x)
        if not math.isfinite(d) or abs(d) < 1e-14:
            return None
        xn = x - f(x) / d
        if not math.isfinite(xn):
            return None
        if abs(xn - x) < 1e-14:
            return xn if abs(f(xn)) < 1e-8 else None
        x = xn
    return None


def steps_to_goal(res, ref):
    """First step number where the answer is within 0.0001 of the true root."""
    for s in res["steps"]:
        if abs(s["x"] - ref) < GOAL:
            return s["i"]
    return None


# --------------------------------------------------------------------------
# INPUTS (main page, no sidebar)
# --------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("<div class='stitle'>Step 1 · Pick an equation</div>", unsafe_allow_html=True)
    c_pre, c_eq = st.columns([1, 1])
    preset_name = c_pre.selectbox("Choose an example", list(PRESETS.keys()),
                                  help="Pick a ready-made equation, or choose the last option and edit the box on the right.")
    p_expr, p_a, p_b, p_x0, p_note = PRESETS[preset_name]
    expr_text = c_eq.text_input("The equation (we solve f(x) = 0)", value=p_expr, key=f"expr_{preset_name}",
                                help="Write only the left side. Use x, + − * / ^ and sin, cos, exp, log, sqrt. "
                                     "Example: x^3 - x - 2")
    if p_note:
        st.info(p_note)

    st.markdown("<div class='stitle' style='margin-top:.6rem'>Step 2 · Starting numbers</div>",
                unsafe_allow_html=True)
    ca, cb, cx = st.columns(3)
    a = ca.number_input("a  (left end)", value=p_a, key=f"a_{preset_name}", step=0.1, format="%.3f",
                        help="Left end of an interval that contains the answer. Example: for x² − 2 = 0 use a = 1.")
    b = cb.number_input("b  (right end)", value=p_b, key=f"b_{preset_name}", step=0.1, format="%.3f",
                        help="Right end of the interval. Must be bigger than a. Example: for x² − 2 = 0 use b = 2.")
    x0 = cx.number_input("x₀  (one guess)", value=p_x0, key=f"x0_{preset_name}", step=0.1, format="%.3f",
                         help="A single starting guess near the answer. Example: for x² − 2 = 0 use x₀ = 1.5.")
    ca.markdown("<div class='cap'>Used by Bisection, Regula-Falsi, Secant</div>", unsafe_allow_html=True)
    cb.markdown("<div class='cap'>Used by Bisection, Regula-Falsi, Secant</div>", unsafe_allow_html=True)
    cx.markdown("<div class='cap'>Used only by Newton-Raphson</div>", unsafe_allow_html=True)

try:
    expr, f, df, dexpr = compile_expr(expr_text)
except Exception as exc:  # noqa: BLE001
    st.error(f"I could not read that equation. {exc}")
    st.stop()

if not a < b:
    st.error("Make a smaller than b, so [a, b] is a proper interval.")
    st.stop()

fa0, fb0 = f(a), f(b)
has_sign_change = math.isfinite(fa0) and math.isfinite(fb0) and fa0 * fb0 < 0
if has_sign_change:
    st.success(f"f({a:g}) = {fa0:.4g} and f({b:g}) = {fb0:.4g} have opposite signs, "
               "so the answer is trapped between a and b.")
else:
    st.warning(f"f({a:g}) = {fa0:.4g} and f({b:g}) = {fb0:.4g} have the same sign, "
               "so Bisection and Regula-Falsi are skipped. Change a or b until the signs differ.")

ref = reference_root(f, df, a, b, x0)
if ref is None:
    st.warning("I could not find an answer near these numbers. Try a different a, b or x₀.")
    st.stop()

RESULTS = {
    "Bisection": bisection(f, a, b, STEPS),
    "Regula-Falsi": regula_falsi(f, a, b, STEPS),
    "Newton-Raphson": newton(f, df, x0, STEPS),
    "Secant": secant(f, a, b, STEPS),
}
for r in RESULTS.values():
    if "err" not in r and r["steps"]:
        r["goal"] = steps_to_goal(r, ref)
        r["last"] = r["steps"][-1]

st.write("")
st.markdown(
    f"<div class='note'>Solving <b>{sp.sstr(expr).replace('**', '^')} = 0</b> &nbsp;·&nbsp; "
    f"slope (derivative) f′(x) = {sp.sstr(dexpr).replace('**', '^')} &nbsp;·&nbsp; true answer ≈ <b>{ref:.8f}</b></div>",
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------------------------------
# RESULT CARDS
# --------------------------------------------------------------------------
st.markdown("<div class='stitle'>Answers after 10 steps</div>", unsafe_allow_html=True)
cols = st.columns(4)
for col, (name, r) in zip(cols, RESULTS.items()):
    if "err" in r:
        body = (f"<div class='mroot' style='color:#8a958f'>skipped</div>"
                f"<div class='msub'>{r['err']}</div>")
    elif not r["steps"]:
        body = f"<div class='mroot' style='color:#8a958f'>no answer</div><div class='msub'>{r.get('note', '')}</div>"
    else:
        last = r["last"]
        err = abs(last["x"] - ref)
        if r["goal"]:
            sub = f"<span class='good'>4 decimals in {r['goal']} steps</span>"
        else:
            sub = f"<span class='slow'>not at 4 decimals yet</span> · error {err:.1e}"
        extra = f"<br>{r['note'].capitalize()}" if r.get("note") and not r["goal"] else ""
        body = f"<div class='mroot'>{last['x']:.7f}</div><div class='msub'>{sub}{extra}</div>"
    col.markdown(
        f"<div class='mcard' style='border-top:5px solid {COLORS[name]}'>"
        f"<div class='mname' style='color:{COLORS[name]}'>{name}</div>{body}"
        f"<div class='msub' style='margin-top:.35rem;color:#7b8a82'>Needs: {NEEDS[name]}</div></div>",
        unsafe_allow_html=True,
    )

finished = {n: r["goal"] for n, r in RESULTS.items() if "err" not in r and r.get("goal")}
if finished:
    best_steps = min(finished.values())
    winners = [n for n, v in finished.items() if v == best_steps]
    st.write("")
    st.success(f"Fastest to 4 decimal places: **{' and '.join(winners)}** in **{best_steps}** "
               f"step{'s' if best_steps != 1 else ''}.")

with st.expander("New to this? 30-second guide"):
    st.markdown(
        "- A **root** is the x that makes f(x) = 0. The graph of f(x) crosses the x-axis there.\n"
        "- **Bisection** keeps cutting an interval in half. **Regula-Falsi** draws a straight line between the two ends.\n"
        "- **Newton-Raphson** follows the tangent line from one guess. **Secant** follows the line through two guesses.\n"
        "- **4 decimal places** means the answer is within 0.0001 of the true root.\n"
        "- Every question runs **10 steps**, so you can compare the methods fairly."
    )

st.write("")
tab_race, tab_step, tab_table, tab_apps, tab_learn = st.tabs(
    ["Compare", "Step by step", "Error table", "Where it is used", "What I learned"]
)

# --------------------------------------------------------------------------
# Tab 1: compare (notebook-style table + convergence chart)
# --------------------------------------------------------------------------
with tab_race:
    st.markdown("<div class='stitle'>Each step, side by side</div>", unsafe_allow_html=True)
    table = {}
    for name, r in RESULTS.items():
        if "err" in r:
            table[name] = [np.nan] * STEPS
            continue
        xs = [s["x"] for s in r["steps"]]
        if xs:
            xs = xs + [xs[-1]] * (STEPS - len(xs))   # a finished method keeps its final answer
        else:
            xs = [np.nan] * STEPS
        table[name] = xs
    dfs = pd.DataFrame(table, index=range(1, STEPS + 1))
    dfs.index.name = "Step"

    def shade(col):
        return ["background-color:#DDEFE5; font-weight:700" if (np.isfinite(v) and abs(v - ref) < GOAL) else ""
                for v in col]

    st.dataframe(dfs.style.apply(shade, axis=0).format("{:.8f}", na_rep="–"), width="stretch", height=390)
    st.caption("Green cells are within 0.0001 of the true answer (4 decimal places).")

    st.markdown("<div class='stitle' style='margin-top:1rem'>How close is each method? (lower is better)</div>",
                unsafe_allow_html=True)
    fig = go.Figure()
    for name, r in RESULTS.items():
        if "err" in r or not r["steps"]:
            continue
        errs = [max(abs(s["x"] - ref), 1e-16) for s in r["steps"]]
        fig.add_trace(go.Scatter(x=[s["i"] for s in r["steps"]], y=errs, mode="lines+markers", name=name,
                                 line=dict(color=COLORS[name], width=3),
                                 marker=dict(symbol=SYMBOLS[name], size=9, color=COLORS[name])))
    fig.add_hline(y=GOAL, line_dash="dot", line_color="#6b7a72",
                  annotation_text="4 decimal places", annotation_position="top right")
    fig.update_yaxes(type="log", title="Distance from the true answer (log scale)", exponentformat="power")
    fig.update_xaxes(title="Step", dtick=1, range=[0.5, STEPS + 0.5])
    fig.update_layout(template="plotly_white", height=410, legend=dict(orientation="h", y=1.12),
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("A line dropping steeply means the method is fast. Newton-Raphson roughly doubles the correct "
               "digits every step.")

    speed_rows = []
    for name, r in RESULTS.items():
        speed_rows.append({
            "Method": name,
            "Needs": NEEDS[name],
            "Steps to 4 decimals": ("–" if "err" in r else (str(r["goal"]) if r.get("goal") else "more than 10")),
            "Speed": SPEED[name],
        })
    st.dataframe(pd.DataFrame(speed_rows), hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# Tab 2: step by step
# --------------------------------------------------------------------------
def step_text(name, res, k):
    if k == 0:
        if name in ("Bisection", "Regula-Falsi"):
            lo, hi = res["start"]["bracket"]
            return (f"Start with the interval [{lo:g}, {hi:g}]. f(a) and f(b) have opposite signs, "
                    "so the root is inside.")
        if name == "Newton-Raphson":
            return f"Start from the guess x₀ = {res['start']['points'][0]:g}."
        p = res["start"]["points"]
        return f"Start from two points, x₀ = {p[0]:g} and x₁ = {p[1]:g}."
    s = res["steps"][k - 1]
    if name == "Bisection":
        lo, hi = s["bracket"]
        return (f"Step {k}: the middle of [{lo:.6g}, {hi:.6g}] is x = {s['x']:.8g} and f(x) = {s['fx']:.3e}. "
                "The sign of f(x) tells us which half still holds the root, so we keep that half.")
    if name == "Regula-Falsi":
        return (f"Step {k}: the straight line between the two ends crosses the x-axis at "
                f"x = {s['x']:.8g}, where f(x) = {s['fx']:.3e}. One end moves to this point.")
    if name == "Newton-Raphson":
        return (f"Step {k}: the tangent line at x = {s['frm'][0]:.8g} crosses the x-axis at "
                f"x = {s['x']:.8g}, where f(x) = {s['fx']:.3e}.")
    return (f"Step {k}: the line through the last two points crosses the x-axis at "
            f"x = {s['x']:.8g}, where f(x) = {s['fx']:.3e}. No derivative needed.")


def step_figure(name, res, k):
    steps = res["steps"]
    xs_all = [a, b] + [s["x"] for s in steps if math.isfinite(s["x"])] + list(res["pre"])
    lo, hi = min(xs_all), max(xs_all)
    w = (hi - lo) or 1.0
    lo, hi = lo - 0.08 * w, hi + 0.08 * w
    xx = np.linspace(lo, hi, 500)
    yy = np.array([f(v) for v in xx])
    fin = yy[np.isfinite(yy)]
    ylo, yhi = min(fin.min(), 0), max(fin.max(), 0)
    pad = 0.1 * ((yhi - ylo) or 1.0)
    ylo, yhi = ylo - pad, yhi + pad
    col = COLORS[name]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="#1B2420", width=2.5)))
    fig.add_hline(y=0, line_color="#999", line_width=1)
    br = res["start"].get("bracket") if k == 0 else steps[k - 1].get("bracket")
    if br:
        fig.add_vrect(x0=br[0], x1=br[1], fillcolor=col, opacity=0.12, line_width=0)
    if k == 0:
        pts = res["start"].get("points", [])
        if pts:
            fig.add_trace(go.Scatter(x=pts, y=[f(p) for p in pts], mode="markers", name="start",
                                     marker=dict(size=12, color=col, symbol=SYMBOLS[name])))
    if k > 1:
        prev = steps[: k - 1]
        fig.add_trace(go.Scatter(x=[s["x"] for s in prev], y=[s["fx"] for s in prev], mode="markers",
                                 name="earlier steps",
                                 marker=dict(size=8, color=col, opacity=0.35, symbol=SYMBOLS[name])))
    if k >= 1:
        s = steps[k - 1]
        if s["frm"]:
            fig.add_trace(go.Scatter(x=[s["frm"][0], s["x"]], y=[s["frm"][1], 0], mode="lines",
                                     name="construction line", line=dict(color=col, width=2)))
        fig.add_trace(go.Scatter(x=[s["x"], s["x"]], y=[0, s["fx"]], mode="lines", showlegend=False,
                                 line=dict(color=col, width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=[s["x"]], y=[s["fx"]], mode="markers", name=f"x{k}",
                                 marker=dict(size=14, color=col, symbol=SYMBOLS[name],
                                             line=dict(color="white", width=1.5))))
    fig.add_trace(go.Scatter(x=[ref], y=[0], mode="markers", name="true root",
                             marker=dict(size=13, color="rgba(0,0,0,0)", symbol="diamond-open",
                                         line=dict(color="#1F6F58", width=2.5))))
    fig.update_xaxes(range=[lo, hi], title="x", zeroline=False)
    fig.update_yaxes(range=[ylo, yhi], title="f(x)", zeroline=False)
    fig.update_layout(template="plotly_white", height=450, margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=1.08))
    return fig


with tab_step:
    usable = [n for n, r in RESULTS.items() if "err" not in r and r["steps"]]
    if not usable:
        st.info("No method produced steps for these numbers.")
    else:
        name = st.radio("Method", usable, horizontal=True, key="method_pick")
        res = RESULTS[name]
        n = len(res["steps"])
        k = st.slider("Step", 0, n, min(3, n), key=f"step_{name}_{preset_name}_{n}")
        st.plotly_chart(step_figure(name, res, k), width="stretch")
        st.markdown(f"<div class='note'>{step_text(name, res, k)}</div>", unsafe_allow_html=True)
        st.caption("The green diamond is the true root. Move the slider to replay the method one step at a time.")

# --------------------------------------------------------------------------
# Tab 3: error table
# --------------------------------------------------------------------------
with tab_table:
    st.markdown(
        "How far is each guess from the true answer x*?\n\n"
        "- **Absolute error** = |xₙ − x*|\n"
        "- **Relative error** = |xₙ − x*| / |x*|\n"
        "- **Percentage error** = relative error × 100"
    )
    usable = [n for n, r in RESULTS.items() if "err" not in r and r["steps"]]
    if usable:
        pick = st.selectbox("Show the table for", usable, key="table_pick")
        data = []
        for s in RESULTS[pick]["steps"]:
            ae = abs(s["x"] - ref)
            rel = ae / abs(ref) if abs(ref) > 1e-12 else float("nan")
            data.append({"Step": s["i"], "xₙ": s["x"], "f(xₙ)": s["fx"], "Absolute error": ae,
                         "Relative error": rel, "Percentage error (%)": rel * 100})
        dfm = pd.DataFrame(data)
        st.dataframe(
            dfm.style.format({"xₙ": "{:.10f}", "f(xₙ)": "{:.3e}", "Absolute error": "{:.3e}",
                              "Relative error": "{:.3e}", "Percentage error (%)": "{:.3e}"}),
            hide_index=True, width="stretch",
        )
        st.download_button("Download this table (CSV)", dfm.to_csv(index=False),
                           file_name=f"{pick}_steps.csv", mime="text/csv")

# --------------------------------------------------------------------------
# Tab 4: where it is used in computing
# --------------------------------------------------------------------------
with tab_apps:
    st.markdown("<div class='stitle'>Root finding hides inside everyday software</div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1.container(border=True):
        st.markdown("**Debugging with `git bisect`**  \nFinding the first bad commit is Bisection on a list. "
                    "Every test halves the search, so 1,000 commits need only about 10 tests.")
    with g2.container(border=True):
        st.markdown("**Square roots in hardware**  \nCalculators and CPUs solve x² − n = 0 with Newton-Raphson. "
                    "The famous fast inverse square root in Quake III is one Newton step.")
    g3, g4 = st.columns(2)
    with g3.container(border=True):
        st.markdown("**Machine learning**  \nTraining finds where a gradient is zero. Newton and secant-style "
                    "methods (such as BFGS) power many optimisers.")
    with g4.container(border=True):
        st.markdown("**Graphics and games**  \nRay tracing solves f(t) = 0 to find where a ray hits a surface. "
                    "Physics engines find the time of impact the same way.")

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### Demo 1 · Hunt a bug like git bisect")
        commits = st.slider("Commits in history", 10, 5000, 1000, key="commits")
        bad = st.number_input("First bad commit (hidden)", 1, commits, min(613, commits), key="bad")
        lo, hi, tests = 1, commits, []
        while lo < hi:
            mid = (lo + hi) // 2
            is_bad = mid >= bad
            tests.append({"Test": len(tests) + 1, "Searching": f"{lo} to {hi}", "Checked commit": mid,
                          "Result": "bad" if is_bad else "good"})
            if is_bad:
                hi = mid
            else:
                lo = mid + 1
        st.dataframe(pd.DataFrame(tests), hide_index=True, width="stretch", height=260)
        st.success(f"Found commit **{lo}** in **{len(tests)}** tests. "
                   f"Checking one by one could take {commits}.")
    with d2:
        st.markdown("#### Demo 2 · √n with Newton-Raphson")
        nval = st.number_input("Find the square root of", min_value=0.01, value=2.0, key="sqrt_n")
        xk, rows2, true = max(nval, 1.0), [], math.sqrt(nval)
        for i in range(1, 8):
            xk = 0.5 * (xk + nval / xk)
            err = abs(xk - true)
            digits = min(15, int(-math.log10(err))) if err > 0 else 15
            rows2.append({"Step": i, "Estimate": f"{xk:.12f}", "Error": f"{err:.1e}", "Correct digits": digits})
        st.dataframe(pd.DataFrame(rows2), hide_index=True, width="stretch", height=290)
        st.caption("Rule: xₙ₊₁ = ½(xₙ + n/xₙ). It is Newton-Raphson on x² − n. The correct digits roughly double each step.")

# --------------------------------------------------------------------------
# Tab 5: what I learned
# --------------------------------------------------------------------------
with tab_learn:
    st.markdown("#### Objective")
    st.write("Find the roots of f(x) = 0 with four numerical methods, then compare how fast and how accurately "
             "each one reaches the answer, and connect the idea to real computing problems.")
    st.markdown("#### Mathematics used (Unit 2: Numerical Methods)")
    st.markdown(
        "- Types of errors: absolute, relative and percentage\n"
        "- Solving algebraic and transcendental equations: Bisection, Regula-Falsi, Newton-Raphson, Secant\n"
        "- Derivatives (worked out symbolically with SymPy) for Newton-Raphson"
    )
    st.markdown("#### What I learned")
    st.markdown(
        "1. **Bisection is slow but safe.** It always works when f(a) and f(b) have opposite signs.\n"
        "2. **Newton-Raphson is the fastest but can fail.** The preset x³ − 2x + 2 shows it bouncing forever.\n"
        "3. **Secant is a good middle path.** It is nearly as fast as Newton and needs no derivative.\n"
        "4. **Regula-Falsi can stall.** One end of the interval often stays still.\n"
        "5. **A log-scale error chart makes speed easy to see.** A steeper drop means a faster method."
    )
    st.markdown("#### How it connects to computing")
    st.write("Bisection is binary search (git bisect). Newton-Raphson computes square roots in hardware and "
             "drives optimisation in machine learning. Root finding also locates ray and surface hits in "
             "graphics and impact times in physics engines.")