# app.py — QuickBite Diner Intelligence (v2)
# ------------------------------------------------------------
# Run locally:   streamlit run app.py
# Needs (same folder): food_models.joblib, diners.csv, orders.csv
# ------------------------------------------------------------
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="QuickBite Diner Intelligence", page_icon="🍔", layout="wide")
HERE = Path(__file__).parent

# ---------- design tokens ----------
INK, MUTED, PAPER, LINE = "#18212F", "#5B6576", "#F4F6F9", "#DDE2EA"
RISK, WATCH, SAFE = "#D7263D", "#E08A00", "#0F9D7A"
SEG_META = {
    "Families":        dict(color="#E39B0B", emoji="👨‍👩‍👧", blurb="Big, infrequent family orders. Highest basket value, lowest loyalty."),
    "Budget Students": dict(color="#3159D6", emoji="🎒", blurb="Young, frequent, small orders. Discount-driven and very loyal."),
    "Premium Foodies": dict(color="#6A8F3A", emoji="🍣", blurb="Quality over price. Rarely use discounts, highest lifetime spend."),
}
FALLBACK = ["#8E44AD", "#16A085", "#C0392B", "#2C3E50"]
ITEM_EMOJI = {"pizza": "🍕", "burger": "🍔", "fries": "🍟", "cola": "🥤", "wrap": "🌯", "salad": "🥗",
              "sushi": "🍣", "coffee": "☕", "dessert": "🍰", "pasta": "🍝", "wine": "🍷", "biryani": "🍛",
              "raita": "🥣", "steak": "🥩", "green_tea": "🍵", "miso_soup": "🍜", "garlic_bread": "🥖"}
LABELS = {"orders_per_month": "Orders per month", "avg_order_value": "Avg order value",
          "avg_items_per_order": "Items per order", "days_since_last_order": "Days since last order",
          "discount_rate": "Discount usage", "age": "Age", "total_spend": "Total spend"}


def item_name(i):
    return f"{ITEM_EMOJI.get(i, '🍽️')} {i.replace('_', ' ').capitalize()}"


def seg_color(name, names):
    return SEG_META.get(name, {}).get("color") or FALLBACK[list(names).index(name) % len(FALLBACK)]


def prob_color(p, thr):
    return SAFE if p >= max(thr, 0.7) else (WATCH if p >= thr else RISK)


# ---------- styles ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');
.stApp {{ background:{PAPER}; }}
.block-container {{ padding-top:3.6rem; max-width:1250px; }}
h1, h2, h3, h4 {{ font-family:'Bricolage Grotesque', 'Figtree', system-ui, sans-serif !important; color:{INK}; letter-spacing:-0.01em; }}
.qb-body, .qb-body * {{ font-family:'Figtree', system-ui, sans-serif; }}
[data-testid="stSidebar"] {{ background:#FFFFFF; border-right:1px solid {LINE}; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; padding:10px 16px; }}
[data-testid="stMetric"] {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:14px 16px; }}
[data-testid="stMetricValue"] {{ font-family:'Bricolage Grotesque', sans-serif; font-weight:700; }}

.qb-head h1 {{ font-size:2.4rem; font-weight:800; margin:0; line-height:1.05; padding:0; }}
.qb-head p {{ color:{MUTED}; margin:6px 0 4px; max-width:62ch; font-size:1.02rem; }}

/* the order ticket: the one bold element on the page */
.qb-ticket {{ background:#fff; border-radius:14px; position:relative; overflow:hidden;
             box-shadow:0 1px 0 {LINE}, 0 12px 30px -18px rgba(24,33,47,.5); }}
.qb-ticket .strip {{ height:8px; }}
.qb-ticket .top {{ padding:18px 22px 16px; display:grid; grid-template-columns:1fr auto; gap:18px; align-items:start; }}
.qb-ticket .seg {{ font-family:'Bricolage Grotesque',sans-serif; font-size:1.8rem; font-weight:800; color:{INK}; line-height:1.1; }}
.qb-ticket .blurb {{ color:{MUTED}; font-size:.95rem; margin-top:6px; max-width:40ch; }}
.qb-ticket .prob {{ text-align:right; }}
.qb-ticket .prob .num {{ font-family:'Bricolage Grotesque',sans-serif; font-size:3.6rem; font-weight:800; line-height:.95; }}
.qb-ticket .prob .lab {{ color:{MUTED}; font-size:.9rem; margin-top:4px; }}
.qb-ticket .perf {{ border-top:2px dashed {LINE}; margin:0 16px; position:relative; }}
.qb-ticket .perf:before, .qb-ticket .perf:after {{ content:''; position:absolute; top:-12px; width:22px; height:22px;
             border-radius:50%; background:{PAPER}; }}
.qb-ticket .perf:before {{ left:-28px; }} .qb-ticket .perf:after {{ right:-28px; }}
.qb-ticket .bottom {{ padding:14px 22px 18px; display:grid; grid-template-columns:1fr 1fr; gap:14px 18px; }}
.qb-k {{ color:{MUTED}; font-size:.85rem; margin-bottom:3px; }}
.qb-v {{ color:{INK}; font-weight:600; }}
.qb-pill {{ display:inline-block; padding:3px 11px; border-radius:999px; font-weight:700; font-size:.85rem; color:#fff; }}
.qb-chip {{ display:inline-block; background:{PAPER}; border:1px solid {LINE}; border-radius:999px; padding:4px 11px;
           margin:0 6px 6px 0; font-size:.92rem; color:{INK}; }}
.qb-action {{ border-left:4px solid; background:#fff; border-radius:0 10px 10px 0; padding:12px 16px; margin-top:14px; color:{INK}; line-height:1.5; }}
.qb-action b {{ font-family:'Bricolage Grotesque',sans-serif; }}
.qb-card {{ background:#fff; border:1px solid {LINE}; border-radius:12px; padding:16px 18px; height:100%; }}
.qb-card h4 {{ margin:0 0 4px; font-size:1.15rem; padding:0; }}
.qb-card .sub {{ color:{MUTED}; font-size:.9rem; margin-bottom:10px; line-height:1.4; }}
.qb-row {{ display:flex; justify-content:space-between; border-top:1px solid {LINE}; padding:6px 0; font-size:.93rem; }}
.qb-row span:first-child {{ color:{MUTED}; }} .qb-row span:last-child {{ font-weight:600; color:{INK}; }}
.qb-insight {{ background:#fff; border:1px solid {LINE}; border-left:4px solid {INK}; border-radius:0 12px 12px 0;
              padding:14px 18px; color:{INK}; margin:6px 0 14px; line-height:1.55; }}
.qb-insight b {{ font-family:'Bricolage Grotesque',sans-serif; }}
@media (max-width: 700px) {{ .qb-ticket .bottom, .qb-ticket .top {{ grid-template-columns:1fr; }} .qb-ticket .prob {{ text-align:left; }} }}
</style>
""", unsafe_allow_html=True)


def chart_layout(fig, height=320, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Figtree, sans-serif", color=INK, size=13),
                      showlegend=legend, legend=dict(orientation="h", y=-0.2, x=0))
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    return fig


def show(fig, key):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


# ---------- models & data ----------
@st.cache_resource
def load_models():
    return joblib.load(HERE / "food_models.joblib")


M = load_models()
scaler, kmeans = M["scaler"], M["kmeans"]
seg_names, clf, FEATURES, reco = M["segment_names"], M["clf"], M["features"], M["reco"]
CLF_FEATURES = FEATURES + ["age", "total_spend"]
SEGMENTS = [seg_names[k] for k in sorted(seg_names)]


def segment_of(df):
    labels = kmeans.predict(scaler.transform(df[FEATURES]))
    return np.array([seg_names.get(int(c), f"Segment {c}") for c in labels])


def reorder_prob(df):
    return clf.predict_proba(df[CLF_FEATURES])[:, 1]


@st.cache_data
def load_data():
    if not (HERE / "diners.csv").exists() or not (HERE / "orders.csv").exists():
        return None, None
    d = pd.read_csv(HERE / "diners.csv")
    o = pd.read_csv(HERE / "orders.csv")
    d["segment"] = segment_of(d)
    d["p_reorder"] = reorder_prob(d)
    d["monthly_value"] = d.orders_per_month * d.avg_order_value
    d["value_at_risk"] = (1 - d.p_reorder) * d.monthly_value
    o = o.merge(d[["diner_id", "segment"]], on="diner_id", how="left")
    return d, o


@st.cache_data
def basket_stats(o, segment):
    """Pairwise rules + best 2–3 item bundles, computed directly (no mlxtend needed)."""
    sub = o if segment == "All diners" else o[o.segment == segment]
    baskets = sub.groupby("order_id").item.apply(lambda s: tuple(sorted(set(s))))
    n = len(baskets)
    counts = {}
    for b in baskets:
        for k in (1, 2, 3):
            for combo in itertools.combinations(b, k):
                counts[combo] = counts.get(combo, 0) + 1
    sup = {c: v / n for c, v in counts.items()}
    rules = []
    for c, s in sup.items():
        if len(c) == 2:
            for a, b in (c, c[::-1]):
                conf = s / sup[(a,)]
                rules.append(dict(antecedent=a, consequent=b, support=s, confidence=conf, lift=conf / sup[(b,)]))
    rules = pd.DataFrame(rules)
    min_sup = 0.02 if segment == "All diners" else 0.08
    cand = [dict(items=c, support=s, lift=s / np.prod([sup[(i,)] for i in c]),
                 together=s / max(sup[(i,)] for i in c))   # all-confidence: how often the full set travels together
            for c, s in sup.items() if len(c) >= 2 and s >= min_sup]
    bundles = pd.DataFrame(cand)
    if len(bundles):
        rank = "together" if segment == "All diners" else "lift"
        bundles = bundles[bundles.lift > 1].sort_values([rank, "support"], ascending=False)
        keep, seen = [], []
        for _, r in bundles.iterrows():          # skip bundles that overlap one already shown
            if not any(set(r["items"]) & s for s in seen):
                keep.append(r)
                seen.append(set(r["items"]))
        bundles = pd.DataFrame(keep)
        bundles = bundles[(bundles.lift >= 1.5) | (bundles.index == bundles.index[0])]   # drop weak bundles, keep at least one
    items = sorted({c[0] for c in sup if len(c) == 1})
    return n, rules, bundles, items


def recency_curve(profile, days=range(0, 61)):
    rows = pd.DataFrame([{**profile, "days_since_last_order": d} for d in days])
    return np.array(list(days)), reorder_prob(rows)


def deadline(profile, thr):
    x, y = recency_curve(profile)
    below = x[y < thr]
    return int(below[0]) if len(below) else None


def drivers(profile):
    """Per-feature push toward reorder (+) or churn (−) for a scaled linear model."""
    try:
        est = clf[-1]
        z = clf[:-1].transform(pd.DataFrame([profile])[CLF_FEATURES])[0]
        return pd.Series(est.coef_[0] * z, index=CLF_FEATURES)
    except Exception:
        return None


diners, orders = load_data()
seg_bundle = {}
if orders is not None:
    for s in SEGMENTS:
        b = basket_stats(orders, s)[2]
        if len(b):
            seg_bundle[s] = b.iloc[0]["items"]


def bundle_text(seg):
    b = seg_bundle.get(seg)
    return " + ".join(i.replace("_", " ") for i in b) if b is not None else None


def next_best_action(seg, p, thr, days_left):
    bundle = bundle_text(seg) or "favourite-items"
    top = reco.get(seg, ["a favourite"])[0].replace("_", " ") if reco.get(seg) else "a favourite"
    if p < thr * 0.67:
        return RISK, "Win back now", f"Send a {bundle} bundle with a time-limited discount within 48 hours. Most diners like this one don't come back on their own."
    if p < thr:
        return WATCH, "Nudge this week", f"Send a reminder featuring {top}. A small incentive now is cheaper than a win-back later."
    if seg == "Budget Students":
        return SAFE, "Keep, don't discount", f"Already loyal. Skip discounts and cross-sell the {bundle} combo instead."
    msg = f"Upsell the {bundle} bundle at full price."
    if days_left is not None and days_left > 0:
        msg += f" Re-engage within {days_left} days, before their reorder chance drops below {thr:.0%}."
    return SAFE, "Grow this diner", msg


# ---------- sidebar ----------
PRESETS = {
    "Student":        dict(age=21, orders_per_month=14, avg_order_value=40,  avg_items_per_order=2, days_since_last_order=3,  discount_rate=0.70, total_spend=550),
    "Quiet family":   dict(age=42, orders_per_month=3,  avg_order_value=180, avg_items_per_order=6, days_since_last_order=20, discount_rate=0.30, total_spend=600),
    "Foodie":         dict(age=33, orders_per_month=9,  avg_order_value=100, avg_items_per_order=3, days_since_last_order=6,  discount_rate=0.05, total_spend=900),
    "Drifter":        dict(age=34, orders_per_month=6,  avg_order_value=110, avg_items_per_order=3, days_since_last_order=16, discount_rate=0.10, total_spend=750),
}
DEFAULT = dict(age=28, orders_per_month=8, avg_order_value=90, avg_items_per_order=3, days_since_last_order=10, discount_rate=0.30, total_spend=800)
for k, v in DEFAULT.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown("### Diner profile")
    st.caption("Load an example diner, or set the sliders yourself.")
    bcols = st.columns(2)
    for i, name in enumerate(PRESETS):
        bcols[i % 2].button(name, on_click=apply_preset, args=(name,), width="stretch")
    st.slider("Age", 18, 70, key="age")
    st.slider("Orders per month", 1, 20, key="orders_per_month")
    st.slider("Average order value (AED)", 15, 250, key="avg_order_value")
    st.slider("Average items per order", 1, 10, key="avg_items_per_order")
    st.slider("Days since last order", 0, 60, key="days_since_last_order")
    st.slider("Discount usage rate", 0.0, 1.0, step=0.05, key="discount_rate")
    st.number_input("Total spend so far (AED)", 50, 20000, step=50, key="total_spend")
    st.divider()
    thr = st.slider("At-risk threshold", 0.3, 0.8, 0.6, 0.05,
                    help="Diners below this reorder chance are flagged. 0.6 caught 44 of 48 churners in testing vs 40 at 0.5, with the same accuracy.")

profile = {k: st.session_state[k] for k in DEFAULT}
row = pd.DataFrame([profile])
seg = segment_of(row)[0]
p = float(reorder_prob(row)[0])
color = seg_color(seg, SEGMENTS)
pcol = prob_color(p, thr)
dl = deadline(profile, thr)
days_left = None if dl is None else dl - profile["days_since_last_order"]

# ---------- header ----------
st.markdown("""
<div class="qb-head qb-body">
  <h1>QuickBite diner intelligence</h1>
  <p>Who your diners are, what they order together, and who is about to stop ordering.</p>
</div>""", unsafe_allow_html=True)

tab_lookup, tab_seg, tab_basket, tab_watch = st.tabs(["🔎 Diner lookup", "👥 Segments", "🛒 Baskets", "⚠️ Churn watchlist"])

# ================= TAB 1: DINER LOOKUP =================
with tab_lookup:
    status = "Likely to reorder" if p >= thr else "At risk of churning"
    if dl is None:
        clock = "Stays above the at-risk line even after 60 quiet days"
    elif days_left > 0:
        clock = f"{days_left} days left before the at-risk line (day {dl})"
    elif dl == 0:
        clock = "Below the at-risk line even after a fresh order"
    else:
        clock = f"Crossed the at-risk line {-days_left} days ago (day {dl})"
    monthly = profile["orders_per_month"] * profile["avg_order_value"]
    chips = "".join(f'<span class="qb-chip">{item_name(i)}</span>' for i in reco.get(seg, []))
    meta = SEG_META.get(seg, {})
    ac, atitle, amsg = next_best_action(seg, p, thr, days_left)

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(f"""
<div class="qb-ticket qb-body">
  <div class="strip" style="background:{color}"></div>
  <div class="top">
    <div>
      <div class="qb-k">Segment</div>
      <div class="seg">{meta.get('emoji', '🍽️')} {seg}</div>
      <div class="blurb">{meta.get('blurb', '')}</div>
    </div>
    <div class="prob">
      <div class="num" style="color:{pcol}">{p:.0%}</div>
      <div class="lab">chance of reordering</div>
      <div style="margin-top:8px"><span class="qb-pill" style="background:{pcol}">{status}</span></div>
    </div>
  </div>
  <div class="perf"></div>
  <div class="bottom">
    <div><div class="qb-k">Reorder clock</div><div class="qb-v">{clock}</div></div>
    <div><div class="qb-k">Monthly order value</div><div class="qb-v">AED {monthly:,.0f}, of which AED {(1-p)*monthly:,.0f} at risk</div></div>
    <div style="grid-column:1/-1"><div class="qb-k">Recommended items</div>{chips or '—'}</div>
  </div>
</div>
<div class="qb-action qb-body" style="border-color:{ac}"><b>{atitle}.</b> {amsg}</div>
""", unsafe_allow_html=True)

    with right:
        st.markdown("#### What if they stay quiet?")
        x, y = recency_curve(profile)
        fig = go.Figure()
        fig.add_hrect(y0=0, y1=thr, fillcolor=RISK, opacity=0.06, line_width=0)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=INK, width=3),
                                 hovertemplate="Day %{x}: %{y:.0%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[profile["days_since_last_order"]], y=[p], mode="markers",
                                 marker=dict(size=15, color=pcol, line=dict(color="#fff", width=2)),
                                 hovertemplate="Today: %{y:.0%}<extra></extra>"))
        fig.add_hline(y=thr, line_dash="dot", line_color=RISK,
                      annotation_text=f"At-risk line ({thr:.0%})", annotation_position="top right",
                      annotation_font_color=RISK)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.04])
        fig.update_xaxes(title="Days since last order")
        show(chart_layout(fig, 290, legend=False), "whatif")
        st.caption("Same diner, every other input held fixed. Days since last order is the strongest churn signal in the model.")

    st.markdown("#### Why the model thinks so")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        dr = drivers(profile)
        if dr is not None:
            dr = dr.sort_values()
            fig = go.Figure(go.Bar(x=dr.values, y=[LABELS[i] for i in dr.index], orientation="h",
                                   marker_color=[SAFE if v > 0 else RISK for v in dr.values],
                                   hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
            fig.update_xaxes(title="← pushes toward churn  |  pushes toward reorder →", zeroline=True,
                             zerolinewidth=2, zerolinecolor=INK)
            show(chart_layout(fig, 300, legend=False), "drivers")
            st.caption("How much each input moved this prediction, compared with an average diner.")
        else:
            st.info("The driver breakdown needs a scaled linear classifier (a scikit-learn pipeline ending in LogisticRegression).")
    with c2:
        if diners is not None:
            seg_df = diners[diners.segment == seg]
            feats = FEATURES + ["total_spend"]
            pct_d = [(diners[f] < profile[f]).mean() * 100 for f in feats]
            pct_s = [(diners[f] < seg_df[f].median()).mean() * 100 for f in feats]
            ys = list(range(len(feats)))
            fig = go.Figure()
            for i in ys:
                fig.add_shape(type="line", x0=0, x1=100, y0=i, y1=i, line=dict(color=LINE, width=6), layer="below")
            fig.add_trace(go.Scatter(x=pct_s, y=ys, mode="markers", name=f"Typical {seg.lower()}",
                                     marker=dict(symbol="line-ns", size=20, line=dict(width=4, color=color)),
                                     hovertemplate="Segment median: higher than %{x:.0f}% of diners<extra></extra>"))
            fig.add_trace(go.Scatter(x=pct_d, y=ys, mode="markers", name="This diner",
                                     marker=dict(size=14, color=INK, line=dict(color="#fff", width=2)),
                                     hovertemplate="This diner: higher than %{x:.0f}% of diners<extra></extra>"))
            fig.update_yaxes(tickvals=ys, ticktext=[LABELS[f] for f in feats], showgrid=False)
            fig.update_xaxes(range=[-3, 103], title="Percentile among all diners", ticksuffix="%")
            show(chart_layout(fig, 300), "percentiles")
            st.caption(f"Where this diner sits among all {len(diners)} diners, next to a typical member of {seg}.")

# ================= TAB 2: SEGMENTS =================
with tab_seg:
    if diners is None:
        st.warning("Add diners.csv and orders.csv next to app.py to see segment insights.")
    else:
        flagged = diners.p_reorder < thr
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Diners", f"{len(diners):,}")
        k2.metric("Monthly order value", f"AED {diners.monthly_value.sum()/1000:,.0f}K")
        k3.metric("Value at risk", f"AED {diners.value_at_risk.sum()/1000:,.0f}K",
                  f"{diners.value_at_risk.sum()/diners.monthly_value.sum():.0%} of total", delta_color="off", delta_arrow="off")
        k4.metric("Diners flagged at risk", f"{flagged.sum()}", f"{flagged.mean():.0%} of base", delta_color="off", delta_arrow="off")

        g = diners.groupby("segment")
        summary = pd.DataFrame({
            "n": g.size(), "aov": g.avg_order_value.mean(), "opm": g.orders_per_month.mean(),
            "reorder": g.reordered.mean() if "reordered" in diners else g.p_reorder.mean(),
            "value": g.monthly_value.sum(), "risk": g.value_at_risk.sum(), "age": g.age.mean(),
            "disc": g.discount_rate.mean()}).reindex(SEGMENTS)
        worst = summary.risk.idxmax()
        seg_dl = {s: deadline(diners[diners.segment == s][CLF_FEATURES].median().to_dict(), thr) for s in SEGMENTS}
        safest = summary.reorder.idxmax()
        wd = seg_dl[worst]

        st.markdown(f"""
<div class="qb-insight qb-body"><b>Where the money leaks.</b> {worst} account for AED {summary.loc[worst,'risk']/1000:,.0f}K
of the AED {summary.risk.sum()/1000:,.0f}K monthly value at risk ({summary.loc[worst,'risk']/summary.risk.sum():.0%}).
A typical diner in this segment falls below the at-risk line after only <b>{wd if wd is not None else '60+'} days</b> without an order,
so their win-back offers need to go out sooner than anyone else's. {safest} reorder {summary.loc[safest,'reorder']:.0%} of the time,
so discounts spent on them are mostly wasted.</div>
""", unsafe_allow_html=True)

        cols = st.columns(len(SEGMENTS), gap="medium")
        for c, s in zip(cols, SEGMENTS):
            r = summary.loc[s]
            items = "".join(f'<span class="qb-chip">{item_name(i)}</span>' for i in reco.get(s, [])[:4])
            dd = seg_dl[s]
            c.markdown(f"""
<div class="qb-card qb-body" style="border-top:6px solid {seg_color(s, SEGMENTS)}">
  <h4>{SEG_META.get(s, {}).get('emoji', '')} {s}</h4>
  <div class="sub">{SEG_META.get(s, {}).get('blurb', '')}</div>
  <div class="qb-row"><span>Diners</span><span>{int(r.n)} ({r.n/len(diners):.0%})</span></div>
  <div class="qb-row"><span>Average age</span><span>{r.age:.0f}</span></div>
  <div class="qb-row"><span>Orders per month</span><span>{r.opm:.1f}</span></div>
  <div class="qb-row"><span>Average order value</span><span>AED {r.aov:,.0f}</span></div>
  <div class="qb-row"><span>Discount usage</span><span>{r.disc:.0%}</span></div>
  <div class="qb-row"><span>Reorder rate</span><span>{r.reorder:.0%}</span></div>
  <div class="qb-row"><span>Win-back deadline</span><span>{'day ' + str(dd) if dd is not None else '60+ days'}</span></div>
  <div style="margin-top:10px">{items}</div>
</div>""", unsafe_allow_html=True)

        st.write("")
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("#### Order value at risk, by segment")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=SEGMENTS, x=summary.value - summary.risk, orientation="h", name="Expected to stay",
                                 marker_color=SAFE, hovertemplate="AED %{x:,.0f}<extra>Expected to stay</extra>"))
            fig.add_trace(go.Bar(y=SEGMENTS, x=summary.risk, orientation="h", name="At risk",
                                 marker_color=RISK, hovertemplate="AED %{x:,.0f}<extra>At risk</extra>"))
            fig.update_layout(barmode="stack")
            fig.update_xaxes(tickprefix="AED ", tickformat=",.0f")
            show(chart_layout(fig, 300), "leak")
        with b:
            st.markdown("#### How fast each segment goes cold")
            fig = go.Figure()
            for s in SEGMENTS:
                x, y = recency_curve(diners[diners.segment == s][CLF_FEATURES].median().to_dict())
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=s, line=dict(width=3, color=seg_color(s, SEGMENTS)),
                                         hovertemplate="Day %{x}: %{y:.0%}<extra>" + s + "</extra>"))
            fig.add_hline(y=thr, line_dash="dot", line_color=RISK)
            fig.update_yaxes(tickformat=".0%", range=[0, 1.04])
            fig.update_xaxes(title="Days since last order")
            show(chart_layout(fig, 300), "cold")
            st.caption("Reorder chance for each segment's typical diner as their silence grows.")

        st.markdown("#### The diner map")
        rng = np.random.default_rng(0)
        fig = go.Figure()
        for s in SEGMENTS:
            sd = diners[diners.segment == s]
            fig.add_trace(go.Scatter(x=sd.orders_per_month + rng.uniform(-.25, .25, len(sd)),
                                     y=sd.avg_order_value, mode="markers", name=s,
                                     marker=dict(color=seg_color(s, SEGMENTS), size=8, opacity=.65),
                                     customdata=np.c_[sd.diner_id, sd.p_reorder.round(2)],
                                     hovertemplate="%{customdata[0]}<br>%{y:.0f} AED per order<br>Reorder chance %{customdata[1]}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[profile["orders_per_month"]], y=[profile["avg_order_value"]], mode="markers",
                                 name="Diner in sidebar", marker=dict(symbol="star", size=24, color=INK, line=dict(color="#fff", width=2))))
        fig.update_xaxes(title="Orders per month")
        fig.update_yaxes(title="Average order value (AED)")
        show(chart_layout(fig, 380), "map")
        st.caption("Each dot is a diner. The segments separate cleanly on order frequency and basket value alone.")

# ================= TAB 3: BASKETS =================
with tab_basket:
    if orders is None:
        st.warning("Add orders.csv next to app.py to see basket insights.")
    else:
        f1, f2 = st.columns([2, 1])
        scope = f1.radio("Show baskets for", ["All diners"] + SEGMENTS, horizontal=True)
        min_conf = f2.slider("Minimum confidence", 0.3, 0.95, 0.5, 0.05)
        n, rules, bundles, items = basket_stats(orders, scope)

        st.markdown(f"#### Bundles worth promoting ({n:,} orders)")
        if len(bundles):
            bc = st.columns(3, gap="medium")
            for c, (_, b) in zip(bc, bundles.head(3).iterrows()):
                c.markdown(f"""
<div class="qb-card qb-body">
  <h4>{' + '.join(item_name(i) for i in b['items'])}</h4>
  <div class="sub">When any one of these is ordered, the whole set comes with it {b['together']:.0%} of the time.</div>
  <div class="qb-row"><span>Share of all orders</span><span>{b['support']:.0%}</span></div>
  {'' if scope == 'All diners' else f'<div class="qb-row"><span>Lift</span><span>{b["lift"]:.1f}×</span></div>'}
</div>""", unsafe_allow_html=True)
        else:
            st.info("No bundle clears the support threshold for this group.")

        notes = {
            "Budget Students": "<b>Students already buy a fixed combo.</b> Cola is in nearly every student order, so cola rules have a lift close to 1 and say nothing new. Wrap and fries is the only real cross-sell, which makes a wrap meal deal the lever here.",
            "All diners": "<b>Read overall lifts with care.</b> Items like sushi or biryani are bought by only one segment, which inflates lift across the whole base. Pick a segment to see what really drives each group.",
            "Families": "<b>Families order meals, not items.</b> Pizza nights (pizza, garlic bread, dessert, cola) and biryani nights (biryani, raita, salad) are two distinct occasions. Promote them as two named family bundles.",
            "Premium Foodies": "<b>Foodies order cuisines.</b> A Japanese set (sushi, miso soup, green tea) and a steak dinner (steak, salad, wine) dominate. Sell them as chef's sets at full price; this segment barely uses discounts.",
        }
        if scope in notes:
            st.markdown(f'<div class="qb-insight qb-body" style="margin-top:14px">{notes[scope]}</div>', unsafe_allow_html=True)

        a, b = st.columns([1.2, 1], gap="large")
        with a:
            st.markdown("#### If they order X, they also order Y")
            r = rules[(rules.confidence >= min_conf) & (rules.lift > 1.05) & (rules.support >= 0.02)].sort_values("lift", ascending=False)
            r = r.assign(x=r.antecedent.map(item_name), y=r.consequent.map(item_name), confidence=r.confidence * 100)
            st.dataframe(r[["x", "y", "confidence", "lift", "support"]].head(25),
                         hide_index=True, width="stretch", height=390,
                         column_config={
                             "x": "If they order", "y": "They also order",
                             "confidence": st.column_config.ProgressColumn("Confidence", format="%.0f%%", min_value=0, max_value=100),
                             "lift": st.column_config.NumberColumn("Lift", format="%.2f"),
                             "support": st.column_config.NumberColumn("Support", format="%.3f")})
            st.caption("Confidence is the share of X orders that also include Y. Lift above 1 means they're ordered together more than chance.")
        with b:
            st.markdown("#### Pairing strength (lift)")
            mat = pd.DataFrame(np.nan, index=items, columns=items)
            for _, rr in rules.iterrows():
                if rr.support >= 0.01:
                    mat.loc[rr.antecedent, rr.consequent] = rr.lift
            labels = [i.replace("_", " ") for i in items]
            fig = go.Figure(go.Heatmap(z=mat.values, x=labels, y=labels,
                                       colorscale=[[0, "#FFFFFF"], [0.15, "#F6D9A8"], [1, "#B5341F"]],
                                       hovertemplate="%{y} + %{x}<br>Lift %{z:.2f}<extra></extra>",
                                       colorbar=dict(thickness=10, outlinewidth=0)))
            fig.update_xaxes(tickangle=-45, showgrid=False)
            fig.update_yaxes(autorange="reversed", showgrid=False)
            show(chart_layout(fig, 430, legend=False), "heat")

# ================= TAB 4: CHURN WATCHLIST =================
with tab_watch:
    if diners is None:
        st.warning("Add diners.csv next to app.py to score your diner base.")
    else:
        f1, f2 = st.columns([2, 1])
        pick = f1.multiselect("Segments", SEGMENTS, default=SEGMENTS)
        top_n = f2.slider("Rows to show", 10, 100, 25, 5)
        w = diners[(diners.p_reorder < thr) & (diners.segment.isin(pick))].sort_values("value_at_risk", ascending=False)

        k1, k2, k3 = st.columns(3)
        k1.metric("Diners to contact", f"{len(w)}")
        k2.metric("Monthly value at risk", f"AED {w.value_at_risk.sum():,.0f}")
        share = w.value_at_risk.head(top_n).sum() / w.value_at_risk.sum() if len(w) else 0
        k3.metric(f"Value covered by the top {top_n}", f"{share:.0%}")

        if len(w):
            w = w.assign(p_pct=w.p_reorder * 100, action=[next_best_action(s, pp, thr, None)[1] for s, pp in zip(w.segment, w.p_reorder)],
                         offer=[bundle_text(s) or "" for s in w.segment])
            cols_ = ["diner_id", "segment", "p_pct", "days_since_last_order", "monthly_value", "value_at_risk", "action", "offer"]
            st.dataframe(w[cols_].head(top_n), hide_index=True, width="stretch", height=430,
                         column_config={
                             "diner_id": "Diner", "segment": "Segment",
                             "p_pct": st.column_config.ProgressColumn("Reorder chance", format="%.0f%%", min_value=0, max_value=100),
                             "days_since_last_order": st.column_config.NumberColumn("Days quiet"),
                             "monthly_value": st.column_config.NumberColumn("Monthly value", format="AED %.0f"),
                             "value_at_risk": st.column_config.NumberColumn("Value at risk", format="AED %.0f"),
                             "action": "Next step", "offer": "Bundle to offer"})
            st.download_button("Download watchlist (CSV)", w[cols_].round(1).rename(columns={"p_pct": "reorder_chance_pct", "action": "next_step", "offer": "bundle_to_offer"}).to_csv(index=False).encode(),
                               file_name="quickbite_churn_watchlist.csv", mime="text/csv")
            st.caption("Ranked by value at risk = (1 − reorder chance) × orders per month × average order value. "
                       "Start at the top: these diners cost QuickBite the most if they leave.")
        else:
            st.success("No diners in the selected segments fall below the at-risk threshold.")
