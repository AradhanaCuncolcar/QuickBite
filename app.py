# app.py — QuickBite Diner Intelligence Dashboard
# ------------------------------------------------------------
# Run locally:   streamlit run app.py
# Needs:         food_models.joblib  (created at the end of the lab notebook) in the same folder
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="QuickBite Diner Intelligence", page_icon="🍔")


@st.cache_resource
def load_models():
    return joblib.load("food_models.joblib")


M = load_models()
scaler, kmeans = M["scaler"], M["kmeans"]
seg_names, clf, FEATURES, reco = M["segment_names"], M["clf"], M["features"], M["reco"]


def analyse(profile: dict):
    """Return (segment name, reorder yes/no, reorder probability, recommended items)."""
    row = pd.DataFrame([profile])
    seg = int(kmeans.predict(scaler.transform(row[FEATURES]))[0])
    name = seg_names.get(seg, f"Segment {seg}")
    reorder = int(clf.predict(row[FEATURES + ["age", "total_spend"]])[0])
    prob = float(clf.predict_proba(row[FEATURES + ["age", "total_spend"]])[0][1])
    return name, reorder, prob, reco.get(name, [])


st.title("🍔 QuickBite — Diner Intelligence")
st.write("Enter a diner's profile. The app returns their **segment**, whether they're likely to **reorder**, "
         "and the **items we'd recommend** to their segment.")

st.sidebar.header("Diner profile")
age       = st.sidebar.slider("Age", 18, 70, 28)
opm       = st.sidebar.slider("Orders per month", 1, 20, 8)
aov       = st.sidebar.slider("Average order value (AED)", 15, 250, 90)
items     = st.sidebar.slider("Average items per order", 1, 10, 3)
recency   = st.sidebar.slider("Days since last order", 0, 60, 10)
discount  = st.sidebar.slider("Discount usage rate", 0.0, 1.0, 0.3, 0.05)
total     = st.sidebar.number_input("Total spend so far (AED)", 50, 20000, 800, step=50)

profile = dict(age=age, orders_per_month=opm, avg_order_value=aov,
               avg_items_per_order=items, days_since_last_order=recency,
               discount_rate=discount, total_spend=total)

name, reorder, prob, items_reco = analyse(profile)

c1, c2 = st.columns(2)
c1.metric("Diner Segment", name)
c2.metric("Will Reorder?", "Yes ✅" if reorder else "At risk ⚠️", f"{prob*100:.0f}% likely")

st.subheader("Recommended items for this diner")
st.write(" · ".join(items_reco) if items_reco else "—")

if not reorder:
    st.info("This diner looks **at risk of churning** — a win-back offer on their favourite items could help.")

st.caption("Segments, reorder model, and recommendations come from the models you trained in the lab notebook.")
