import io, json, zipfile, datetime as dt
import numpy as np, pandas as pd, plotly.express as px, streamlit as st
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="EcoScope Research Agent", page_icon="🌊", layout="wide")
st.markdown("""<style>
.hero{background:linear-gradient(120deg,#0E7C86,#12303A);color:#fff;padding:1.2rem 1.5rem;border-radius:14px;margin-bottom:1rem}
.hero h1{margin:0;font-size:1.7rem}.hero p{margin:.2rem 0 0;opacity:.85}
.warn{background:#FFF4DC;border-left:4px solid #E0A100;padding:.6rem .9rem;border-radius:6px;font-size:.9rem}
</style>""", unsafe_allow_html=True)

MODEL = "llama-3.3-70b-versatile"
PAIRS = {"chl_a": "sat_chl_a", "turbidity": "sat_turbidity", "temp": "sat_temp"}

# ---------- data ----------
def demo():
    rng = np.random.default_rng(42)
    rows, srows = [], []
    for site, (lat, lon, k) in {"S1": (24.50, 118.10, 1.0), "S2": (24.52, 118.12, 1.6), "S3": (24.54, 118.15, 2.3)}.items():
        for d in pd.date_range("2024-01-15", "2025-12-15", freq="MS") + pd.Timedelta(days=14):
            seas = np.sin((d.dayofyear - 100) / 365 * 2 * np.pi)
            temp = 20 + 8 * seas + rng.normal(0, 1)
            tp = max(0.02, 0.05 * k + 0.02 * seas + rng.normal(0, .01))
            tn = max(0.3, 1.2 + 0.3 * k + rng.normal(0, .15))
            chl = max(1, 6 * k + 25 * tp * 10 * (seas > 0) + rng.normal(0, 3))
            tur = max(1, 8 * k + rng.normal(0, 2))
            rows.append(dict(date=d, site=site, lat=lat, lon=lon, chl_a=chl, turbidity=tur, temp=temp, TP=tp, TN=tn))
            sd = d + pd.Timedelta(days=int(rng.integers(-2, 3)))
            srows.append(dict(date=sd, site=site, sat_chl_a=chl * 0.8 + rng.normal(0, 4), sat_turbidity=tur * 1.1 + rng.normal(0, 3),
                              sat_temp=temp + rng.normal(0.5, 1.2), cloud_pct=float(rng.integers(0, 60))))
    return pd.DataFrame(rows), pd.DataFrame(srows)

def read(f):
    return pd.read_excel(f) if f.name.lower().endswith(("xlsx", "xls")) else pd.read_csv(f)

def clean(df, log_to):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    n0 = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["site"] = df["site"].astype(str).str.strip()
    df = df.dropna(subset=["date"]).drop_duplicates()
    for c in df.columns:
        if c not in ("date", "site"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    miss = int(df.isna().sum().sum())
    log_to.append(dict(step="Clean", source=f"{n0} raw rows", processing="Parsed dates, trimmed site names, coerced numerics, dropped duplicates/invalid dates",
                       calculation="-", test="-", result=f"{len(df)} rows kept, {miss} missing cells", interpretation="Missing values left as NaN (not imputed)"))
    return df.sort_values("date").reset_index(drop=True)

def match(f, s, tol, cloud, log_to):
    s = s.copy()
    if "cloud_pct" in s:
        s = s[s["cloud_pct"] <= cloud]
    s["sat_date"] = s["date"]
    m = pd.merge_asof(f.sort_values("date"), s.sort_values("date").drop(columns=[c for c in ("lat", "lon") if c in s]),
                      on="date", by="site", tolerance=pd.Timedelta(days=tol), direction="nearest")
    n_m = int(m["sat_date"].notna().sum())
    log_to.append(dict(step="Match", source="Field x Satellite", processing=f"Cloud <= {cloud}%; nearest-date join per site within ±{tol} d",
                       calculation="pandas.merge_asof", test="-", result=f"{n_m}/{len(f)} field records matched",
                       interpretation="Low match count limits validation power" if n_m < 20 else "Adequate matchups"))
    return m

# ---------- analyses (deterministic) ----------
def validate(m):
    rows = []
    for fv, sv in PAIRS.items():
        if fv in m and sv in m:
            d = m[[fv, sv]].dropna()
            if len(d) >= 5:
                r, p = stats.pearsonr(d[fv], d[sv]); rho, ps = stats.spearmanr(d[fv], d[sv])
                lr = stats.linregress(d[fv], d[sv])
                rows.append(dict(variable=fv, n=len(d), pearson_r=r, pearson_p=p, spearman_rho=rho, spearman_p=ps,
                                 slope=lr.slope, intercept=lr.intercept, R2=lr.rvalue ** 2,
                                 RMSE=float(np.sqrt(((d[sv] - d[fv]) ** 2).mean())), bias=float((d[sv] - d[fv]).mean())))
    return pd.DataFrame(rows).round(4)

def spatial(f):
    cols = [c for c in ("chl_a", "turbidity", "temp", "TP", "TN") if c in f]
    tab = f.groupby("site")[cols].agg(["mean", "std"]).round(3)
    tests = []
    for c in cols:
        g = [x[c].dropna().values for _, x in f.groupby("site") if x[c].notna().sum() > 1]
        if len(g) > 1:
            h, p = stats.kruskal(*g); tests.append(dict(variable=c, kruskal_H=round(h, 3), p=round(p, 5)))
    return tab, pd.DataFrame(tests)

def temporal(f, var="chl_a"):
    ts = f.set_index("date")[var].resample("MS").mean().interpolate(limit=2).dropna()
    out = {"series": ts, "decomp": None}
    if len(ts) >= 24:
        out["decomp"] = seasonal_decompose(ts, period=12, model="additive")
    if len(ts) >= 6:
        lr = stats.linregress(np.arange(len(ts)), ts.values)
        tau, p = stats.kendalltau(np.arange(len(ts)), ts.values)
        out["trend"] = dict(slope_per_month=round(lr.slope, 4), kendall_tau=round(tau, 3), p=round(p, 4))
    return out

def drivers(f):
    d = f[["chl_a", "TP", "TN"]].dropna() if all(c in f for c in ("chl_a", "TP", "TN")) else pd.DataFrame()
    if len(d) < 10:
        return None, None
    z = (d - d.mean()) / d.std()
    fit = sm.OLS(z["chl_a"], sm.add_constant(z[["TP", "TN"]])).fit()
    tab = pd.DataFrame({"std_beta": fit.params, "p": fit.pvalues, "CI_low": fit.conf_int()[0], "CI_high": fit.conf_int()[1]}).round(4)
    sp = {c: dict(zip(("rho", "p"), np.round(stats.spearmanr(d[c], d["chl_a"]), 4))) for c in ("TP", "TN")}
    return tab, dict(R2=round(fit.rsquared, 3), n=len(d), spearman=sp)

def pca(f):
    cols = [c for c in ("chl_a", "turbidity", "temp", "TP", "TN") if c in f]
    d = f[cols].dropna()
    if len(d) < 10 or len(cols) < 3:
        return None
    p = PCA(n_components=2).fit(StandardScaler().fit_transform(d))
    return pd.DataFrame(p.components_.T, index=cols, columns=["PC1", "PC2"]).round(3), np.round(p.explained_variance_ratio_, 3)

def bloom(m, thr):
    d = m.dropna(subset=["chl_a", "sat_chl_a"]).copy()
    if len(d) < 5: return None
    d["field_bloom"], d["sat_bloom"] = d["chl_a"] > thr, d["sat_chl_a"] > thr
    ct = pd.crosstab(d["field_bloom"], d["sat_bloom"]).reindex(index=[False, True], columns=[False, True], fill_value=0)
    acc = (ct.values[0, 0] + ct.values[1, 1]) / ct.values.sum()
    return ct, round(float(acc), 3), int(d["field_bloom"].sum())

# ---------- LLM (low token) ----------
def llm(key, system, user, max_tokens=600):
    from groq import Groq
    r = Groq(api_key=key).chat.completions.create(model=MODEL, temperature=0.2, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content

def rcode(meta):
    return f'''# Reproduces core analysis. Generated {meta["run_at"]}
library(dplyr); library(readr)
f <- read_csv("field_clean.csv"); s <- read_csv("satellite_clean.csv")
# Match: nearest satellite date per site within {meta["date_tolerance_days"]} d, cloud <= {meta["cloud_threshold_pct"]}%
s <- s %>% filter(cloud_pct <= {meta["cloud_threshold_pct"]})
m <- read_csv("matched.csv")
cor.test(m$chl_a, m$sat_chl_a, method="pearson"); cor.test(m$chl_a, m$sat_chl_a, method="spearman")
summary(lm(sat_chl_a ~ chl_a, data=m))
summary(lm(scale(chl_a) ~ scale(TP) + scale(TN), data=f))
kruskal.test(chl_a ~ site, data=f)
'''

# ---------- UI ----------
st.markdown('<div class="hero"><h1>🌊 EcoScope Research Agent</h1><p>Field + satellite water-quality analytics · auditable · reproducible</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Setup")
    key = st.secrets.get("GROQ_API_KEY", "") or st.text_input("Groq API key", type="password")
    use_demo = st.toggle("Use demo data", value=True)
    ff = sf = None
    if not use_demo:
        ff = st.file_uploader("Field data (CSV/XLSX)", type=["csv", "xlsx", "xls"])
        sf = st.file_uploader("Satellite data (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    st.subheader("Processing metadata")
    prod = st.text_input("Satellite product", "Sentinel-2 MSI L2A")
    atm = st.text_input("Atmospheric correction", "Sen2Cor")
    algo = st.text_input("Chl-a algorithm", "OC3/NDCI (specify)")
    res = st.number_input("Spatial resolution (m)", 10, 1000, 20)
    cloud = st.slider("Cloud threshold (%)", 0, 100, 30)
    tol = st.slider("Date tolerance (± days)", 0, 10, 3)
    thr = st.number_input("Bloom threshold Chl-a (µg/L)", 5.0, 100.0, 20.0)

question = st.text_area("Research question", "Is phosphorus or nitrogen more strongly associated with phytoplankton biomass (Chl-a), and does satellite Chl-a agree with field data?", height=80)
if st.button("🚀 Run research agent", type="primary"):
    log = []
    try:
        fr, sr = demo() if use_demo else (read(ff), read(sf))
    except Exception as e:
        st.error(f"Please upload both files with required columns. ({e})"); st.stop()
    f, s = clean(fr, log), clean(sr, log)
    m = match(f, s, tol, cloud, log)
    R = dict(f=f, s=s, m=m, val=validate(m), bloom=bloom(m, thr), drv=drivers(f), pca=pca(f), tmp=temporal(f))
    R["spa"] = spatial(f)
    v = R["val"]
    for _, r in v.iterrows():
        log.append(dict(step="Validate", source="Matched field–satellite pairs", processing=f"n={int(r.n)}", calculation="y_sat = a + b·y_field; RMSE; bias",
                        test="Pearson, Spearman, OLS", result=f"{r.variable}: r={r.pearson_r}, R²={r.R2}, RMSE={r.RMSE}, bias={r.bias}",
                        interpretation="Satellite ≈ field" if r.R2 > .5 else "Weak agreement: do not treat satellite as ground truth"))
    if R["drv"][0] is not None:
        t, i = R["drv"]
        log.append(dict(step="Drivers", source="Field TP, TN, Chl-a", processing="z-score standardisation", calculation="z(Chl) = b0 + b1·z(TP) + b2·z(TN)",
                        test="OLS, Spearman", result=f"β_TP={t.loc['TP','std_beta']} (p={t.loc['TP','p']}), β_TN={t.loc['TN','std_beta']} (p={t.loc['TN','p']}), R²={i['R2']}",
                        interpretation="Larger |β| = stronger association (not causation)"))
    R["meta"] = dict(run_at=dt.datetime.now().isoformat(timespec="seconds"), satellite_product=prod, atmospheric_correction=atm, algorithm=algo,
                     spatial_resolution_m=res, cloud_threshold_pct=cloud, date_tolerance_days=tol, bloom_threshold_ugL=thr,
                     processing_level="L2A" if "L2" in prod else "see product", stats="Pearson, Spearman, OLS, Kruskal-Wallis, Kendall, PCA, STL-additive decomposition",
                     coordinates=f[["site", "lat", "lon"]].drop_duplicates().to_dict("records") if "lat" in f else "n/a", demo_data=use_demo, question=question)
    R["log"] = log
    # compact payload for LLM (low token)
    pay = dict(validation=v.to_dict("records"), drivers=None if R["drv"][0] is None else dict(table=R["drv"][0].to_dict(), info=R["drv"][1]),
               trend=R["tmp"].get("trend"), bloom_accuracy=None if R["bloom"] is None else R["bloom"][1], site_tests=R["spa"][1].to_dict("records"))
    R["payload"] = pay
    R["narr"] = None
    if key:
        try:
            with st.spinner("Groq is interpreting the statistics…"):
                R["narr"] = llm(key, "You are an aquatic ecologist. Use ONLY the supplied statistics. Cite numbers. List limitations (clouds, atmospheric correction, sediment, depth, adjacency, sensor). Output: ## Interpretation, ## Limitations, ## Methods (draft), ## Results (draft). Be concise.",
                                f"Question: {question}\nMeta: {json.dumps(R['meta'], default=str)[:900]}\nStats: {json.dumps(pay, default=str)[:3500]}", 900)
        except Exception as e:
            st.warning(f"Groq call failed: {e}")
    st.session_state["R"] = R

R = st.session_state.get("R")
if not R:
    st.info("Choose demo or upload data in the sidebar, then run the agent."); st.stop()

st.markdown('<div class="warn">⚠️ Satellite estimates are not ground truth. Conclusions below are validated against field data; check clouds, atmospheric correction, sediment, depth and adjacency effects.</div>', unsafe_allow_html=True)
t1, t2, t3, t4, t5, t6 = st.tabs(["✅ Validation", "🗺️ Spatial & Temporal", "🧪 Drivers (N vs P)", "🟢 Bloom", "🤖 Agent report", "🔍 Audit & Export"])

with t1:
    st.dataframe(R["val"], use_container_width=True)
    for fv, sv in PAIRS.items():
        if fv in R["m"] and sv in R["m"] and R["m"][[fv, sv]].dropna().shape[0] > 4:
            st.plotly_chart(px.scatter(R["m"], x=fv, y=sv, color="site", trendline="ols", title=f"{fv}: field vs satellite"), use_container_width=True)
    with st.expander("Matched dataset"): st.dataframe(R["m"])

with t2:
    tab, tests = R["spa"]
    c1, c2 = st.columns(2)
    c1.subheader("Site means ± SD"); c1.dataframe(tab)
    c2.subheader("Kruskal–Wallis between sites"); c2.dataframe(tests)
    var = st.selectbox("Variable", [c for c in ("chl_a", "turbidity", "temp") if c in R["f"]])
    st.plotly_chart(px.line(R["f"], x="date", y=var, color="site", markers=True), use_container_width=True)
    st.plotly_chart(px.box(R["f"], x="site", y=var, color="site"), use_container_width=True)
    tmp = R["tmp"]
    if tmp.get("trend"): st.write("Trend (monthly Chl-a):", tmp["trend"])
    if tmp["decomp"] is not None:
        dc = tmp["decomp"]
        st.plotly_chart(px.line(pd.DataFrame({"trend": dc.trend, "seasonal": dc.seasonal, "residual": dc.resid}).reset_index().melt("date"), x="date", y="value", facet_row="variable", title="Seasonal decomposition (Chl-a)"), use_container_width=True)
    else: st.caption("Seasonal decomposition needs ≥24 months.")
    if R["pca"] is not None:
        st.subheader("PCA loadings"); st.dataframe(R["pca"][0]); st.caption(f"Explained variance: {R['pca'][1]}")

with t3:
    t, i = R["drv"]
    if t is None: st.info("Need TP, TN, chl_a columns with ≥10 complete rows.")
    else:
        st.dataframe(t); st.json(i)
        st.plotly_chart(px.scatter(R["f"], x="TP", y="chl_a", color="site", trendline="ols"), use_container_width=True)
        st.plotly_chart(px.scatter(R["f"], x="TN", y="chl_a", color="site", trendline="ols"), use_container_width=True)

with t4:
    if R["bloom"] is None: st.info("Not enough matched data.")
    else:
        ct, acc, nb = R["bloom"]
        st.metric("Field bloom records", nb); st.metric("Satellite bloom classification accuracy", acc)
        st.dataframe(ct.rename(index={False: "field: no", True: "field: bloom"}, columns={False: "sat: no", True: "sat: bloom"}))
        st.caption("Cyanobacteria-specific detection needs phycocyanin/spectral indices (Phase 2); this flags high Chl-a only.")

with t5:
    st.write("**Question:**", R["meta"]["question"])
    st.markdown(R["narr"] or "_Add a Groq key to generate the interpretation, limitations and Methods/Results drafts._")

with t6:
    st.subheader("Step-by-step audit trail")
    st.dataframe(pd.DataFrame(R["log"]), use_container_width=True)
    st.subheader("Recorded metadata"); st.json(R["meta"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("field_clean.csv", R["f"].to_csv(index=False)); z.writestr("satellite_clean.csv", R["s"].to_csv(index=False))
        z.writestr("matched.csv", R["m"].to_csv(index=False)); z.writestr("validation_table.csv", R["val"].to_csv(index=False))
        if R["drv"][0] is not None: z.writestr("drivers_table.csv", R["drv"][0].to_csv())
        z.writestr("audit_trail.csv", pd.DataFrame(R["log"]).to_csv(index=False))
        z.writestr("metadata.json", json.dumps(R["meta"], indent=2, default=str)); z.writestr("analysis.R", rcode(R["meta"]))
        z.writestr("report.md", f"# Report\n\n**Question:** {R['meta']['question']}\n\n{R['narr'] or ''}\n\n## Validation\n{R['val'].to_string(index=False)}\n")
        for fv, sv in PAIRS.items():
            if fv in R["m"] and sv in R["m"] and R["m"][[fv, sv]].dropna().shape[0] > 4:
                z.writestr(f"fig_{fv}_validation.html", px.scatter(R["m"], x=fv, y=sv, color="site", trendline="ols").to_html())
    st.download_button("⬇️ Download reproducible package (.zip)", buf.getvalue(), "ecoscope_analysis.zip", "application/zip", type="primary")
