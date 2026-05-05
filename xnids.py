# ============================================================
#  X-NIDS — xnids.py
#  Network Intrusion Detection System — Streamlit Dashboard
#  Model   : Random Forest (UNSW-NB15)
#  Extras  : SHAP explainability, Plotly charts, CSV export
# ============================================================

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="X-NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  GLOBALS
# ─────────────────────────────────────────────────────────────
now          = datetime.now()
MAX_ROWS     = 500_000
SHAP_SAMPLE  = 200      # max packets used for SHAP computation
MODEL_FILE   = "x_nids_model.joblib"
PREP_FILE    = "preprocessor.joblib"
METRICS_FILE = "model_metrics.json"

# ─────────────────────────────────────────────────────────────
#  CSS — terminal / cyberpunk theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700&display=swap');

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stHeader"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}

*, *::before, *::after {
    font-family: 'Share Tech Mono', monospace !important;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #000000 !important;
    color: #00ff41 !important;
}

.main .block-container {
    background-color: #000000 !important;
    padding: 1.5rem 2rem !important;
    max-width: 100% !important;
}

[data-testid="stSidebar"] {
    background-color: #030303 !important;
    border-right: 1px solid #003300 !important;
}
[data-testid="stSidebar"] * { color: #00ff41 !important; }
[data-testid="stSidebar"] code {
    background: #001a00 !important;
    color: #00ff41 !important;
    border: 1px solid #002200 !important;
    border-radius: 0 !important;
    padding: 1px 6px !important;
    font-size: 11px !important;
}

h1, h2, h3, h4 {
    font-family: 'Orbitron', monospace !important;
    color: #00ff41 !important;
    text-shadow: 0 0 8px rgba(0,255,65,0.5) !important;
    letter-spacing: 3px !important;
}

[data-testid="metric-container"] {
    background: #030303 !important;
    border: 1px solid #003300 !important;
    border-radius: 0 !important;
    padding: 14px 16px !important;
    box-shadow: 0 0 10px rgba(0,255,65,0.1) !important;
    transition: box-shadow 0.2s !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 0 20px rgba(0,255,65,0.25) !important;
    border-color: #00ff41 !important;
}
[data-testid="stMetricLabel"] {
    color: #005500 !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #00ff41 !important;
    font-size: 22px !important;
    font-weight: 900 !important;
}

[data-testid="stFileUploader"] {
    background: #030303 !important;
    border: 1px dashed #003300 !important;
    border-radius: 0 !important;
    padding: 16px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #00ff41 !important; }
[data-testid="stFileUploader"] * { color: #00ff41 !important; background: transparent !important; }
[data-testid="stFileUploader"] section { background: #030303 !important; border: none !important; }
[data-testid="stFileUploader"] button {
    background: transparent !important;
    color: #00ff41 !important;
    border: 1px solid #003300 !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploaderDropzone"] { background: #030303 !important; border: none !important; }

.stButton > button,
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: #00ff41 !important;
    border: 1px solid #003300 !important;
    border-radius: 0 !important;
    letter-spacing: 2px !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
    padding: 8px 20px !important;
}
.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover {
    background: #00ff41 !important;
    color: #000000 !important;
    border-color: #00ff41 !important;
    box-shadow: 0 0 15px rgba(0,255,65,0.4) !important;
}

[data-testid="stDataFrame"] { border: 1px solid #002200 !important; }

[data-testid="stSlider"] * { color: #00ff41 !important; }
[data-baseweb="track-background"] { background: #001a00 !important; }
[data-baseweb="track-foreground"] { background: #00ff41 !important; }
[data-baseweb="slider"] [role="slider"] {
    background: #00ff41 !important;
    border: 2px solid #000000 !important;
    box-shadow: 0 0 8px #00ff41 !important;
}

[data-testid="stCheckbox"] * { color: #00ff41 !important; }

[data-baseweb="tab-list"] {
    background: #000000 !important;
    border-bottom: 1px solid #002200 !important;
    gap: 0 !important;
}
button[data-baseweb="tab"] {
    color: #004400 !important;
    background: #000000 !important;
    border-radius: 0 !important;
    letter-spacing: 2px !important;
    padding: 10px 28px !important;
    border-right: 1px solid #001500 !important;
    font-size: 11px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00ff41 !important;
    background: #030303 !important;
    border-bottom: 2px solid #00ff41 !important;
    text-shadow: 0 0 8px rgba(0,255,65,0.5) !important;
}

.stSpinner > div { border-top-color: #00ff41 !important; }
[data-testid="stAlert"] {
    background: #030303 !important;
    border-radius: 0 !important;
    border-left: 3px solid #00ff41 !important;
}

hr { border-color: #001500 !important; }

::-webkit-scrollbar       { width: 3px; background: #000000; }
::-webkit-scrollbar-thumb { background: #002200; }
::-webkit-scrollbar-thumb:hover { background: #00ff41; }

p, span, div, li, td, th, label { color: #00ff41 !important; }
code {
    background: #001a00 !important;
    color: #00ff41 !important;
    border: 1px solid #002200 !important;
    border-radius: 0 !important;
    padding: 1px 5px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, feature_cols: list, label_encoders: dict) -> pd.DataFrame:
    """
    Align df to feature_cols, apply saved LabelEncoders,
    fill NaN, return float DataFrame.
    Unseen categorical values → -1 (safe fallback).
    """
    X = df[feature_cols].fillna(0).copy()
    for col, le in label_encoders.items():
        if col in X.columns:
            known = set(le.classes_)
            X[col] = X[col].astype(str).apply(
                lambda v: int(le.transform([v])[0]) if v in known else -1
            )
    # Coerce any remaining object columns
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.Categorical(X[col]).codes
    return X.astype(float)


def validate_csv(df: pd.DataFrame, feature_cols: list) -> list[str]:
    """Return list of missing required columns."""
    return [c for c in feature_cols if c not in df.columns]


def plotly_layout(title: str, height: int = 380, left_margin: int = 50) -> dict:
    return dict(
        title=dict(text=title, font=dict(color="#00ff41", size=12,
                   family="Share Tech Mono"), x=0),
        paper_bgcolor="#000000",
        plot_bgcolor="#030303",
        font=dict(color="#00ff41", family="Share Tech Mono", size=11),
        xaxis=dict(gridcolor="#001500", color="#00ff41",
                   linecolor="#002200", zerolinecolor="#001500"),
        yaxis=dict(gridcolor="#001500", color="#00ff41",
                   linecolor="#002200", zerolinecolor="#001500"),
        height=height,
        margin=dict(t=40, b=40, l=left_margin, r=20),
        legend=dict(bgcolor="#030303", bordercolor="#002200",
                    borderwidth=1, font=dict(color="#00ff41")),
    )


def compute_shap(model, X_sample: pd.DataFrame):
    """
    Safely compute mean absolute SHAP values.
    Handles both old (list) and new (ndarray) shap output formats.
    Returns (importance_array, feature_names) — always same length.
    """
    explainer   = shap.TreeExplainer(model)
    sv          = explainer.shap_values(X_sample)

    if isinstance(sv, list):          # old format: [class0, class1]
        shap_matrix = np.abs(sv[1])
    elif sv.ndim == 3:                # new format: (n, features, classes)
        shap_matrix = np.abs(sv[:, :, 1])
    else:                             # new format: (n, features)
        shap_matrix = np.abs(sv)

    imp   = shap_matrix.mean(axis=0)
    names = X_sample.columns.tolist()
    n     = min(len(names), len(imp))
    return imp[:n], names[:n]


# ─────────────────────────────────────────────────────────────
#  LOAD MODEL, PREPROCESSOR, METRICS  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model          = joblib.load(MODEL_FILE)
    prep           = joblib.load(PREP_FILE)
    feature_cols   = prep["feature_cols"]
    label_encoders = prep.get("label_encoders", {})
    return model, feature_cols, label_encoders


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    if not os.path.exists(METRICS_FILE):
        return None
    with open(METRICS_FILE) as f:
        return json.load(f)


# ── Try loading ───────────────────────────────────────────────
try:
    model, feature_cols, label_encoders = load_artifacts()
    model_ok  = True
    model_err = ""
except Exception as err:
    model_ok      = False
    model_err     = str(err)
    feature_cols  = []
    label_encoders = {}

metrics = load_metrics()


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Branding ─────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:16px 0;
                border-bottom:1px solid #002200; margin-bottom:18px;'>
        <p style='font-family:Orbitron; font-size:15px;
                  letter-spacing:4px; margin:0; color:#00ff41;
                  text-shadow: 0 0 8px rgba(0,255,65,0.5);'>
            ⚙ X-NIDS
        </p>
        <p style='font-size:9px; color:#003300; letter-spacing:2px; margin:4px 0 0;'>
            CONTROL PANEL
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Status ───────────────────────────────────────────────
    status_icon = "🟢 `ONLINE`" if model_ok else "🔴 `ERROR`"
    st.markdown(f"**STATUS** &nbsp; {status_icon}")
    st.markdown(f"**TIME** &nbsp;&nbsp;&nbsp; `{now.strftime('%H:%M:%S')}`")
    st.markdown(f"**DATE** &nbsp;&nbsp;&nbsp; `{now.strftime('%Y-%m-%d')}`")

    # ── Model info (dynamic from metrics.json) ────────────────
    st.markdown("---")
    st.markdown("**MODEL**")
    st.markdown("- Classifier `Random Forest`")
    st.markdown("- Dataset &nbsp;&nbsp; `UNSW-NB15`")

    if metrics:
        m = metrics.get("metrics", {})
        acc_pct = f"{m.get('accuracy', 0)*100:.2f}%"
        f1_val  = f"{m.get('f1_attack', 0):.2f}"
        build   = metrics.get("build", "—")
        n_feat  = metrics.get("n_features", len(feature_cols))
        st.markdown(f"- Accuracy &nbsp;`{acc_pct}`")
        st.markdown(f"- F1 Attack &nbsp;`{f1_val}`")
        st.markdown(f"- Features &nbsp; `{n_feat}`")
        st.markdown(f"- Build &nbsp;&nbsp;&nbsp;&nbsp; `{build}`")
    else:
        st.markdown(f"- Features &nbsp; `{len(feature_cols) if model_ok else '—'}`")
        st.markdown("- *(run train_model.py for metrics)*")

    # ── Settings ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**SETTINGS**")
    threshold     = st.slider("ALERT THRESHOLD", 0.1, 0.9, 0.5, 0.05)
    show_shap     = st.checkbox("SHAP FORENSICS",  True)
    show_charts   = st.checkbox("THREAT CHARTS",   True)
    show_timeline = st.checkbox("THREAT TIMELINE", True)

    # ── Legend ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("🔴 HIGH &nbsp;&nbsp; ≥ `0.80`")
    st.markdown("🟡 MEDIUM ≥ `{:.2f}`".format(threshold))
    st.markdown("🟢 LOW &nbsp;&nbsp;&nbsp; < `{:.2f}`".format(threshold))

    # ── Footer ───────────────────────────────────────────────
    build_label = metrics.get("build", "—") if metrics else "—"
    st.markdown(f"""
    <div style='margin-top:24px; text-align:center;
                border-top:1px solid #001500; padding-top:14px;'>
        <p style='font-size:9px; color:#002200; letter-spacing:1px; margin:0;'>
            X-NIDS ENTERPRISE v9.5<br>
            RANDOM FOREST + SHAP<br>
            BUILD {build_label}
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
acc_display = (
    f"{metrics['metrics']['accuracy']*100:.2f}%"
    if metrics else "—"
)
auc_display = (
    f"AUC {metrics['metrics']['auc_roc']:.4f}"
    if metrics else ""
)

st.markdown(f"""
<div style='text-align:center; border-bottom:1px solid #002200;
            padding-bottom:20px; margin-bottom:28px;'>
    <h1 style='font-size:42px; margin:0; letter-spacing:10px;'>
        X-NIDS ENTERPRISE
    </h1>
    <p style='letter-spacing:6px; font-size:11px; margin:8px 0 4px; color:#00ff41;'>
        [ NETWORK INTRUSION DETECTION SYSTEM v9.5 ]
    </p>
    <p style='color:#002200; font-size:10px; letter-spacing:2px; margin:0;'>
        UNSW-NB15 &nbsp;·&nbsp; RANDOM FOREST &nbsp;·&nbsp;
        ACCURACY {acc_display} &nbsp;·&nbsp; {auc_display}
    </p>
</div>
""", unsafe_allow_html=True)

# ── Model load error ─────────────────────────────────────────
if not model_ok:
    st.error(f"❌ MODEL LOAD FAILED — {model_err}")
    st.info("Run `python train_model.py` to generate model files.")
    st.stop()


# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📡  TRAFFIC ANALYZER",
    "📊  MODEL METRICS",
    "ℹ   SYSTEM INFO",
])


# ════════════════════════════════════════════════════════════
#  TAB 1 — TRAFFIC ANALYZER
# ════════════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader(
        "DROP NETWORK CAPTURE CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    # ── Empty state ──────────────────────────────────────────
    if uploaded is None:
        st.markdown("""
        <div style='text-align:center; padding:70px 20px;
                    border:1px dashed #001a00; margin:24px 0;'>
            <p style='font-size:36px; margin:0; color:#002200;'>⬆</p>
            <p style='letter-spacing:4px; font-size:13px; margin:10px 0 6px;'>
                AWAITING NETWORK CAPTURE
            </p>
            <p style='color:#002200; font-size:10px; letter-spacing:1px; margin:0;'>
                UPLOAD CSV FILE TO BEGIN THREAT ANALYSIS
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Load file ────────────────────────────────────────
        with st.spinner("LOADING FILE..."):
            try:
                df_raw = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"❌ FAILED TO READ FILE: {e}")
                st.stop()

        # ── File size limit ──────────────────────────────────
        if len(df_raw) > MAX_ROWS:
            st.warning(
                f"⚠ FILE TRUNCATED TO {MAX_ROWS:,} ROWS FOR PERFORMANCE "
                f"(uploaded: {len(df_raw):,})"
            )
            df_raw = df_raw.head(MAX_ROWS)

        # ── Validate schema ──────────────────────────────────
        missing_cols = validate_csv(df_raw, feature_cols)
        if missing_cols:
            st.error(
                f"❌ MISSING {len(missing_cols)} REQUIRED FEATURES:\n"
                f"`{missing_cols[:8]}`"
                + (" ..." if len(missing_cols) > 8 else "")
            )
            with st.expander("SHOW ALL MISSING COLUMNS"):
                st.code(str(missing_cols))
            st.stop()

        # ── Inference ────────────────────────────────────────
        with st.spinner("RUNNING INFERENCE..."):
            X     = preprocess(df_raw, feature_cols, label_encoders)
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]

        # ── Build output DataFrame ───────────────────────────
        df_out = df_raw.copy()
        df_out["THREAT_SCORE"] = probs.round(4)
        df_out["PREDICTION"]   = ["⚠ ATTACK" if p else "✓ NORMAL" for p in preds]
        df_out["ALERT"] = [
            "🔴 HIGH"   if p >= 0.8      else
            "🟡 MEDIUM" if p >= threshold else
            "🟢 LOW"
            for p in probs
        ]

        total  = len(df_out)
        n_high = int((probs >= 0.8).sum())
        n_med  = int(((probs >= threshold) & (probs < 0.8)).sum())
        n_low  = int((probs < threshold).sum())
        n_atk  = int(preds.sum())
        det_rt = n_atk / total * 100 if total > 0 else 0.0

        # ── KPI metrics ──────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("TOTAL PACKETS",  f"{total:,}")
        c2.metric("🔴 HIGH",        f"{n_high:,}")
        c3.metric("🟡 MEDIUM",      f"{n_med:,}")
        c4.metric("🟢 CLEAR",       f"{n_low:,}")
        c5.metric("DETECTION RATE", f"{det_rt:.1f}%")

        st.markdown("---")

        # ── Packet table ─────────────────────────────────────
        st.markdown("##### ▶ PACKET ANALYSIS")
        st.caption(f"Showing 1,000 of {total:,} packets — export full report below")

        view_cols  = ["ALERT", "PREDICTION", "THREAT_SCORE"] + feature_cols[:6]
        disp_df    = df_out[view_cols].head(1000).reset_index(drop=True)
        st.dataframe(disp_df, use_container_width=True, height=380)

        # ── Export ───────────────────────────────────────────
        export_csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 EXPORT FULL REPORT",
            data=export_csv,
            file_name=f"xnids_{now.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        # ── Charts ───────────────────────────────────────────
        if show_charts:
            st.markdown("---")
            st.markdown("##### ▶ THREAT DISTRIBUTION")
            col1, col2 = st.columns(2)

            with col1:
                fig1 = go.Figure()
                fig1.add_trace(go.Histogram(
                    x=probs, nbinsx=50,
                    marker_color="#00ff41",
                    marker_line=dict(color="#000000", width=0.3),
                    opacity=0.9, name="Score",
                    hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra></extra>",
                ))
                fig1.add_vline(
                    x=threshold, line_dash="dash",
                    line_color="#ff3333", line_width=1.5,
                    annotation_text=f"  THRESHOLD {threshold:.2f}",
                    annotation_font=dict(color="#ff3333", size=10),
                )
                fig1.update_layout(**plotly_layout("THREAT SCORE DISTRIBUTION"))
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = go.Figure(go.Pie(
                    labels=["HIGH", "MEDIUM", "LOW"],
                    values=[max(n_high, 0), max(n_med, 0), max(n_low, 0)],
                    marker=dict(
                        colors=["#cc0000", "#aaaa00", "#00cc33"],
                        line=dict(color="#000000", width=2),
                    ),
                    hole=0.6,
                    textfont=dict(size=11, family="Share Tech Mono"),
                    hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
                    direction="clockwise",
                    sort=False,
                ))
                lo2 = plotly_layout("ALERT BREAKDOWN")
                lo2["margin"] = dict(t=40, b=20, l=20, r=20)
                fig2.update_layout(**lo2)
                st.plotly_chart(fig2, use_container_width=True)

        # ── Timeline ─────────────────────────────────────────
        if show_timeline:
            st.markdown("---")
            st.markdown("##### ▶ THREAT TIMELINE")

            if len(probs) > 5000:
                idx    = np.linspace(0, len(probs) - 1, 5000, dtype=int)
                t_x, t_y = idx, probs[idx]
                t_note = " (downsampled 5k pts)"
            else:
                t_x, t_y = list(range(len(probs))), probs
                t_note   = ""

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=t_x, y=t_y,
                mode="lines",
                line=dict(color="#00ff41", width=1),
                fill="tozeroy",
                fillcolor="rgba(0,255,65,0.04)",
                name="Score",
                hovertemplate="Packet #%{x}<br>Score: %{y:.4f}<extra></extra>",
            ))
            fig3.add_hrect(
                y0=0.8, y1=1.0,
                fillcolor="rgba(200,0,0,0.05)",
                line_width=0,
            )
            fig3.add_hline(
                y=threshold, line_dash="dash",
                line_color="#ff3333", line_width=1,
                annotation_text="  THRESHOLD",
                annotation_font=dict(color="#ff3333", size=10),
            )
            lo3 = plotly_layout(
                f"PACKET THREAT SCORE TIMELINE{t_note}", height=280
            )
            lo3["yaxis"]["range"] = [0, 1]
            lo3["xaxis"]["title"] = dict(
                text="PACKET INDEX", font=dict(size=10, color="#003300")
            )
            fig3.update_layout(**lo3)
            st.plotly_chart(fig3, use_container_width=True)

        # ── SHAP ─────────────────────────────────────────────
        if show_shap:
            st.markdown("---")
            st.markdown("##### ▶ FORENSIC ANALYSIS — SHAP")

            attack_idx = np.where(preds == 1)[0]
            normal_idx = np.where(preds == 0)[0]
            n_a = min(150, len(attack_idx))
            n_n = min(50,  len(normal_idx))

            parts = []
            if n_a > 0:
                parts.append(
                    np.random.choice(attack_idx, n_a, replace=False)
                )
            if n_n > 0:
                parts.append(
                    np.random.choice(normal_idx, n_n, replace=False)
                )

            if not parts:
                st.warning("⚠ NO PACKETS AVAILABLE FOR SHAP ANALYSIS.")
            else:
                sample_idx = np.concatenate(parts).astype(int)
                X_sample   = X.iloc[sample_idx]

                with st.spinner(
                    f"COMPUTING SHAP VALUES ON {len(X_sample)} PACKETS..."
                ):
                    try:
                        imp, shap_names = compute_shap(model, X_sample)

                        imp_df = (
                            pd.DataFrame({"feature": shap_names, "importance": imp})
                            .sort_values("importance", ascending=True)
                            .tail(15)
                        )

                        fig4 = go.Figure(go.Bar(
                            x=imp_df["importance"],
                            y=imp_df["feature"],
                            orientation="h",
                            marker=dict(
                                color=imp_df["importance"],
                                colorscale=[
                                    [0.0, "#001500"],
                                    [0.4, "#004400"],
                                    [0.7, "#008800"],
                                    [1.0, "#00ff41"],
                                ],
                                showscale=True,
                                colorbar=dict(
                                    title=dict(
                                        text="IMPACT",
                                        font=dict(color="#00ff41", size=10),
                                    ),
                                    tickfont=dict(color="#003300", size=9),
                                    outlinecolor="#002200",
                                    outlinewidth=0.5,
                                    thickness=10,
                                ),
                            ),
                            hovertemplate=(
                                "<b>%{y}</b><br>SHAP: %{x:.5f}<extra></extra>"
                            ),
                        ))
                        lo4 = plotly_layout(
                            f"TOP 15 THREAT INDICATORS  "
                            f"(SHAP · n={len(X_sample)})",
                            480, 150,
                        )
                        lo4["xaxis"]["title"] = dict(
                            text="MEAN |SHAP VALUE|",
                            font=dict(size=10, color="#003300"),
                        )
                        lo4["margin"]["r"] = 60
                        fig4.update_layout(**lo4)
                        st.plotly_chart(fig4, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ SHAP ERROR: {e}")
                        st.info(
                            "Disable SHAP FORENSICS in the sidebar "
                            "if the error persists."
                        )


# ════════════════════════════════════════════════════════════
#  TAB 2 — MODEL METRICS  (dynamic from model_metrics.json)
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("##### ▶ PERFORMANCE METRICS")

    if metrics:
        m = metrics.get("metrics", {})

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ACCURACY",  f"{m.get('accuracy', 0)*100:.2f}%")
        m2.metric("AUC-ROC",   f"{m.get('auc_roc', 0):.4f}")
        m3.metric("PRECISION", f"{m.get('precision', 0):.4f}")
        m4.metric("RECALL",    f"{m.get('recall', 0):.4f}")
        m5.metric("F1 ATTACK", f"{m.get('f1_attack', 0):.4f}")

        st.markdown("---")

        # ── FPR / FNR ────────────────────────────────────────
        st.markdown("##### ▶ FALSE ALARM RATES")
        f1c, f2c, f3c, f4c = st.columns(4)
        f1c.metric("FALSE POSITIVE RATE",
                   f"{m.get('fpr', 0)*100:.2f}%",
                   delta="target < 5%",
                   delta_color="off")
        f2c.metric("FALSE NEGATIVE RATE",
                   f"{m.get('fnr', 0)*100:.2f}%",
                   delta="target < 5%",
                   delta_color="off")
        f3c.metric("TRUE POSITIVES",  f"{m.get('tp', 0):,}")
        f4c.metric("TRUE NEGATIVES",  f"{m.get('tn', 0):,}")

        # ── Confusion matrix ─────────────────────────────────
        st.markdown("---")
        st.markdown("##### ▶ CONFUSION MATRIX")
        tn_ = m.get("tn", 0)
        fp_ = m.get("fp", 0)
        fn_ = m.get("fn", 0)
        tp_ = m.get("tp", 0)

        cm_fig = go.Figure(go.Heatmap(
            z=[[tn_, fp_], [fn_, tp_]],
            x=["PRED NORMAL", "PRED ATTACK"],
            y=["REAL NORMAL", "REAL ATTACK"],
            text=[[f"{tn_:,}", f"{fp_:,}"],
                  [f"{fn_:,}", f"{tp_:,}"]],
            texttemplate="%{text}",
            colorscale=[[0, "#000000"], [0.5, "#003300"], [1, "#00ff41"]],
            showscale=False,
            hovertemplate="%{y} → %{x}: %{text}<extra></extra>",
        ))
        lo_cm = plotly_layout("CONFUSION MATRIX", height=300, left_margin=120)
        lo_cm["xaxis"]["side"] = "top"
        cm_fig.update_layout(**lo_cm)
        st.plotly_chart(cm_fig, use_container_width=True)

        # ── Classification report ────────────────────────────
        st.markdown("---")
        st.markdown("##### ▶ CLASSIFICATION REPORT")
        cr = metrics.get("classification_report", {})
        rows = []
        for cls in ["Normal", "Attack", "macro avg", "weighted avg"]:
            if cls in cr:
                r = cr[cls]
                rows.append({
                    "CLASS":     cls.upper(),
                    "PRECISION": f"{r.get('precision', 0):.4f}",
                    "RECALL":    f"{r.get('recall', 0):.4f}",
                    "F1":        f"{r.get('f1-score', 0):.4f}",
                    "SUPPORT":   f"{int(r.get('support', 0)):,}",
                })
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        # ── Attack types ─────────────────────────────────────
        attack_types = metrics.get("attack_types", [])
        if attack_types:
            st.markdown("---")
            st.markdown("##### ▶ ATTACK CATEGORIES IN DATASET")
            st.code("  ·  ".join(attack_types))

    else:
        st.warning(
            "⚠ `model_metrics.json` NOT FOUND — "
            "run `python train_model.py` to generate it."
        )

    # ── Dataset stats ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### ▶ DATASET STATS")
    d1, d2, d3, d4 = st.columns(4)
    n_train = metrics.get("n_train", "—") if metrics else "—"
    n_test  = metrics.get("n_test",  "—") if metrics else "—"
    n_total = (
        f"{(metrics['n_train'] + metrics['n_test']):,}"
        if metrics else "—"
    )
    d1.metric("TOTAL",    n_total)
    d2.metric("TRAIN",    f"{n_train:,}" if isinstance(n_train, int) else n_train)
    d3.metric("TEST",     f"{n_test:,}"  if isinstance(n_test,  int) else n_test)
    d4.metric("FEATURES", str(len(feature_cols)))

    # ── RF Feature importance ─────────────────────────────────
    st.markdown("---")
    st.markdown("##### ▶ FEATURE IMPORTANCE (TOP 20)")
    try:
        fi_df = (
            pd.DataFrame({
                "feature":    feature_cols,
                "importance": model.feature_importances_,
            })
            .sort_values("importance", ascending=True)
            .tail(20)
        )

        fig5 = go.Figure(go.Bar(
            x=fi_df["importance"],
            y=fi_df["feature"],
            orientation="h",
            marker=dict(
                color=fi_df["importance"],
                colorscale=[[0, "#001500"], [0.5, "#006600"], [1, "#00ff41"]],
            ),
            hovertemplate="<b>%{y}</b><br>%{x:.5f}<extra></extra>",
        ))
        lo5 = plotly_layout(
            "RANDOM FOREST FEATURE IMPORTANCES (GINI)", 540, 160
        )
        fig5.update_layout(**lo5)
        st.plotly_chart(fig5, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠ FEATURE IMPORTANCE ERROR: {e}")


# ════════════════════════════════════════════════════════════
#  TAB 3 — SYSTEM INFO
# ════════════════════════════════════════════════════════════
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### ▶ ABOUT X-NIDS")
        st.markdown("""
**X-NIDS** is a production-grade AI-powered Network Intrusion
Detection System trained on the **UNSW-NB15** benchmark dataset.

Classifies network packets as **Normal** or **Attack** using
a Random Forest model with **SHAP** explainability for
SOC analyst workflows.

---
**CAPABILITIES**
- Binary classification: Normal / Attack
- Multi-level alert triage: HIGH / MEDIUM / LOW
- SHAP-based forensic explainability
- Threat timeline visualization
- CSV export with timestamps
- Configurable alert threshold
- Dynamic metrics from `model_metrics.json`
        """)

    with col_b:
        st.markdown("##### ▶ DATASET + STACK")
        st.markdown("""
**UNSW-NB15**
- Source: University of New South Wales
- Samples: 257,673
- Features: 43
- Attack types: DoS, Fuzzers, Exploits,
  Generic, Reconnaissance, Backdoor,
  Analysis, Shellcode, Worms

---
**TECH STACK**
- `scikit-learn` Random Forest
- `SHAP` Explainability (XAI)
- `Streamlit` Web Interface
- `Plotly` Visualizations
- `Python 3.11`

---
**REFERENCES**
- Moustafa & Slay (2015). *UNSW-NB15.*
- Lundberg & Lee (2017). *SHAP.*
- Breiman (2001). *Random Forests.*
        """)

    # ── Build info ────────────────────────────────────────────
    if metrics:
        st.markdown("---")
        st.markdown("##### ▶ BUILD INFO")
        bi1, bi2, bi3 = st.columns(3)
        bi1.metric("BUILD",      metrics.get("build", "—"))
        bi2.metric("TRAINED AT", metrics.get("trained_at", "—")[:10])
        bi3.metric("MODEL",      metrics.get("model", "—"))


# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
acc_footer = (
    f"{metrics['metrics']['accuracy']*100:.2f}% ACCURACY"
    if metrics else "run train_model.py"
)
st.markdown(f"""
<hr style='border-color:#001500; margin-top:40px;'>
<div style='text-align:center; padding:8px 0;'>
    <p style='color:#002200; font-size:9px; letter-spacing:2px; margin:0;'>
        X-NIDS ENTERPRISE v9.5 &nbsp;·&nbsp;
        RANDOM FOREST + SHAP &nbsp;·&nbsp;
        UNSW-NB15 &nbsp;·&nbsp;
        {acc_footer}
    </p>
</div>
""", unsafe_allow_html=True)
