import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.common.common import CONFIG
from datetime import datetime, timedelta

API_URL = CONFIG["api"]["app_api_url"]

st.set_page_config(
    page_title="London Weather Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --bg:      #0d1117;
        --surface: #161b22;
        --border:  #30363d;
        --accent:  #58a6ff;
        --accent2: #f78166;
        --text:    #e6edf3;
        --muted:   #8b949e;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .status-up   { color: #3fb950; font-weight: 700; }
    .status-down { color: var(--accent2); font-weight: 700; }
    .section-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 2rem 0;
    }

    [data-testid="stDateInput"] input,
    [data-testid="stSelectbox"] select,
    [data-testid="stTextInput"] input {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }
    div[data-baseweb="select"] > div {
        background-color: var(--surface) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }
    .stButton > button {
        background: var(--accent);
        color: #0d1117;
        border: none;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover { background: #79c0ff; color: #0d1117; }

    [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-family: 'Space Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


def check_api_status():
    try:
        r = requests.get(f"{API_URL}/", allow_redirects=False, timeout=2)
        return r.status_code in [200, 302, 307]
    except Exception:
        return False


def fetch_version():
    try:
        r = requests.get(f"{API_URL}/version", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_combined(start: str, end: str):
    try:
        r = requests.get(
            f"{API_URL}/predictions/combined",
            params={"start_date": start, "end_date": end},
            timeout=5
        )
        if r.status_code == 200:
            return pd.DataFrame(r.json())
        elif r.status_code == 404:
            return pd.DataFrame()
        else:
            st.error(f"Erreur API : {r.status_code} — {r.json().get('detail', '')}")
            return None
    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")
        return None


def fetch_monitoring(model_name=None, date=None):
    try:
        params = {}
        if model_name:
            params["model_name"] = model_name
        if date:
            params["date"] = date
        r = requests.get(f"{API_URL}/monitoring", params=params, timeout=5)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 404:
            return None
        else:
            st.error(f"Erreur API monitoring : {r.status_code} — {r.json().get('detail', '')}")
            return None
    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")
        return None


PLOT_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(family="DM Sans", color="#e6edf3"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=20, b=10)
)

COLORS = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#ffa657", "#79c0ff"]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌤 Weather Forecast")
    st.markdown("---")

    api_ok = check_api_status()
    status_html = (
        '<span class="status-up">● ONLINE</span>' if api_ok
        else '<span class="status-down">● OFFLINE</span>'
    )
    st.markdown(f"**API Status** &nbsp; {status_html}", unsafe_allow_html=True)

    version = fetch_version()
    if version:
        st.markdown(f"**Modèle** `{version.get('model_name', '—')}`")
        st.markdown(f"**Version** `{version.get('model_version', '—')}`")

    st.markdown("---")
    st.markdown("### 📅 Période — Prédictions")

    default_end   = datetime.now()
    default_start = default_end - timedelta(days=3)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Début", value=default_start)
        start_time = st.time_input("Heure", value=datetime.strptime("00:00", "%H:%M").time(), key="start_time")
    with col2:
        end_date = st.date_input("Fin", value=default_end)
        end_time = st.time_input("Heure", value=datetime.strptime("23:00", "%H:%M").time(), key="end_time")

    start_dt = f"{start_date} {start_time.strftime('%H:%M')}"
    end_dt   = f"{end_date} {end_time.strftime('%H:%M')}"

    st.markdown("---")
    st.markdown("### 🔬 Monitoring")

    monitor_date = st.date_input(
        "Date d'inférence",
        value=None,
        key="monitor_date",
        help="Filtre les prédictions générées à cette date"
    )

    st.markdown("---")
    st.button("⟳ Actualiser")


# ── Guard ─────────────────────────────────────────────────────────────────────
st.markdown("# Prédictions météo")

if not api_ok:
    st.error(" L'API est inaccessible.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Prédictions
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"###  Prédictions  `{start_dt}` → `{end_dt}`")

df = fetch_combined(start_dt, end_dt)

if df is None:
    st.stop()

if df.empty:
    st.warning("Aucune donnée pour cette période.")
else:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    has_observed = "observed_temp" in df.columns and df["observed_temp"].notna().any()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Points de prédiction", len(df))
    with col2:
        st.metric("Modèles", df["model_name"].nunique())
    with col3:
        st.metric("Min prédit", f"{df['predicted_value'].min():.1f} °C")
    with col4:
        st.metric("Max prédit", f"{df['predicted_value'].max():.1f} °C")

    if has_observed:
        df_cmp = df.dropna(subset=["observed_temp"])
        mae  = (df_cmp["predicted_value"] - df_cmp["observed_temp"]).abs().mean()
        rmse = ((df_cmp["predicted_value"] - df_cmp["observed_temp"]) ** 2).mean() ** 0.5
        col5, col6 = st.columns(2)
        with col5:
            st.metric("MAE", f"{mae:.2f} °C")
        with col6:
            st.metric("RMSE", f"{rmse:.2f} °C")

    models          = df["model_name"].unique().tolist()
    selected_models = st.multiselect("Modèles à afficher", models, default=models)

    if not selected_models:
        st.warning("Sélectionne au moins un modèle.")
    else:
        df_filtered = df[df["model_name"].isin(selected_models)]
        color_map   = {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}

        fig = go.Figure()

        if has_observed:
            df_obs = df_filtered.dropna(subset=["observed_temp"]).drop_duplicates("timestamp")
            fig.add_trace(go.Scatter(
                x=df_obs["timestamp"], y=df_obs["observed_temp"],
                mode="lines", name="Observé",
                line=dict(color="#e6edf3", width=2, dash="dot"),
                opacity=0.8
            ))

        for model in selected_models:
            df_model = df_filtered[df_filtered["model_name"] == model]
            fig.add_trace(go.Scatter(
                x=df_model["timestamp"], y=df_model["predicted_value"],
                mode="lines+markers", name=f"Prédit — {model}",
                line=dict(color=color_map[model], width=2),
                marker=dict(size=4)
            ))

        fig.update_layout(**PLOT_LAYOUT, height=500)
        fig.update_xaxes(gridcolor="#21262d", linecolor="#30363d", title="Timestamp")
        fig.update_yaxes(gridcolor="#21262d", linecolor="#30363d", title="Température (°C)")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Données brutes"):
            cols_display = ["timestamp", "model_name", "horizon", "predicted_value"]
            if has_observed:
                cols_display.append("observed_temp")
            st.dataframe(
                df_filtered[cols_display].rename(columns={
                    "timestamp":       "Timestamp",
                    "model_name":      "Modèle",
                    "horizon":         "Horizon (h)",
                    "predicted_value": "Prédit (°C)",
                    "observed_temp":   "Observé (°C)"
                }),
                use_container_width=True,
                hide_index=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Monitoring
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### 🔬 Monitoring — Erreurs de prédiction")

available_models = df["model_name"].unique().tolist() if df is not None and not df.empty else []
monitor_model = st.selectbox(
    "Modèle à analyser",
    options=[None] + available_models,
    format_func=lambda x: "Tous les modèles" if x is None else x,
    key="monitor_model"
)

monitor_result = fetch_monitoring(
    model_name=monitor_model,
    date=str(monitor_date) if monitor_date else None
)

if monitor_result is None:
    st.info("Aucune donnée de monitoring disponible — les observations réelles ne sont peut-être pas encore disponibles pour cette période.")
else:
    # KPIs
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("MAE", f"{monitor_result['mae']:.2f} °C")
    with col_b:
        st.metric("RMSE", f"{monitor_result['rmse']:.2f} °C")
    with col_c:
        st.metric("Points comparés", len(monitor_result["data"]))

    df_mon = pd.DataFrame(monitor_result["data"])
    df_mon["timestamp"] = pd.to_datetime(df_mon["timestamp"])
    df_mon["error"]     = df_mon["predicted_value"] - df_mon["observed"]

    # ── Subplots ──────────────────────────────────────────────────────────────
    fig_mon = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06
    )

    # Panel 1 — Observé vs Prédit
    fig_mon.add_trace(go.Scatter(
        x=df_mon["timestamp"], y=df_mon["observed"],
        mode="lines", name="Observé",
        line=dict(color="#e6edf3", width=2, dash="dot"),
        opacity=0.8
    ), row=1, col=1)

    fig_mon.add_trace(go.Scatter(
        x=df_mon["timestamp"], y=df_mon["predicted_value"],
        mode="lines+markers", name="Prédit",
        line=dict(color="#58a6ff", width=2),
        marker=dict(size=4)
    ), row=1, col=1)

    # Panel 2 — Erreur point par point
    fig_mon.add_trace(go.Bar(
        x=df_mon["timestamp"],
        y=df_mon["error"],
        name="Erreur (prédit − observé)",
        marker_color=["#f78166" if e > 0 else "#3fb950" for e in df_mon["error"]],
        opacity=0.8
    ), row=2, col=1)

    fig_mon.add_hline(y=0, line=dict(color="#30363d", width=1), row=2, col=1)

    fig_mon.update_layout(
        **PLOT_LAYOUT,
        height=520,
    )
    fig_mon.update_xaxes(gridcolor="#21262d", linecolor="#30363d")
    fig_mon.update_yaxes(gridcolor="#21262d", linecolor="#30363d")
    fig_mon.update_yaxes(title_text="Température (°C)", row=1, col=1)
    fig_mon.update_yaxes(title_text="Erreur (°C)",      row=2, col=1)

    st.plotly_chart(fig_mon, use_container_width=True)

    with st.expander("Données brutes — monitoring"):
        st.dataframe(
            df_mon.rename(columns={
                "timestamp":       "Timestamp",
                "predicted_value": "Prédit (°C)",
                "observed":        "Observé (°C)",
                "error":           "Erreur (°C)"
            }),
            use_container_width=True,
            hide_index=True
        )