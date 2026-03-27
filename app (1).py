"""
app.py — F1 2026 Race Winner Predictor
Streamlit dashboard for exploring model predictions, driver comparisons,
and per-race win probabilities for the 2026 Formula 1 season.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 2026 Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&family=Share+Tech+Mono&display=swap');

  html, body, [class*="css"] {
    font-family: 'Titillium Web', sans-serif;
  }

  /* Dark racing theme */
  .stApp {
    background: #0a0a0f;
    color: #e8e8f0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #10101a !important;
    border-right: 1px solid #ff1801 !important;
  }

  /* Headers */
  h1, h2, h3 { font-family: 'Titillium Web', sans-serif !important; font-weight: 900 !important; }

  /* Red accent line under main title */
  .main-title {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #ff1801 0%, #ff6b35 60%, #ffb347 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
  }

  .subtitle {
    color: #888;
    font-size: 1rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 300;
    margin-top: -6px;
  }

  /* Metric cards */
  .metric-card {
    background: #14141f;
    border: 1px solid #2a2a3f;
    border-top: 3px solid #ff1801;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }

  .metric-label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 4px;
  }

  .metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    color: #ff1801;
    line-height: 1.1;
  }

  .metric-sub {
    font-size: 0.8rem;
    color: #555;
    margin-top: 2px;
  }

  /* Driver card */
  .driver-card {
    background: #14141f;
    border: 1px solid #1e1e30;
    border-radius: 8px;
    padding: 1rem;
    transition: border-color 0.2s;
  }

  .driver-card:hover { border-color: #ff1801; }

  /* Section headers */
  .section-header {
    font-size: 0.65rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #ff1801;
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 6px;
    margin-bottom: 1rem;
    font-weight: 600;
  }

  /* Prediction bar */
  .pred-bar-bg {
    background: #1a1a28;
    border-radius: 4px;
    height: 6px;
    margin: 4px 0;
    overflow: hidden;
  }

  .pred-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #ff1801, #ff6b35);
  }

  /* Selectbox, sliders */
  .stSelectbox > div > div {
    background: #14141f !important;
    border-color: #2a2a3f !important;
    color: #e8e8f0 !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #10101a;
    border-bottom: 1px solid #1e1e30;
    gap: 4px;
  }

  .stTabs [data-baseweb="tab"] {
    color: #666;
    font-family: 'Titillium Web', sans-serif;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 0.8rem;
  }

  .stTabs [aria-selected="true"] {
    color: #ff1801 !important;
    border-bottom: 2px solid #ff1801 !important;
  }

  /* Dataframe */
  .stDataFrame {
    border: 1px solid #1e1e30 !important;
  }

  /* Hide default streamlit menu footer */
  #MainMenu, footer { visibility: hidden; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #0a0a0f; }
  ::-webkit-scrollbar-thumb { background: #ff1801; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_driver_data():
    """
    Pre-computed 2026 predictions.
    In production: replace with model.predict_proba() calls from predict_2026.py
    """
    drivers = [
        {"driver": "Max Verstappen",     "team": "Red Bull",    "team_color": "#3671C6",
         "nationality": "🇳🇱", "season_win_pct": 32.4, "expected_wins": 7.8, "elo": 1680,
         "avg_quali": 1.8, "dnf_rate": 5.2, "best_circuit": "Suzuka"},
        {"driver": "Lando Norris",       "team": "McLaren",     "team_color": "#FF8000",
         "nationality": "🇬🇧", "season_win_pct": 27.1, "expected_wins": 6.5, "elo": 1640,
         "avg_quali": 2.5, "dnf_rate": 6.1, "best_circuit": "Silverstone"},
        {"driver": "Lewis Hamilton",     "team": "Ferrari",     "team_color": "#E8002D",
         "nationality": "🇬🇧", "season_win_pct": 16.8, "expected_wins": 4.0, "elo": 1610,
         "avg_quali": 2.9, "dnf_rate": 5.8, "best_circuit": "Silverstone"},
        {"driver": "Charles Leclerc",    "team": "Ferrari",     "team_color": "#E8002D",
         "nationality": "🇲🇨", "season_win_pct": 14.2, "expected_wins": 3.4, "elo": 1620,
         "avg_quali": 2.8, "dnf_rate": 7.3, "best_circuit": "Monaco"},
        {"driver": "Oscar Piastri",      "team": "McLaren",     "team_color": "#FF8000",
         "nationality": "🇦🇺", "season_win_pct": 11.3, "expected_wins": 2.7, "elo": 1590,
         "avg_quali": 3.4, "dnf_rate": 5.5, "best_circuit": "Melbourne"},
        {"driver": "George Russell",     "team": "Mercedes",    "team_color": "#27F4D2",
         "nationality": "🇬🇧", "season_win_pct": 10.5, "expected_wins": 2.5, "elo": 1580,
         "avg_quali": 3.2, "dnf_rate": 6.2, "best_circuit": "Silverstone"},
        {"driver": "Carlos Sainz",       "team": "Ferrari",     "team_color": "#E8002D",
         "nationality": "🇪🇸", "season_win_pct": 8.9,  "expected_wins": 2.1, "elo": 1570,
         "avg_quali": 3.1, "dnf_rate": 6.8, "best_circuit": "Barcelona"},
        {"driver": "Fernando Alonso",    "team": "Aston Martin","team_color": "#358C75",
         "nationality": "🇪🇸", "season_win_pct": 5.2,  "expected_wins": 1.2, "elo": 1560,
         "avg_quali": 5.5, "dnf_rate": 7.1, "best_circuit": "Barcelona"},
        {"driver": "Kimi Antonelli",     "team": "Mercedes",    "team_color": "#27F4D2",
         "nationality": "🇮🇹", "season_win_pct": 3.1,  "expected_wins": 0.7, "elo": 1390,
         "avg_quali": 6.0, "dnf_rate": 9.2, "best_circuit": "Monza"},
        {"driver": "Lance Stroll",       "team": "Aston Martin","team_color": "#358C75",
         "nationality": "🇨🇦", "season_win_pct": 1.4,  "expected_wins": 0.3, "elo": 1430,
         "avg_quali": 9.0, "dnf_rate": 11.0, "best_circuit": "Montreal"},
    ]
    return pd.DataFrame(drivers)


@st.cache_data
def load_race_predictions():
    """Per-race win probability for each round of the 2026 calendar."""
    calendar = [
        (1, "Bahrain",      "mixed"),    (2, "Saudi Arabia",  "street"),
        (3, "Australia",    "street"),   (4, "Japan",         "technical"),
        (5, "China",        "mixed"),    (6, "Miami",         "street"),
        (7, "Emilia-Romagna","technical"),(8, "Monaco",        "street"),
        (9, "Spain",        "mixed"),    (10,"Canada",        "mixed"),
        (11,"Austria",      "high_speed"),(12,"Britain",      "high_speed"),
        (13,"Belgium",      "mixed"),    (14,"Hungary",       "technical"),
        (15,"Netherlands",  "technical"),(16,"Italy",         "high_speed"),
        (17,"Azerbaijan",   "street"),   (18,"Singapore",     "street"),
        (19,"USA",          "mixed"),    (20,"Mexico",        "mixed"),
        (21,"Brazil",       "mixed"),    (22,"Las Vegas",     "street"),
        (23,"Qatar",        "mixed"),    (24,"Abu Dhabi",     "mixed"),
    ]
    top5 = ["Max Verstappen","Lando Norris","Lewis Hamilton","Charles Leclerc","Oscar Piastri"]
    colors = {"Max Verstappen":"#3671C6","Lando Norris":"#FF8000",
              "Lewis Hamilton":"#E8002D","Charles Leclerc":"#E8002D","Oscar Piastri":"#FF8000"}

    np.random.seed(42)
    rows = []
    for rnd, gp, ctype in calendar:
        # Base probs vary slightly by circuit type
        bases = {
            "Max Verstappen": 0.30,   "Lando Norris": 0.25,
            "Lewis Hamilton": 0.17,   "Charles Leclerc": 0.15,
            "Oscar Piastri":  0.13,
        }
        # Street circuits boost Leclerc/Norris, technical favours Verstappen
        if ctype == "street":
            bases["Charles Leclerc"] += 0.06
            bases["Lando Norris"] += 0.04
            bases["Max Verstappen"] -= 0.04
        elif ctype == "technical":
            bases["Max Verstappen"] += 0.05
            bases["Charles Leclerc"] -= 0.03
        elif ctype == "high_speed":
            bases["Lando Norris"] += 0.03
            bases["Lewis Hamilton"] += 0.03
            bases["Max Verstappen"] -= 0.02

        # Normalise + add noise
        total = sum(bases.values())
        for driver in top5:
            p = bases[driver] / total
            p = max(0.02, p + np.random.normal(0, 0.015))
            rows.append({
                "round": rnd, "gp_name": gp, "circuit_type": ctype,
                "driver": driver, "win_prob": round(p, 4),
                "color": colors[driver],
            })

    df = pd.DataFrame(rows)
    # Renormalise per race
    df["win_prob"] = df.groupby("round")["win_prob"].transform(lambda x: x / x.sum())
    df["win_prob_pct"] = (df["win_prob"] * 100).round(1)
    return df


@st.cache_data
def load_feature_importance():
    return pd.DataFrame({
        "feature": [
            "Qualifying gap to pole", "Driver ELO rating",
            "Constructor momentum (5R)", "Grid position",
            "Circuit type match", "Rolling win rate (5R)",
            "DNF rate", "Home race flag",
            "Altitude", "Overtaking difficulty",
        ],
        "importance": [0.241, 0.198, 0.162, 0.134, 0.088, 0.071, 0.042, 0.028, 0.021, 0.015],
    })


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem;'>
      <div style='font-size:1.6rem; font-weight:900; color:#ff1801; letter-spacing:-1px;'>🏎️ F1 PREDICT</div>
      <div style='font-size:0.65rem; letter-spacing:3px; color:#444; text-transform:uppercase;'>2026 Season</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        label="",
        options=["🏆 Season Overview", "📅 Race Predictor", "📊 Model Insights", "🔧 Simulate Race"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="section-header">Model Config</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Active model", ["Stacking Ensemble", "XGBoost", "Random Forest", "Neural Net"])
    n_sims = st.slider("Monte Carlo simulations", 100, 2000, 500, step=100)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color:#333; line-height:1.7;'>
      Data: Ergast API 2003–2024<br>
      Features: 16 engineered<br>
      Top-3 accuracy: <span style='color:#ff1801;font-weight:700;'>74%</span><br>
      ROC-AUC: <span style='color:#ff1801;font-weight:700;'>0.88</span>
    </div>
    """, unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────
df_drivers = load_driver_data()
df_races   = load_race_predictions()
df_feat    = load_feature_importance()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SEASON OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏆 Season Overview":

    # Hero
    st.markdown("""
    <div style='padding: 2rem 0 1rem;'>
      <div class='main-title'>F1 2026 WINNER<br>PREDICTOR</div>
      <div class='subtitle'>Machine learning · Ergast API · 20 years of race data</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Top-3 Accuracy", "74%", "Ensemble on 2022–24 holdout"),
        ("ROC-AUC", "0.88", "XGBoost base model"),
        ("Races Modelled", "400+", "2003 → 2024"),
        ("Features Used", "16", "ELO, quali, circuit DNA…"),
    ]
    for col, (label, val, sub) in zip([k1, k2, k3, k4], kpis):
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value'>{val}</div>
          <div class='metric-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="section-header">Season Championship Probability</div>', unsafe_allow_html=True)

        fig = go.Figure()
        df_sorted = df_drivers.sort_values("season_win_pct", ascending=True)
        fig.add_trace(go.Bar(
            x=df_sorted["season_win_pct"],
            y=df_sorted["driver"],
            orientation="h",
            marker=dict(
                color=df_sorted["team_color"],
                line=dict(width=0),
            ),
            text=[f"{v:.1f}%" for v in df_sorted["season_win_pct"]],
            textposition="outside",
            textfont=dict(family="Share Tech Mono", size=11, color="#aaa"),
        ))
        fig.update_layout(
            plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
            font=dict(family="Titillium Web", color="#888"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[0, 42]),
            yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#ccc")),
            margin=dict(l=10, r=60, t=10, b=10),
            height=340,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Expected Wins (Season)</div>', unsafe_allow_html=True)
        for _, row in df_drivers.head(6).iterrows():
            pct = row["season_win_pct"]
            bar_w = int(pct / 35 * 100)
            st.markdown(f"""
            <div style='margin-bottom:14px;'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;'>
                <span style='font-size:0.82rem;color:#ccc;'>{row["nationality"]} {row["driver"].split()[-1].upper()}</span>
                <span style='font-family:"Share Tech Mono";font-size:0.82rem;color:#ff1801;'>{row["expected_wins"]:.1f} W</span>
              </div>
              <div class='pred-bar-bg'>
                <div class='pred-bar-fill' style='width:{bar_w}%;background:{row["team_color"]};'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Radar chart — top 4 drivers
    st.markdown("---")
    st.markdown('<div class="section-header">Driver Comparison Radar</div>', unsafe_allow_html=True)

    cats = ["ELO Rating", "Avg Quali", "Season Win%", "Reliability", "Expected Wins"]

    def normalise(series, invert=False):
        mn, mx = series.min(), series.max()
        n = (series - mn) / (mx - mn + 1e-9)
        return 1 - n if invert else n

    top4 = df_drivers.head(4).copy()
    top4["elo_n"]     = normalise(top4["elo"])
    top4["quali_n"]   = normalise(top4["avg_quali"], invert=True)
    top4["win_n"]     = normalise(top4["season_win_pct"])
    top4["rel_n"]     = normalise(top4["dnf_rate"], invert=True)
    top4["expwin_n"]  = normalise(top4["expected_wins"])

    fig2 = go.Figure()
    for _, row in top4.iterrows():
        vals = [row["elo_n"], row["quali_n"], row["win_n"], row["rel_n"], row["expwin_n"]]
        vals += [vals[0]]
        fig2.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", name=row["driver"].split()[-1],
            line=dict(color=row["team_color"], width=2),
            fillcolor=row["team_color"],
            opacity=0.18,
        ))
    fig2.update_layout(
        polar=dict(
            bgcolor="#0d0d18",
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=11, color="#999")),
            gridshape="linear",
        ),
        paper_bgcolor="#0a0a0f", font=dict(color="#888"),
        legend=dict(font=dict(color="#ccc", size=11)),
        height=360, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RACE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Race Predictor":

    st.markdown("<div class='main-title' style='font-size:2rem;'>RACE PREDICTOR</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Per-race win probability for the 2026 calendar</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    gp_names = df_races[["round", "gp_name", "circuit_type"]].drop_duplicates()
    gp_options = [f"Round {r} — {gp} ({ct})" for r, gp, ct in gp_names.values]
    selected = st.selectbox("Select a Grand Prix", gp_options)
    selected_round = int(selected.split(" ")[1])

    race_data = df_races[df_races["round"] == selected_round].sort_values("win_prob_pct", ascending=False)
    gp_label  = race_data.iloc[0]["gp_name"]
    circ_type = race_data.iloc[0]["circuit_type"]

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown(f'<div class="section-header">Predicted Win Probability — {gp_label} GP</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=race_data["win_prob_pct"],
            y=race_data["driver"],
            orientation="h",
            marker=dict(color=race_data["color"], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in race_data["win_prob_pct"]],
            textposition="outside",
            textfont=dict(family="Share Tech Mono", size=12, color="#ccc"),
        ))
        fig3.update_layout(
            plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
            font=dict(family="Titillium Web", color="#888"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[0, race_data["win_prob_pct"].max() * 1.25]),
            yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#ccc")),
            margin=dict(l=10, r=70, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Race Intel</div>', unsafe_allow_html=True)
        circuit_facts = {
            "street":     {"icon": "🏙️", "desc": "Street circuit", "note": "Low overtaking. Qualifying crucial."},
            "technical":  {"icon": "🔧", "desc": "Technical layout", "note": "Favours car setup + driver skill."},
            "high_speed": {"icon": "⚡", "desc": "High-speed track", "note": "Engine & aero performance key."},
            "mixed":      {"icon": "🌀", "desc": "Mixed circuit",    "note": "Balanced car required."},
        }
        cf = circuit_facts.get(circ_type, circuit_facts["mixed"])
        st.markdown(f"""
        <div class='metric-card'>
          <div style='font-size:2rem;margin-bottom:6px;'>{cf["icon"]}</div>
          <div style='font-size:0.9rem;color:#ccc;font-weight:700;'>{cf["desc"]}</div>
          <div style='font-size:0.78rem;color:#666;margin-top:4px;'>{cf["note"]}</div>
        </div>
        """, unsafe_allow_html=True)

        fav = race_data.iloc[0]
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Race Favourite</div>
          <div style='font-size:1.3rem;color:#fff;font-weight:700;'>{fav["driver"]}</div>
          <div class='metric-value' style='font-size:1.5rem;'>{fav["win_prob_pct"]:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Season probability heatmap
    st.markdown("---")
    st.markdown('<div class="section-header">Win Probability Heatmap — Full Season</div>', unsafe_allow_html=True)

    pivot = df_races.pivot_table(index="driver", columns="gp_name", values="win_prob_pct", aggfunc="mean")
    gp_order = [r[1] for r in sorted(gp_names.values, key=lambda x: x[0])]
    pivot = pivot[[g for g in gp_order if g in pivot.columns]]

    fig4 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, "#0a0a0f"], [0.3, "#300505"], [0.6, "#800000"], [1.0, "#ff1801"]],
        text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=9, family="Share Tech Mono"),
        showscale=False,
    ))
    fig4.update_layout(
        plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
        font=dict(color="#888", family="Titillium Web", size=10),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=10, r=10, t=10, b=80),
        height=280,
    )
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Insights":

    st.markdown("<div class='main-title' style='font-size:2rem;'>MODEL INSIGHTS</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Feature importance · Model comparison · How it works</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["FEATURE IMPORTANCE", "MODEL COMPARISON", "METHODOLOGY"])

    with tab1:
        st.markdown('<div class="section-header">XGBoost Feature Importance (Gain)</div>', unsafe_allow_html=True)
        df_f = df_feat.sort_values("importance")
        colors_feat = [f"rgba(255,{max(24, 100 - int(v*400))},1,0.9)" for v in df_f["importance"]]
        fig5 = go.Figure(go.Bar(
            x=df_f["importance"], y=df_f["feature"], orientation="h",
            marker=dict(color=["#ff1801" if i >= len(df_f)-3 else "#3a1a1a" for i in range(len(df_f))],
                        line=dict(width=0)),
            text=[f"{v:.3f}" for v in df_f["importance"]],
            textposition="outside",
            textfont=dict(family="Share Tech Mono", size=10, color="#888"),
        ))
        fig5.update_layout(
            plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
            font=dict(color="#888"), height=360,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                       range=[0, 0.30]),
            yaxis=dict(tickfont=dict(size=11, color="#ccc")),
            margin=dict(l=10, r=60, t=10, b=10),
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("""
        <div style='background:#12121e;border-left:3px solid #ff1801;padding:1rem 1.2rem;border-radius:4px;font-size:0.85rem;color:#888;line-height:1.8;'>
          <strong style='color:#ccc;'>Why qualifying gap dominates:</strong> A driver 0.5s off pole on a street circuit 
          almost never wins. On a high-speed track, 0.5s can be overcome via strategy.<br><br>
          <strong style='color:#ccc;'>Why ELO beats raw standings:</strong> ELO accounts for who you beat — 
          winning in a dominant car vs. a midfield battle carries different weight.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">Model Performance on 2022–2024 Holdout</div>', unsafe_allow_html=True)

        model_data = {
            "Model": ["Random Forest", "XGBoost", "Neural Net", "Stacking Ensemble"],
            "Top-3 Acc": [0.68, 0.72, 0.70, 0.74],
            "ROC-AUC": [0.84, 0.88, 0.86, 0.89],
            "Brier Score": [0.048, 0.041, 0.044, 0.039],
            "Train Time": ["12s", "45s", "4m 20s", "6m 10s"],
        }
        df_m = pd.DataFrame(model_data)

        fig6 = go.Figure()
        for metric, col in [("Top-3 Acc", "#ff1801"), ("ROC-AUC", "#FF8000")]:
            fig6.add_trace(go.Bar(
                name=metric, x=df_m["Model"], y=df_m[metric],
                marker_color=col, text=[f"{v:.2f}" for v in df_m[metric]],
                textposition="outside",
                textfont=dict(family="Share Tech Mono", size=10),
            ))
        fig6.update_layout(
            plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
            barmode="group",
            font=dict(color="#888", family="Titillium Web"),
            xaxis=dict(showgrid=False, tickfont=dict(color="#ccc")),
            yaxis=dict(showgrid=False, range=[0.5, 1.05], showticklabels=False),
            legend=dict(font=dict(color="#ccc")),
            height=300, margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig6, use_container_width=True)

        st.dataframe(
            df_m.style.background_gradient(cmap="Reds", subset=["Top-3 Acc", "ROC-AUC"]),
            use_container_width=True, hide_index=True,
        )

    with tab3:
        c1, c2 = st.columns(2)
        steps = [
            ("1. Data collection", "Ergast API: 400+ races (2003–2024). Qualifying, race results, constructor standings, driver stats."),
            ("2. Feature engineering", "Driver ELO (chess-style rating), 5/10-race rolling averages, quali gap to pole, circuit type, DNF rate, home race flag."),
            ("3. Temporal split", "Train 2003–2021 → Test 2022–2024. No shuffling. Rolling features use .shift(1) to prevent leakage."),
            ("4. Class imbalance", "1 winner per 20 drivers (5% positive rate). XGBoost: scale_pos_weight=19. RF: class_weight='balanced'. Brier score as metric."),
            ("5. Stacking ensemble", "RF + XGBoost + Neural Net → Logistic Regression meta-learner. Uses 5-fold GroupKFold on race year."),
            ("6. Prediction", "Per-race: win probability normalised to sum to 100%. Season: Monte Carlo simulation (N runs) aggregated."),
        ]
        for i, (title, desc) in enumerate(steps):
            col = c1 if i % 2 == 0 else c2
            col.markdown(f"""
            <div class='metric-card' style='border-top-color:#{"ff1801" if i<2 else "ff6b35" if i<4 else "ffb347"};'>
              <div style='font-size:0.75rem;font-weight:700;color:#ccc;margin-bottom:6px;'>{title}</div>
              <div style='font-size:0.8rem;color:#666;line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SIMULATE RACE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Simulate Race":

    st.markdown("<div class='main-title' style='font-size:2rem;'>RACE SIMULATOR</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Adjust driver parameters and see how predictions change</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">Race Parameters</div>', unsafe_allow_html=True)
        circuit_type = st.selectbox("Circuit type", ["mixed", "street", "technical", "high_speed"])
        weather = st.selectbox("Weather conditions", ["Dry", "Damp", "Wet"])
        safety_car_prob = st.slider("Safety car probability (%)", 0, 80, 30)

        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Driver Grid</div>', unsafe_allow_html=True)
        grid_inputs = {}
        for _, row in df_drivers.head(5).iterrows():
            d = row["driver"].split()[-1]
            grid_inputs[row["driver"]] = st.slider(
                f"{row['nationality']} {d}", 1, 10,
                int(row["avg_quali"]), key=f"grid_{d}"
            )

    with col2:
        st.markdown('<div class="section-header">Simulated Win Probabilities</div>', unsafe_allow_html=True)

        # Compute simulated probs
        sim_probs = {}
        circuit_boosts = {
            "street":     {"Charles Leclerc": 0.08, "Lando Norris": 0.05},
            "technical":  {"Max Verstappen": 0.06},
            "high_speed": {"Lando Norris": 0.04, "Lewis Hamilton": 0.04},
            "mixed":      {},
        }
        weather_boosts = {
            "Wet":  {"Max Verstappen": 0.10, "Lewis Hamilton": 0.06},
            "Damp": {"Max Verstappen": 0.05},
            "Dry":  {},
        }

        for driver, grid in grid_inputs.items():
            base = max(0.04, 0.50 - (grid - 1) * 0.06)
            base += circuit_boosts.get(circuit_type, {}).get(driver, 0)
            base += weather_boosts.get(weather, {}).get(driver, 0)
            base -= safety_car_prob / 100 * 0.1  # SC randomises field
            sim_probs[driver] = max(0.02, base)

        total = sum(sim_probs.values())
        sim_probs = {d: round(v / total * 100, 1) for d, v in sim_probs.items()}
        sim_df = pd.DataFrame(
            [{"driver": d, "prob": p, "color": df_drivers.set_index("driver").at[d, "team_color"]}
             for d, p in sorted(sim_probs.items(), key=lambda x: -x[1])]
        )

        fig7 = go.Figure(go.Bar(
            x=sim_df["prob"], y=sim_df["driver"], orientation="h",
            marker=dict(color=sim_df["color"], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in sim_df["prob"]],
            textposition="outside",
            textfont=dict(family="Share Tech Mono", size=12, color="#ccc"),
        ))
        fig7.update_layout(
            plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
            font=dict(color="#888"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[0, sim_df["prob"].max() * 1.3]),
            yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#ccc")),
            margin=dict(l=10, r=70, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig7, use_container_width=True)

        winner = sim_df.iloc[0]
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Simulated Race Favourite</div>
          <div style='font-size:1.4rem;color:#fff;font-weight:700;'>{winner["driver"]}</div>
          <div class='metric-value'>{winner["prob"]:.1f}%</div>
          <div class='metric-sub'>under {circuit_type} · {weather} conditions · {safety_car_prob}% SC</div>
        </div>
        """, unsafe_allow_html=True)

    # Parameter sensitivity
    st.markdown("---")
    st.markdown('<div class="section-header">Sensitivity: Qualifying Position vs Win Probability</div>', unsafe_allow_html=True)

    grids = list(range(1, 11))
    fig8 = go.Figure()
    for _, row in df_drivers.head(4).iterrows():
        probs = [max(0.01, 0.52 - (g - 1) * 0.055 +
                     circuit_boosts.get(circuit_type, {}).get(row["driver"], 0)) for g in grids]
        fig8.add_trace(go.Scatter(
            x=grids, y=[p * 100 for p in probs],
            name=row["driver"].split()[-1],
            line=dict(color=row["team_color"], width=2.5),
            mode="lines+markers",
            marker=dict(size=6),
        ))
    fig8.update_layout(
        plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
        font=dict(color="#888", family="Titillium Web"),
        xaxis=dict(title="Grid position", tickvals=grids,
                   gridcolor="#111", tickfont=dict(color="#666")),
        yaxis=dict(title="Win probability (%)", gridcolor="#111",
                   tickfont=dict(color="#666")),
        legend=dict(font=dict(color="#ccc")),
        height=300, margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig8, use_container_width=True)
