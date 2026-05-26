# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import MatchPredictor
from src.simulation import WorldCupSimulator

st.set_page_config(page_title="World Cup Prediction Engine", page_icon="⚽", layout="wide")

st.title("⚽ AI-Powered Football World Cup Prediction Engine")
st.markdown("Analiză predictivă folosind un model **XGBoost** antrenat pe 15.000+ meciuri internaționale și simulări **Monte Carlo**.")

@st.cache_resource
def load_components():
    return MatchPredictor(), WorldCupSimulator()

predictor, simulator = load_components()

tab1, tab2, tab3 = st.tabs(["🔮 Predicție Meci Direct", "📊 Statistici Monte Carlo", "🏆 Bracket Live (1 Turneu)"])

# Încărcăm automat grupele din teams_2026.csv pentru a fi 100% dinamice
path_echipe = "data/raw/teams_2026.csv"
world_cup_groups = {}
if os.path.exists(path_echipe):
    teams_df = pd.read_csv(path_echipe, names=['team', 'elo', 'group'])
    for group_name, group_data in teams_df.groupby('group'):
        world_cup_groups[f"Group {group_name}"] = group_data['team'].tolist()

# --- TAB 1: PREDICȚIE MECI DIRECT ---
with tab1:
    st.header("Predicție meci individual")
    lista_echipe = sorted(list(predictor.elo_database.keys()))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        echipa_gazda = st.selectbox("Echipa Gazdă", lista_echipe, index=lista_echipe.index("France") if "France" in lista_echipe else 0)
    with col2:
        is_neutral = st.checkbox("Meci pe teren neutru", value=True)
    with col3:
        echipa_oaspeti = st.selectbox("Echipa Oaspeți", lista_echipe, index=lista_echipe.index("Romania") if "Romania" in lista_echipe else 0)
        
    if st.button("Generează Predicție", type="primary"):
        if echipa_gazda == echipa_oaspeti:
            st.warning("Te rugăm să selectezi două echipe diferite.")
        else:
            rezultat = predictor.predict_match(echipa_gazda, echipa_oaspeti, is_neutral=is_neutral)
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric(label=f"Victorie {echipa_gazda}", value=f"{rezultat[echipa_gazda]*100:.1f}%")
            m_col2.metric(label="Egal", value=f"{rezultat['Draw']*100:.1f}%")
            m_col3.metric(label=f"Victorie {echipa_oaspeti}", value=f"{rezultat[echipa_oaspeti]*100:.1f}%")
            
            df_plot = pd.DataFrame({
                'Rezultat': list(rezultat.keys()),
                'Probabilitate (%)': [v * 100 for v in rezultat.values()]
            })
            fig = px.bar(df_plot, x='Rezultat', y='Probabilitate (%)', text='Probabilitate (%)', color='Rezultat',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: STATISTICI MONTE CARLO ---
with tab2:
    st.header("Simulare Globală Monte Carlo")
    iteratii = st.slider("Număr de turnee simulate pentru statistici", min_value=100, max_value=2000, value=500, step=100)
    
    if st.button("Pornește Simularea Statistică", type="primary"):
        with st.spinner("Se simulează turneele în fundal..."):
            df_res = simulator.run_monte_carlo(world_cup_groups, iterations=iteratii)
            st.success(f"Gata! S-au simulat {iteratii} de turnee complete.")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Tabel Rezultate")
                st.dataframe(df_res.style.format({"Probability (%)": "{:.1f}%"}), use_container_width=True)
            with c2:
                # Graficul Plotly Wow cerut
                fig = px.bar(
                    df_res.head(10),
                    x='Probability (%)',
                    y='Team',
                    orientation='h',
                    text='Probability (%)',
                    title="🏆 World Cup 2026 Winner Probabilities (Monte Carlo)",
                    color='Probability (%)',
                    color_continuous_scale='Viridis'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Insight text dinamic
                best_team = df_res.iloc[0]
                st.success(
                    f"🏆 **Most likely winner:** {best_team['Team']} "
                    f"with {best_team['Probability (%)']:.1f}% chance"
                )
                
                st.write("---")
                st.subheader("Top 5 Favorites")
                num_favorites = min(5, len(df_res))
                for i in range(num_favorites):
                    row = df_res.iloc[i]
                    st.write(f"{i+1}. **{row['Team']}** – {row['Probability (%)']:.1f}%")

# --- TAB 3: BRACKET LIVE ---
with tab3:
    st.header("Visual Bracket Simulator")
    st.markdown("Generează **un singur scenariu complet** pentru fazele eliminatorii.")
    
    if st.button("Simulează un Tablou Eliminatoriu", type="primary"):
        qualified = []
        for g_name, g_teams in sorted(world_cup_groups.items()):
            top_2 = simulator.simulate_group(g_teams)
            qualified.extend(top_2)
        optimi = qualified[:16]
        
        sferturi = []
        for i in range(0, len(optimi), 2):
            winner = simulator.simulate_match_outcome(optimi[i], optimi[i+1], can_draw=False)
            sferturi.append(winner)
            
        semifinale = []
        for i in range(0, len(sferturi), 2):
            winner = simulator.simulate_match_outcome(sferturi[i], sferturi[i+1], can_draw=False)
            semifinale.append(winner)
            
        campioana = simulator.simulate_match_outcome(semifinale[0], semifinale[1], can_draw=False)
        
        c_optimi, c_sferturi, c_semi, c_finala, c_campioana = st.columns(5)
        
        with c_optimi:
            st.subheader("16) Optimi")
            for i in range(0, len(optimi), 2):
                st.info(f"🏟️ Meci {i//2 + 1}\n\n**{optimi[i]}**\nvs\n**{optimi[i+1]}**")
                st.write("---")
                
        with c_sferturi:
            st.subheader("8) Sferturi")
            for i in range(0, len(sferturi), 2):
                st.warning(f"🥊 Sfert {i//2 + 1}\n\n**{sferturi[i]}**\nvs\n**{sferturi[i+1]}**")
                st.write("---")
                
        with c_semi:
            st.subheader("4) Semifinale")
            st.error(f"🔥 Semi 1\n\n**{semifinale[0]}**\nvs\n**{semifinale[1]}**")
            st.write("---")
            st.error(f"🔥 Semi 2\n\n**{semifinale[2]}**\nvs\n**{semifinale[3]}**")
            
        with c_finala:
            st.subheader("🥇 Marea Finală")
            st.markdown(f"### ⚔️ Finala\n\n**{semifinale[0]}**\nvs\n**{semifinale[1]}**")
            
        with c_campioana:
            st.subheader("👑 Campioană")
            st.balloons()
            st.success(f"### 🎉 {campioana}\n\na câștigat Cupa Mondială!")