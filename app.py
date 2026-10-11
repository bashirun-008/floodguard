import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="FloodGuard India", layout="wide", page_icon="🌊")

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1F6FB2, #4FC3F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #888; font-size: 0.95rem; margin-top: 0; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('flood_model.pkl')

@st.cache_data
def load_geojson():
    url = "https://cdn.jsdelivr.net/gh/udit-001/india-maps-data@main/geojson/india.geojson"
    return requests.get(url, timeout=20).json()

model = load_model()
india_districts = load_geojson()

feature_cols = ['MonsoonIntensity','TopographyDrainage','RiverManagement','Deforestation',
            'Urbanization','ClimateChange','DamsQuality','Siltation','AgriculturalPractices',
            'Encroachments','IneffectiveDisasterPreparedness','DrainageSystems',
            'CoastalVulnerability','Landslides','Watersheds','DeterioratingInfrastructure',
            'PopulationScore','WetlandLoss','InadequatePlanning','PoliticalFactors']

@st.cache_data
def build_district_data():
    np.random.seed(42)
    district_names = [f['properties'].get('district', 'Unknown') for f in india_districts['features']]
    state_names = [f['properties'].get('st_nm', 'Unknown') for f in india_districts['features']]
    n = len(district_names)
    data = pd.DataFrame({
        'District': district_names, 'State': state_names,
        'MonsoonIntensity': np.random.randint(2, 10, n),
        'TopographyDrainage': np.random.randint(1, 9, n),
        'RiverManagement': np.random.randint(1, 9, n),
        'Deforestation': np.random.randint(1, 9, n),
        'Urbanization': np.random.randint(1, 9, n),
        'ClimateChange': np.random.randint(3, 10, n),
        'DamsQuality': np.random.randint(1, 9, n),
        'Siltation': np.random.randint(1, 9, n),
        'AgriculturalPractices': np.random.randint(1, 9, n),
        'Encroachments': np.random.randint(1, 9, n),
        'IneffectiveDisasterPreparedness': np.random.randint(1, 9, n),
        'DrainageSystems': np.random.randint(1, 9, n),
        'CoastalVulnerability': np.random.randint(0, 8, n),
        'Landslides': np.random.randint(0, 8, n),
        'Watersheds': np.random.randint(1, 9, n),
        'DeterioratingInfrastructure': np.random.randint(1, 9, n),
        'PopulationScore': np.random.randint(1, 9, n),
        'WetlandLoss': np.random.randint(1, 9, n),
        'InadequatePlanning': np.random.randint(1, 9, n),
        'PoliticalFactors': np.random.randint(1, 9, n),
    })
    data['FloodRisk'] = model.predict(data[feature_cols])
    data['id'] = data.index
    return data

district_data = build_district_data()
for i, feature in enumerate(india_districts['features']):
    feature['id'] = i

st.markdown('<p class="main-header">🌊 FloodGuard India</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ML-powered district-level flood risk prediction & early warning system</p>', unsafe_allow_html=True)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🗺️ Risk Map", "📈 Forecast", "🚨 Alerts", "🧠 Model Insights", "🛠️ Policy Simulator"]
)

# ---------- TAB 0: OVERVIEW ----------
with tab0:
    st.subheader("National Snapshot")
    high_risk = (district_data['FloodRisk'] >= 0.6).sum()
    moderate_risk = ((district_data['FloodRisk'] >= 0.4) & (district_data['FloodRisk'] < 0.6)).sum()
    low_risk = (district_data['FloodRisk'] < 0.4).sum()
    total = len(district_data)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts Monitored", f"{total}")
    c2.metric("🔴 High Risk", f"{high_risk}", delta=f"{high_risk/total:.0%} of total", delta_color="off")
    c3.metric("🟡 Moderate Risk", f"{moderate_risk}", delta=f"{moderate_risk/total:.0%} of total", delta_color="off")
    c4.metric("🟢 Low Risk", f"{low_risk}", delta=f"{low_risk/total:.0%} of total", delta_color="off")

    st.markdown("---")
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("##### Top 10 Highest-Risk Districts")
        top10 = district_data.sort_values('FloodRisk', ascending=False).head(10)[['District','State','FloodRisk']]
        top10['FloodRisk'] = (top10['FloodRisk']*100).round(1).astype(str) + '%'
        st.dataframe(top10, hide_index=True, use_container_width=True)
    with col_b:
        st.markdown("##### Risk Distribution")
        dist_fig = px.pie(
            values=[high_risk, moderate_risk, low_risk],
            names=['High Risk', 'Moderate Risk', 'Low Risk'],
            color=['High Risk', 'Moderate Risk', 'Low Risk'],
            color_discrete_map={'High Risk':'#E74C3C','Moderate Risk':'#F39C12','Low Risk':'#2ECC71'},
            hole=0.5
        )
        dist_fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10), showlegend=True)
        st.plotly_chart(dist_fig, use_container_width=True)

    st.info("💡 This system combines climate, terrain, human, and policy factors across 760 Indian districts to identify flood-prone areas before they happen — enabling proactive rather than reactive disaster response.")

# ---------- TAB 1: MAP ----------
with tab1:
    st.subheader("India District-Level Flood Risk Map")
    fig = px.choropleth(
        district_data, geojson=india_districts, locations='id', color='FloodRisk',
        hover_name='District', hover_data=['State', 'FloodRisk'],
        color_continuous_scale='YlOrRd',
        range_color=(district_data['FloodRisk'].min(), district_data['FloodRisk'].max()),
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":10,"l":0,"b":0}, height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Darker red indicates higher predicted flood risk. Tap/hover a district for details.")

# ---------- TAB 2: FORECAST ----------
with tab2:
    st.subheader("5-Day Flood Risk Forecast")
    search_mode = st.radio("Choose district by:", ["Highest Risk (Top 10)", "Search by name"], horizontal=True)

    if search_mode == "Highest Risk (Top 10)":
        top10 = district_data.sort_values('FloodRisk', ascending=False).head(10)
        if 'selected_district' not in st.session_state:
            st.session_state['selected_district'] = top10.iloc[0]['District']
        cols = st.columns(5)
        for i, (_, r) in enumerate(top10.iterrows()):
            col = cols[i % 5]
            if col.button(f"{r['District']}\n{r['FloodRisk']:.0%}", key=f"btn_{i}"):
                st.session_state['selected_district'] = r['District']
    else:
        all_names = sorted(district_data['District'].unique())
        chosen = st.selectbox("Type to search a district:", all_names)
        st.session_state['selected_district'] = chosen

    selected_district = st.session_state.get('selected_district', district_data.iloc[0]['District'])
    row = district_data[district_data['District']==selected_district].iloc[0]
    st.write(f"**Showing forecast for:** {selected_district}, {row['State']} — Current baseline risk: **{row['FloodRisk']:.1%}**")

    use_live_weather = st.checkbox("🌦️ Use live weather data for this district (overrides simulated values)")

    live_monsoon = row['MonsoonIntensity']
    live_climate = row['ClimateChange']
    weather_status = None

    if use_live_weather:
        OWM_API_KEY = os.environ.get("OWM_API_KEY", "")
        try:
            if not OWM_API_KEY:
                raise RuntimeError("OWM_API_KEY not set")
            url = f"https://api.openweathermap.org/data/2.5/weather?q={selected_district},IN&appid={OWM_API_KEY}&units=metric"
            resp = requests.get(url, timeout=5)
            wdata = resp.json()
            if resp.status_code == 200:
                rain_mm = wdata.get('rain', {}).get('1h', 0)
                humidity = wdata.get('main', {}).get('humidity', 50)
                temp = wdata.get('main', {}).get('temp', 25)
                desc = wdata.get('weather', [{}])[0].get('description', 'N/A')

                live_monsoon = min(10, rain_mm * 2 if rain_mm > 0 else humidity / 12)
                live_climate = min(10, humidity / 10)
                weather_status = f"✅ Live weather: {desc}, {temp}°C, {humidity}% humidity, {rain_mm}mm rain (last hr)"
            else:
                weather_status = f"⚠️ Could not fetch live weather for {selected_district} (city name may not match weather API). Using simulated baseline instead."
        except Exception as e:
            weather_status = f"⚠️ Weather fetch failed: {e}. Using simulated baseline instead."

    if weather_status:
        if "✅" in weather_status:
            st.success(weather_status)
        else:
            st.warning(weather_status)

    days = [1,2,3,4,5]
    rows = []
    for day in days:
        r = row[feature_cols].copy()
        r['MonsoonIntensity'] = min(10, live_monsoon + day*1.0)
        r['ClimateChange'] = min(10, live_climate + day*0.6)
        r['RiverManagement'] = min(10, row['RiverManagement'] + day*0.4)
        r['DrainageSystems'] = min(10, row['DrainageSystems'] + day*0.4)
        rows.append(r)
    fdf = pd.DataFrame(rows)
    fdf['Day'] = days
    fdf['PredictedRisk'] = model.predict(fdf[feature_cols])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fdf['Day'], y=fdf['PredictedRisk'], mode='lines+markers',
                               line=dict(color='#E74C3C', width=3), marker=dict(size=10),
                               fill='tozeroy', fillcolor='rgba(231,76,60,0.1)', name='Predicted Risk'))
    fig2.add_hline(y=0.6, line_dash="dash", line_color="darkred", annotation_text="High Risk Threshold (0.6)")
    fig2.update_layout(title=f"5-Day Forecast — {selected_district}, {row['State']}" + (" (Live Weather)" if use_live_weather and weather_status and "✅" in weather_status else ""),
                        xaxis_title="Day", yaxis_title="Predicted Flood Probability",
                        yaxis_range=[0,1], height=450, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.session_state['last_forecast'] = fdf
    st.session_state['last_district'] = selected_district
    st.session_state['last_state'] = row['State']
    st.session_state['last_pop'] = row['PopulationScore']

# ---------- TAB 3: ALERTS ----------
with tab3:
    st.subheader("Automated Alert & Response Panel")
    if 'last_forecast' in st.session_state:
        fdf = st.session_state['last_forecast']
        district_name = st.session_state['last_district']
        state_name = st.session_state['last_state']
        pop_score = st.session_state['last_pop']
        day5_risk = fdf.iloc[-1]['PredictedRisk']
        est_pop = int(pop_score * 45000)

        if day5_risk < 0.4:
            level, action = "LOW", "Monitor conditions. No immediate action required."
        elif day5_risk < 0.6:
            level, action = "MODERATE", "Alert local authorities. Prepare evacuation routes for low-lying areas."
        else:
            level, action = "HIGH", "Issue evacuation order for low-lying areas. Activate emergency shelters."

        st.markdown(f"### 🚨 {district_name}, {state_name}")
        if level == "HIGH":
            st.error(f"**Risk Level: {level}** — {day5_risk:.1%} predicted by Day 5")
        elif level == "MODERATE":
            st.warning(f"**Risk Level: {level}** — {day5_risk:.1%} predicted by Day 5")
        else:
            st.success(f"**Risk Level: {level}** — {day5_risk:.1%} predicted by Day 5")

        col1, col2, col3 = st.columns(3)
        col1.metric("Est. Population at Risk", f"{est_pop:,}")
        col2.metric("Day 5 Risk", f"{day5_risk:.1%}")
        col3.metric("Days to Threshold", "Already crossed" if fdf.iloc[0]['PredictedRisk'] >= 0.6 else str((fdf['PredictedRisk'] >= 0.6).idxmax()+1) if (fdf['PredictedRisk'] >= 0.6).any() else "Not reached")

        st.markdown(f"**📋 Recommended Action:** {action}")
        st.markdown(f"**🏠 Nearest Shelter:** {district_name} District Community Center (demo location)")

        with st.expander("📱 Preview: Public SMS Alert"):
            st.code(f"🚨 FLOOD ALERT: {district_name} district. Risk level {level} ({day5_risk:.0%}). {action} Stay tuned for updates.", language=None)
    else:
        st.info("👈 Go to the Forecast tab first and select a district to generate an alert.")

# ---------- TAB 4: MODEL INSIGHTS ----------
with tab4:
    st.subheader("Model Performance & Explainability")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "XGBoost Regressor")
    c2.metric("R² Score", "0.9599")
    c3.metric("RMSE", "0.00998")

    st.markdown("##### Feature Importance")
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    imp_fig = px.bar(feat_imp, orientation='h', labels={'value':'Importance','index':'Feature'})
    imp_fig.update_layout(height=550, showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(imp_fig, use_container_width=True)

    st.caption("Trained on 50,000 records (naiyakhalid/flood-prediction-dataset, Kaggle) using an 80/20 train-test split.")

# ---------- TAB 5: POLICY SIMULATOR ----------
with tab5:
    st.subheader("🛠️ What-If Policy Simulator")
    st.caption("Simulate infrastructure and policy interventions to see their impact on flood risk — helping prioritize where investment matters most.")

    sim_names = sorted(district_data['District'].unique())
    default_idx = sim_names.index(st.session_state.get('last_district', sim_names[0])) if st.session_state.get('last_district') in sim_names else 0
    sim_district = st.selectbox("Select district to simulate:", sim_names, index=default_idx, key="sim_district_select")

    sim_row = district_data[district_data['District']==sim_district].iloc[0]
    baseline_risk = sim_row['FloodRisk']

    st.markdown(f"**Baseline risk for {sim_district}, {sim_row['State']}: {baseline_risk:.1%}**")
    st.markdown("---")
    st.markdown("##### Adjust interventions (higher = better condition)")

    col1, col2 = st.columns(2)
    with col1:
        new_drainage = st.slider("🚰 Drainage System Quality", 0, 10, int(sim_row['DrainageSystems']), key="sim_drainage")
        new_river = st.slider("🌊 River Management", 0, 10, int(sim_row['RiverManagement']), key="sim_river")
        new_dams = st.slider("🏗️ Dam Quality", 0, 10, int(sim_row['DamsQuality']), key="sim_dams")
    with col2:
        new_deforest = st.slider("🌳 Afforestation Level (10 = well-forested)", 0, 10, 10 - int(sim_row['Deforestation']), key="sim_forest")
        new_prep = st.slider("🛡️ Disaster Preparedness (10 = fully prepared)", 0, 10, 10 - int(sim_row['IneffectiveDisasterPreparedness']), key="sim_prep")
        new_planning = st.slider("📋 Urban Planning Quality (10 = well-planned)", 0, 10, 10 - int(sim_row['InadequatePlanning']), key="sim_planning")

    sim_features = sim_row[feature_cols].copy()
    sim_features['DrainageSystems'] = new_drainage
    sim_features['RiverManagement'] = new_river
    sim_features['DamsQuality'] = new_dams
    sim_features['Deforestation'] = 10 - new_deforest
    sim_features['IneffectiveDisasterPreparedness'] = 10 - new_prep
    sim_features['InadequatePlanning'] = 10 - new_planning

    sim_input = pd.DataFrame([sim_features])[feature_cols]
    new_risk = model.predict(sim_input)[0]
    risk_change = new_risk - baseline_risk

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("Baseline Risk", f"{baseline_risk:.1%}")
    r2.metric("Simulated Risk", f"{new_risk:.1%}", delta=f"{risk_change:+.1%}", delta_color="inverse")
    r3.metric("Risk Reduction", f"{abs(risk_change):.1%}" if risk_change < 0 else "No improvement")

    comp_fig = go.Figure()
    comp_fig.add_trace(go.Bar(
        x=['Baseline', 'After Interventions'],
        y=[baseline_risk, new_risk],
        marker_color=['#E74C3C' if baseline_risk >= 0.6 else '#F39C12', '#2ECC71' if new_risk < 0.4 else ('#F39C12' if new_risk < 0.6 else '#E74C3C')],
        text=[f"{baseline_risk:.1%}", f"{new_risk:.1%}"],
        textposition='outside'
    ))
    comp_fig.update_layout(title="Risk Before vs. After Interventions", yaxis_range=[0,1], height=400, template="plotly_white")
    st.plotly_chart(comp_fig, use_container_width=True)

    if risk_change < -0.05:
        st.success(f"✅ These interventions could meaningfully reduce flood risk in {sim_district} by {abs(risk_change):.1%} — a strong candidate for infrastructure investment.")
    elif risk_change < 0:
        st.info(f"ℹ️ These interventions offer a modest risk reduction of {abs(risk_change):.1%}.")
    else:
        st.warning("⚠️ Try improving drainage, river management, or preparedness sliders to see risk reduction.")
