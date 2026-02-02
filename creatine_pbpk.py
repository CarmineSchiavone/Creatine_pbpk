import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time

# ==========================================
# 1. INTRODUCTION & SYSTEMS PHARMACOLOGY
# ==========================================

def display_intro():
    st.markdown("""
    # 🧬 Creatine PBPK: Muscle & Brain Systems Analyst
    
    ### Hello! 
    I am **Carmine**, and I am a scientist specializing in **Systems Pharmacology**.
    
    **What is Systems Pharmacology?** It is the quantitative study of how drugs or supplements interact with the body as an integrated network. Instead of looking at a single value, we use **Physiologically Based Pharmacokinetic (PBPK)** modeling to simulate the "flux"—the movement of molecules between compartments like plasma, muscle, and brain [1].
    
    ---
    ⚠️ **SCIENTIFIC DISCLAIMER:** *This model uses population-derived Michaelis-Menten constants. Actual saturation depends on individual renal thresholds and genetic SLC6A8 expression.*
    """)

# ==========================================
# 2. THE PBPK PHYSICS ENGINE
# ==========================================

def get_parameters(weight_kg, diet, variability, activity_level):
    """
    Method: Parameter scaling based on literature.
    Variables: 
    - activity_level: Scales Vmax (Transporter capacity) and k_deg (Turnover).
    - diet: Adjusts baseline pool (mmol/kg dm).
    """
    # Distribution Volumes
    v_plasma = weight_kg * 0.05  
    m_dm = weight_kg * 0.40 * 0.25 # kg Dry Mass (dm)
    
    # Baselines (mmol/kg dm) - Source: Hultman (1996) DOI: 10.1152/jappl.1996.81.1.232
    m_base = 125.0 if diet == "Omnivore" else 100.0
    m_ceil = 160.0 
    
    # Exercise Impacts - Source: Steenge et al. (2000) DOI: 10.1152/jappl.2000.89.3.1165
    # Physical activity increases blood flow and CrT translocation.
    activity_multiplier = {"Sedentary": 1.0, "Moderate (3x/wk)": 1.15, "Athlete (Daily)": 1.3}
    act_factor = activity_multiplier[activity_level]
    
    # Kinetic Rates
    # k_deg: Non-enzymatic conversion to creatinine (1.7%/day base)
    k_deg = 0.017 * (1.1 if activity_level == "Athlete (Daily)" else 1.0)
    k_synth_base = 15.0 # mmol/day
    
    # Transporter (SLC6A8) Kinetics
    vmax_m = 5.0 * variability * act_factor # mmol/kg/day
    km_m = 0.05 # mmol/L
    
    # Brain Parameters - Source: Dolan et al. (2019) DOI: 10.1007/s00421-019-04146-x
    k_brain_in = 0.004 * variability
    
    return {
        'v_plasma': v_plasma, 'm_dm': m_dm, 'm_base': m_base, 'm_ceil': m_ceil,
        'k_deg': k_deg, 'k_synth': k_synth_base, 'vmax_m': vmax_m, 'km_m': km_m,
        'k_brain_in': k_brain_in, 'k_ka': 10.0, 'cl_renal': 0.15
    }

def pbpk_model(y, t, daily_dose_mmol, p):
    GI, Cp, M, B = y
    
    # Feedback synthesis inhibition
    synthesis = max(0, p['k_synth'] * (1 - (GI / (GI + 5))))
    
    # Saturable Muscle Uptake (SLC6A8)
    # Down-regulates as M approaches p['m_ceil']
    sat_factor = max(0, (p['m_ceil'] - M) / (p['m_ceil'] - p['m_base']))
    v_uptake_m = (p['vmax_m'] * Cp / (p['km_m'] + Cp)) * sat_factor
    
    # Renal Clearance of Plasma Excess
    renal_ex = p['cl_renal'] * Cp * 24
    
    # ODEs
    dGI = -p['k_ka'] * GI
    dCp = (p['k_ka'] * GI + synthesis - (v_uptake_m * p['m_dm']) - renal_ex) / p['v_plasma']
    dM = v_uptake_m - (p['k_deg'] * M) + (p['k_deg'] * p['m_base'])
    dB = (p['k_brain_in'] * Cp) - (0.017 * (B - 1.0))
    
    return [dGI, dCp, dM, dB]

def run_pbpk_sim(dose_g, params, total_days):
    t_steps = total_days * 24
    y0 = [0.0, 0.05, params['m_base'], 1.0]
    res_list = []
    curr_y = y0
    dose_mmol = dose_g / 131.13 * 1000
    
    for day in range(total_days):
        t_span = np.linspace(day, day+1, 24)
        curr_y[0] += dose_mmol
        sol = odeint(pbpk_model, curr_y, t_span, args=(dose_mmol, params))
        res_list.append(sol[:-1])
        curr_y = sol[-1]
        
    return np.vstack(res_list)

# ==========================================
# 3. INTERFACE
# ==========================================

st.set_page_config(page_title="Creatine PBPK final", layout="wide")
display_intro()

with st.sidebar:
    st.header("1. Biometrics & Lifestyle")
    weight = st.number_input("Weight (kg)", 40, 150, 75)
    diet = st.selectbox("Dietary Baseline", ["Omnivore", "Vegetarian/Vegan"])
    activity = st.selectbox("Physical Activity", ["Sedentary", "Moderate (3x/wk)", "Athlete (Daily)"])
    var = st.slider("Genetic CrT Efficiency", 0.7, 1.3, 1.0)
    
    st.header("2. Supplement Protocol")
    dose = st.slider("Daily Dose (g)", 0, 25, 5)
    months = st.slider("Duration (Months)", 1, 6, 3)

# Simulation
P = get_parameters(weight, diet, var, activity)
sim = run_pbpk_sim(dose, P, months * 30)
days_axis = np.linspace(0, months*30, len(sim))

# --- ANALYSIS ---
m_sat = ((sim[:, 2] - P['m_base']) / (P['m_ceil'] - P['m_base'])) * 100
b_inc = (sim[:, 3] - 1.0) * 8.0 

# Threshold Detection
m_benefit_day = next((d for d, s in zip(days_axis, m_sat) if s >= 90), None)
b_benefit_day = next((d for d, s in zip(days_axis, b_inc) if s >= 5), None)

st.subheader("📊 Systemic Saturation Forecast")
col_res, col_plots = st.columns([1, 3])

with col_res:
    st.markdown("### 🕒 Estimated Benefit Onset")
    if m_benefit_day:
        st.success(f"**Muscle Benefits:** Day {int(m_benefit_day)}")
        st.caption("Enhanced ATP resynthesis & power output.")
    else: st.warning("Muscle saturation not reached.")
        
    if b_benefit_day:
        st.info(f"**Brain Benefits:** Day {int(b_benefit_day)}")
        st.caption("Improved cognitive buffer & fatigue resistance.")
    else: st.error("Brain threshold (5%) not reached.")
    
    st.markdown("---")
    st.write(f"**Vmax (Muscle):** {P['vmax_m']:.2f} mmol/kg/d")
    st.write(f"**Vd (Muscle):** {P['m_dm']:.1f} kg (dm)")

with col_plots:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Plasma Conc (mmol/L)", "Muscle Saturation (%)", "Brain Increase (%)"))
    
    # Plasma
    fig.add_trace(go.Scatter(x=days_axis, y=sim[:, 1], name="Plasma", line=dict(color='#FFA15A')), row=1, col=1)
    
    # Muscle + Efficacy Band
    fig.add_trace(go.Scatter(x=days_axis, y=m_sat, name="Muscle", line=dict(color='#00CC96', width=3)), row=2, col=1)
    fig.add_hrect(y0=90, y1=100, fillcolor="green", opacity=0.2, annotation_text="Efficacy Zone", row=2, col=1)
    
    # Brain + Efficacy Band
    fig.add_trace(go.Scatter(x=days_axis, y=b_inc, name="Brain", line=dict(color='#636EFA')), row=3, col=1)
    fig.add_hrect(y0=5, y1=10, fillcolor="blue", opacity=0.2, annotation_text="Cognitive Target", row=3, col=1)
    
    fig.update_xaxes(title_text="Time (Days)", row=3, col=1)
    fig.update_layout(height=800, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📚 References & Modeling Rationale"):
    st.markdown("""
    - **Physical Activity:** Exercise increases skeletal muscle CrT (SLC6A8) expression and activity via the AMPK pathway [1, 5].
    - **Muscle Threshold (90%):** Clinical performance benefits (strength/sprints) are significantly observed when total muscle creatine reaches >145 mmol/kg dm [1].
    - **Brain Threshold (5%):** Brain creatine is tightly regulated. A 5-10% increase is the standard clinical finding for cognitive improvements in stressed/fatigued states [4].
    - **Reference 1:** Hultman et al. (1996) DOI: [10.1152/jappl.1996.81.1.232](https://doi.org/10.1152/jappl.1996.81.1.232)
    - **Reference 4:** Dolan et al. (2019) DOI: [10.1007/s00421-019-04146-x](https://doi.org/10.1007/s00421-019-04146-x)
    """)