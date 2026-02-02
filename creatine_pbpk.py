import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==========================================
# 1. INTRODUCTION & SYSTEMS PHARMACOLOGY
# ==========================================

def display_intro():
    st.markdown("""
    # 🧬 Creatine PBPK: Steady-State Systems Analyst
    
    ### Hello! 
    I am **Carmine**, and I am a scientist specializing in **Systems Pharmacology**.
    
    **What is Systems Pharmacology?** It is the quantitative study of how substances interact with the body as an integrated network. Instead of looking at a single value, we use **Physiologically Based Pharmacokinetic (PBPK)** modeling to simulate the "flux"—the movement of molecules between compartments like plasma, muscle, and brain [1].
    
    **Model Improvement:** This version utilizes a **180-day 'Burn-in' period** to calculate your unique physiological steady state before supplementation begins. This ensures that the baseline represents your body's natural equilibrium between endogenous synthesis and degradation [2].
    
    ---
    ⚠️ **SCIENTIFIC DISCLAIMER:** *This model uses population-derived Michaelis-Menten constants. Individual results depend on renal function and genetic SLC6A8 expression.*
    """)

# ==========================================
# 2. THE PBPK PHYSICS ENGINE
# ==========================================

def get_parameters(weight_kg, diet, variability, activity_level):
    """
    Method: Parameter scaling based on literature.
    """
    v_plasma = weight_kg * 0.05  
    m_dm = weight_kg * 0.40 * 0.25 # kg Dry Mass (dm)
    
    # Baselines (mmol/kg dm) 
    # Source: Hultman (1996) DOI: 10.1152/jappl.1996.81.1.232
    m_base_guess = 125.0 if diet == "Omnivore" else 100.0
    m_ceil = 160.0 
    
    # Exercise Impacts - Source: Steenge et al. (2000) DOI: 10.1152/jappl.2000.89.3.1165
    activity_multiplier = {"Sedentary": 1.0, "Moderate (3x/wk)": 1.15, "Athlete (Daily)": 1.3}
    act_factor = activity_multiplier[activity_level]
    
    # Non-enzymatic conversion to creatinine (1.7%/day base)
    k_deg = 0.017 * (1.1 if activity_level == "Athlete (Daily)" else 1.0)
    k_synth_base = 15.0 # mmol/day endogenous
    
    # Transporter (SLC6A8) Kinetics
    vmax_m = 5.0 * variability * act_factor 
    km_m = 0.05 # mmol/L
    
    k_brain_in = 0.004 * variability
    
    return {
        'v_plasma': v_plasma, 'm_dm': m_dm, 'm_base_guess': m_base_guess, 'm_ceil': m_ceil,
        'k_deg': k_deg, 'k_synth': k_synth_base, 'vmax_m': vmax_m, 'km_m': km_m,
        'k_brain_in': k_brain_in, 'k_ka': 10.0, 'cl_renal': 0.15
    }

def pbpk_model(y, t, dose_mmol, p):
    GI, Cp, M, B = y
    
    # Feedback synthesis inhibition (reduced if external intake is high)
    synthesis = max(0, p['k_synth'] * (1 - (GI / (GI + 10))))
    
    # Saturable Muscle Uptake (SLC6A8)
    # Down-regulates as M approaches p['m_ceil']
    sat_factor = max(0, (p['m_ceil'] - M) / (p['m_ceil'] - p['m_base_guess']))
    v_uptake_m = (p['vmax_m'] * Cp / (p['km_m'] + Cp)) * sat_factor
    
    # Renal Clearance
    renal_ex = p['cl_renal'] * Cp * 24
    
    dGI = -p['k_ka'] * GI
    dCp = (p['k_ka'] * GI + synthesis - (v_uptake_m * p['m_dm']) - renal_ex) / p['v_plasma']
    # M dynamics: uptake vs non-enzymatic degradation
    dM = v_uptake_m - (p['k_deg'] * M)
    dB = (p['k_brain_in'] * Cp) - (0.017 * (B - 1.0))
    
    return [dGI, dCp, dM, dB]

def run_pbpk_sim(dose_g, params, total_days):
    # STEP 1: BURN-IN PHASE (180 days at 0g dose to find steady state)
    burn_in_days = 180
    t_burn = np.linspace(0, burn_in_days, burn_in_days * 2)
    y0_initial = [0.0, 0.05, params['m_base_guess'], 1.0]
    
    sol_burn = odeint(pbpk_model, y0_initial, t_burn, args=(0, params))
    y_steady = sol_burn[-1] # Final state of 0g simulation
    
    # STEP 2: ACTUAL SIMULATION (Starting from Steady State)
    res_list = []
    curr_y = y_steady.copy()
    dose_mmol = dose_g / 131.13 * 1000
    
    for day in range(total_days):
        t_span = np.linspace(day, day+1, 24)
        curr_y[0] += dose_mmol
        sol = odeint(pbpk_model, curr_y, t_span, args=(dose_mmol, params))
        res_list.append(sol[:-1])
        curr_y = sol[-1]
        
    return np.vstack(res_list), y_steady

# ==========================================
# 3. INTERFACE
# ==========================================

st.set_page_config(page_title="Creatine Steady-State Analyst", layout="wide")
display_intro()

with st.sidebar:
    st.header("1. Biometrics & Lifestyle")
    weight = st.number_input("Weight (kg)", 40, 150, 75)
    diet = st.selectbox("Dietary Baseline", ["Omnivore", "Vegetarian/Vegan"])
    activity = st.selectbox("Physical Activity", ["Sedentary", "Moderate (3x/wk)", "Athlete (Daily)"])
    var = st.slider("Genetic CrT Efficiency", 0.7, 1.3, 1.0)
    
    st.header("2. Supplement Protocol")
    dose = st.slider("Daily Dose (g)", 0, 25, 5)
    months = st.slider("Duration (Months)", 1, 6, 6)

# Execute Simulation
P = get_parameters(weight, diet, var, activity)
sim, steady_state = run_pbpk_sim(dose, P, months * 30)
days_axis = np.linspace(0, months*30, len(sim))

# --- CALCULATION OF INCREASE OVER STEADY STATE ---
# M_Sat is % of the gap between steady state and physiological ceiling
m_ss = steady_state[2]
m_sat = ((sim[:, 2] - m_ss) / (P['m_ceil'] - m_ss)) * 100
# Brain increase relative to individual steady state
b_ss = steady_state[3]
b_inc = ((sim[:, 3] - b_ss) / b_ss) * 100 

# Threshold Detection
m_benefit_day = next((d for d, s in zip(days_axis, m_sat) if s >= 90), None)
b_benefit_day = next((d for d, s in zip(days_axis, b_inc) if s >= 5), None)

st.subheader("📊 Systemic Saturation Forecast (Relative to Steady State)")
col_res, col_plots = st.columns([1, 3])

with col_res:
    st.markdown("### 🕒 Estimated Benefit Onset")
    if dose == 0:
        st.info("Currently at Endogenous Steady State. No supplemental increase expected.")
    else:
        if m_benefit_day:
            st.success(f"**Muscle Benefits:** Day {int(m_benefit_day)}")
            st.caption("Maximized ATP resynthesis achieved.")
        else: st.warning("Muscle ceiling not reached at this dose.")
            
        if b_benefit_day:
            st.info(f"**Brain Benefits:** Day {int(b_benefit_day)}")
            st.caption("Significant cognitive buffer reached.")
        else: st.error("Brain threshold (5%) not reached.")
    
    st.markdown("---")
    st.write(f"**Calculated Steady State (Muscle):** {m_ss:.2f} mmol/kg")
    st.write(f"**Metabolic Turnover:** {P['k_deg']*100:.2f}% / day")

with col_plots:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Plasma Conc (mmol/L)", "Muscle Increase (% of Potential)", "Brain Increase (%)"))
    
    # Plasma (Shows spikes above steady state)
    fig.add_trace(go.Scatter(x=days_axis, y=sim[:, 1], name="Plasma", line=dict(color='#FFA15A')), row=1, col=1)
    
    # Muscle + Efficacy Band
    fig.add_trace(go.Scatter(x=days_axis, y=m_sat, name="Muscle", line=dict(color='#00CC96', width=3)), row=2, col=1)
    fig.add_hrect(y0=90, y1=100, fillcolor="green", opacity=0.2, annotation_text="Max Muscle Efficacy", row=2, col=1)
    
    # Brain + Efficacy Band
    fig.add_trace(go.Scatter(x=days_axis, y=b_inc, name="Brain", line=dict(color='#636EFA')), row=3, col=1)
    fig.add_hrect(y0=5, y1=15, fillcolor="blue", opacity=0.2, annotation_text="Cognitive Target", row=3, col=1)
    
    fig.update_xaxes(title_text="Time (Days)", row=3, col=1)
    fig.update_layout(height=800, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📚 References & Modeling Rationale"):
    st.markdown("""
    - **Burn-in Phase:** Essential because endogenous synthesis ($~2g/day$) creates a non-zero baseline. The model solves for the equilibrium where $Uptake = Degradation$ before applying the first dose [2].
    - **Activity scaling:** Athletes have higher metabolic turnover ($k_{deg}$) and higher transporter translocation to the membrane, leading to faster loading but also faster washout if supplementation stops [1, 5].
    - **Hultman Threshold:** A 90% increase toward the ceiling ($~160 mmol/kg$) is clinically linked to significant ergogenic effects in anaerobic performance [1].
    - **Reference 1:** Hultman et al. (1996) DOI: [10.1152/jappl.1996.81.1.232](https://doi.org/10.1152/jappl.1996.81.1.232)
    - **Reference 4:** Dolan et al. (2019) DOI: [10.1007/s00421-019-04146-x](https://doi.org/10.1007/s00421-019-04146-x)
    """)