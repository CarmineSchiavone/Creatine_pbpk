import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==========================================
# 1. INTRODUCTION & SCIENTIFIC CONTEXT
# ==========================================

def display_intro():
    st.markdown("""
    # 🧬 Creatine Systems Pharmacology Analyst
    
    ### Hello! 
    I am **Carmine**, and I am a scientist specializing in **Systems Pharmacology**.
    
    **What is Systems Pharmacology?** It is an interdisciplinary field that uses mathematical modeling to understand how substances interact with the body as a holistic system. Rather than looking at "creatine levels" in isolation, this PBPK model simulates the dynamic flux between absorption, plasma distribution, saturable tissue uptake, and renal excretion [1].
    
    ---
    ⚠️ **SCIENTIFIC DISCLAIMER:** *This model uses population-derived kinetic constants (Michaelis-Menten). Individual metabolic rates and renal clearance thresholds vary. This is for educational simulation only.*
    """)

# ==========================================
# 2. THE PBPK PHYSICS ENGINE
# ==========================================

def get_parameters(weight_kg, diet, variability):
    """
    Calibrates constants based on literature.
    variability: Factor scaling transporter Vmax (0.7 to 1.3).
    """
    # Distribution Volumes
    v_plasma = weight_kg * 0.05  # Plasma volume approx 5% of BW
    muscle_mass_dm = weight_kg * 0.40 * 0.25 # kg Dry Mass (dm) 
    
    # Baselines (mmol/kg dm)
    # Source: Hultman et al. (1996) DOI: 10.1152/jappl.1996.81.1.232
    m_baseline = 125.0 if diet == "Omnivore" else 100.0
    m_ceiling = 160.0 
    
    # Synthesis & Degradation
    # Endogenous synthesis ~2g/day (15 mmol/day approx)
    # Degradation ~1.7%/day
    # Source: Brosnan et al. (2011) DOI: 10.1007/s00726-011-0853-y
    k_deg = 0.017 
    k_synth_base = 15.0 # mmol/day
    
    # Transporter Kinetics (SLC6A8)
    vmax_m = 5.0 * variability # mmol/kg/day
    km_m = 0.05 # mmol/L (Plasma Cr concentration)
    
    # Brain Parameters
    # Source: Dolan et al. (2019) DOI: 10.1007/s00421-019-04146-x
    k_brain_in = 0.004 * variability
    
    return {
        'v_plasma': v_plasma, 'm_dm': muscle_mass_dm,
        'm_base': m_baseline, 'm_ceil': m_ceiling,
        'k_deg': k_deg, 'k_synth': k_synth_base,
        'vmax_m': vmax_m, 'km_m': km_m,
        'k_brain_in': k_brain_in,
        'k_ka': 10.0, # Rapid GI absorption
        'cl_renal': 0.15 # L/day base clearance
    }

def pbpk_model(y, t, daily_dose_mmol, p):
    """
    State Vector y:
    y[0]: GI Amount (mmol)
    y[1]: Plasma Conc (mmol/L)
    y[2]: Muscle Content (mmol/kg dm)
    y[3]: Brain Index (Relative)
    y[4]: Urine Accumulated (mmol)
    """
    GI, Cp, M, B, U = y
    
    # 1. Endogenous Synthesis (Feedback inhibited by GI intake)
    # If GI intake is high, synthesis shuts down to 0
    synthesis = max(0, p['k_synth'] * (1 - (GI / (GI + 10))))
    
    # 2. GI Absorption
    abs_rate = p['k_ka'] * GI
    
    # 3. Muscle Uptake (Saturable SLC6A8 + Ceiling downregulation)
    sat_factor = max(0, (p['m_ceil'] - M) / (p['m_ceil'] - p['m_base']))
    v_uptake_m = (p['vmax_m'] * Cp / (p['km_m'] + Cp)) * sat_factor
    
    # 4. Brain Uptake (Very slow)
    v_uptake_b = p['k_brain_in'] * Cp
    
    # 5. Elimination (Renal + Degradation to Creatinine)
    # Creatinine conversion happens in muscle/brain (~1.7% total pool)
    degradation = p['k_deg'] * M * p['m_dm']
    # Renal clearance of excess plasma Cr
    renal_excretion = p['cl_renal'] * Cp * 24 # Spikes during load
    
    # ODEs
    dGI = -abs_rate
    dCp = (abs_rate + synthesis - (v_uptake_m * p['m_dm']) - renal_excretion) / p['v_plasma']
    dM = v_uptake_m - (p['k_deg'] * M) + (p['k_deg'] * p['m_base']) # maintain baseline
    dB = v_uptake_b - (0.017 * (B - 1.0))
    dU = renal_excretion + degradation
    
    return [dGI, dCp, dM, dB, dU]

# ==========================================
# 3. SIMULATION MANAGER
# ==========================================

def run_pbpk_sim(protocol, params, total_days):
    t = np.linspace(0, total_days, total_days * 24) # Hourly precision
    y0 = [0.0, 0.05, params['m_base'], 1.0, 0.0]
    
    # Handle Daily Dosing (Events)
    res_list = []
    current_y = y0
    
    dose_mmol = protocol['dose_g'] / 131.13 * 1000 # Convert g to mmol
    
    for day in range(total_days):
        t_span = np.linspace(day, day+1, 24)
        # Apply dose at t=0 of each day
        current_y[0] += dose_mmol
        sol = odeint(pbpk_model, current_y, t_span, args=(dose_mmol, params))
        res_list.append(sol)
        current_y = sol[-1]
        
    full_res = np.vstack(res_list)
    return full_res

# ==========================================
# 4. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Creatine PBPK Analyst", layout="wide")
display_intro()

with st.sidebar:
    st.header("1. Personal Profile")
    weight = st.number_input("Weight (kg)", 40, 150, 75)
    diet = st.selectbox("Dietary Baseline", ["Omnivore", "Vegetarian/Vegan"])
    efficiency = st.slider("Transporter Efficiency", 0.7, 1.3, 1.0)
    
    st.header("2. Supplement Protocol")
    dose_g = st.slider("Daily Dose (g)", 0, 25, 5)
    months = st.slider("Simulation Length (Months)", 1, 6, 3)

# Execute Model
P = get_parameters(weight, diet, efficiency)
simulation = run_pbpk_sim({'dose_g': dose_g}, P, months * 30)

# --- VISUALIZATION (4 COMPARTMENTS) ---
st.subheader(f"Systemic Forecast: {dose_g}g Daily for {months} Months")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Plasma Concentration (mmol/L)", "Muscle Saturation (%)", 
                    "Brain Increase (%)", "Urine Accumulated (mmol)"),
    vertical_spacing=0.15
)

days_axis = np.linspace(0, months*30, len(simulation))

# 1. Plasma Cr (Spikes)
fig.add_trace(go.Scatter(x=days_axis, y=simulation[:, 1], name="Plasma", line=dict(color='#FFA15A')), row=1, col=1)

# 2. Muscle Saturation (%)
m_sat = ((simulation[:, 2] - P['m_base']) / (P['m_ceil'] - P['m_base'])) * 100
fig.add_trace(go.Scatter(x=days_axis, y=m_sat, name="Muscle", line=dict(color='#00CC96', width=3)), row=1, col=2)

# 3. Brain Increase (%)
b_inc = (simulation[:, 3] - 1.0) * 10.0 # Normalized to typical max 10% rise
fig.add_trace(go.Scatter(x=days_axis, y=b_inc, name="Brain", line=dict(color='#636EFA')), row=2, col=1)

# 4. Urine Excretion (Total load)
fig.add_trace(go.Scatter(x=days_axis, y=simulation[:, 4], name="Urine", line=dict(color='#AB63FA')), row=2, col=2)

fig.update_layout(height=800, template="plotly_white", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# --- SCIENTIFIC INSIGHTS ---
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Muscle Plateau:** Once the green curve hits 100%, the SLC6A8 transporters downregulate. Taking more than 5g/day after this point mostly increases urinary excretion.")
with col2:
    st.info("**Brain Persistence:** Note the slow rise in the brain. Unlike muscle, the brain is highly resistant to external creatine flux, requiring weeks to see metabolic changes [4].")
with col3:
    st.info("**Renal Load:** The urine graph tracks the clearance of excess creatine and the metabolic creatinine. For healthy individuals, this is a normal physiological process [2].")

with st.expander("📚 Literature References & DOI"):
    st.markdown("""
    1. **Vicini & van der Graaf (2013).** Systems Pharmacology in Drug Discovery. DOI: [10.1038/psp.2012.22](https://doi.org/10.1038/psp.2012.22)
    2. **Brosnan et al. (2011).** The role of creatine in health and disease. DOI: [10.1007/s00726-011-0853-y](https://doi.org/10.1007/s00726-011-0853-y)
    3. **Hultman et al. (1996).** Muscle creatine loading in men. DOI: [10.1152/jappl.1996.81.1.232](https://doi.org/10.1152/jappl.1996.81.1.232)
    4. **Dolan et al. (2019).** Beyond muscle: effects of creatine on brain. DOI: [10.1007/s00421-019-04146-x](https://doi.org/10.1007/s00421-019-04146-x)
    """)