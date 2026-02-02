import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. INTRODUCTION
# ==========================================

def display_intro():
    st.markdown("""
    # 🧬 Creatine PBPK: Steady-State Systems Analyst
    
    ### Hello! 
    I am **Carmine**, your Systems Pharmacology scientist.
    
    **The "Threshold" Correction:** This updated model implements a **Renal Reabsorption Threshold**. In a natural state (0g intake), your body reabsorbs 99% of creatine, keeping levels stable. Excretion only triggers when plasma levels exceed the physiological limit (~0.05 mmol/L). 
    
    **Result:** If you set the dose to 0g, you will now see a **perfectly flat steady state**, representing your body's natural homeostasis where *Endogenous Synthesis* exactly matches *Degradation*.
    
    ---
    ⚠️ **DISCLAIMER:** *Simulation for educational purposes only.*
    """)

# ==========================================
# 2. CALIBRATED PHYSICS ENGINE
# ==========================================

def get_parameters(weight_kg, diet, variability, activity_level):
    # 1. Volumes
    # Plasma/ECF Volume (L)
    v_central = weight_kg * 0.20 
    
    # Muscle Mass (kg Dry Mass)
    # Muscle is ~40% BW. Dry mass is ~25% of wet muscle.
    m_dm = weight_kg * 0.40 * 0.25 
    
    # 2. Steady State Estimates
    # Total Creatine Pool ~120 mmol/kg dm (Omnivore)
    m_base_target = 100.0 if diet == "Vegetarian/Vegan" else 125.0
    m_ceil = 160.0 
    
    # 3. Turnover (The degradation we must balance)
    # ~1.7% per day converts to Creatinine (Lost)
    activity_multiplier = {"Sedentary": 1.0, "Moderate (3x/wk)": 1.1, "Athlete (Daily)": 1.2}
    act_factor = activity_multiplier[activity_level]
    k_deg = 0.017 * act_factor
    
    # 4. Synthesis (Liver/Kidney)
    # Must match degradation at baseline: Synth = k_deg * Total_Pool
    total_pool_start = m_base_target * m_dm
    k_synth_base = k_deg * total_pool_start # e.g. ~15 mmol/day (2g)
    
    # 5. Transporter Kinetics (SLC6A8)
    # Vmax must be sufficient to maintain the pool
    vmax_m = 100.0 * variability * act_factor # mmol/day capacity
    km_m = 0.1 # mmol/L
    
    # 6. Brain
    k_brain_in = 0.0015 * variability
    
    return {
        'v_c': v_central, 'm_dm': m_dm, 
        'm_base': m_base_target, 'm_ceil': m_ceil,
        'k_deg': k_deg, 'k_synth': k_synth_base,
        'vmax_m': vmax_m, 'km_m': km_m,
        'k_brain_in': k_brain_in, 
        'k_ka': 15.0, # Absorption rate
        'renal_threshold': 0.06, # mmol/L (approx 60 uM)
        'gfr': 180.0 # L/day (Clearance capacity above threshold)
    }

def pbpk_model(y, t, dose_mmol, p):
    GI, Cp, M, B = y
    
    # 1. Synthesis (Feedback loop)
    # If GI intake is high, synthesis drops.
    # We allow synthesis to float to maintain baseline if GI is 0.
    synthesis = p['k_synth'] * (1 / (1 + (GI/2.0)))
    
    # 2. Renal Elimination (Threshold Logic)
    # Only clear creatine if Plasma > Threshold (Reabsorption saturation)
    excess_cp = max(0, Cp - p['renal_threshold'])
    renal_elim = p['gfr'] * excess_cp
    
    # 3. Muscle Uptake (Saturable)
    # Stops when M hits Ceiling
    sat_term = max(0, (p['m_ceil'] - M) / (p['m_ceil'] - 20))
    uptake_m = (p['vmax_m'] * Cp / (p['km_m'] + Cp)) * sat_term
    
    # 4. Fluxes
    absorption = p['k_ka'] * GI
    degradation = p['k_deg'] * M * p['m_dm']
    
    # ODEs
    dGI = -absorption
    dCp = (absorption + synthesis - uptake_m - renal_elim) / p['v_c']
    dM = (uptake_m / p['m_dm']) - (p['k_deg'] * M)
    dB = (p['k_brain_in'] * Cp * 100) - (0.017 * (B - 1.0))
    
    return [dGI, dCp, dM, dB]

def run_simulation(dose_g, params, total_days):
    # STEP 1: BURN-IN (Find Steady State)
    # We run 365 days at 0g. 
    # With the new Threshold logic, Cp should settle just below 0.06
    # and Muscle should settle exactly at m_base.
    y_guess = [0.0, 0.04, params['m_base'], 1.0]
    t_burn = np.linspace(0, 365, 3650)
    sol_burn = odeint(pbpk_model, y_guess, t_burn, args=(0, params))
    y_steady = sol_burn[-1]
    
    # STEP 2: SIMULATION
    t_span_daily = np.linspace(0, 1, 24)
    dose_mmol = dose_g / 131.13 * 1000
    
    res_list = []
    curr_y = y_steady.copy()
    
    for day in range(total_days):
        curr_y[0] += dose_mmol
        sol = odeint(pbpk_model, curr_y, t_span_daily, args=(dose_mmol, params))
        res_list.append(sol)
        curr_y = sol[-1]
        
    return np.vstack(res_list), y_steady

# ==========================================
# 3. UI
# ==========================================

st.set_page_config(page_title="Creatine PBPK Final", layout="wide")
display_intro()

with st.sidebar:
    st.header("1. Parameters")
    weight = st.number_input("Weight (kg)", 50, 120, 75)
    diet = st.selectbox("Diet", ["Omnivore", "Vegetarian/Vegan"])
    activity = st.selectbox("Activity", ["Sedentary", "Moderate (3x/wk)", "Athlete (Daily)"])
    var = st.slider("Genetic Responder", 0.8, 1.2, 1.0)
    
    st.header("2. Protocol")
    dose = st.slider("Daily Dose (g)", 0, 25, 5)
    months = st.slider("Duration (Months)", 1, 6, 3)

# Execution
P = get_parameters(weight, diet, var, activity)
sim, steady_state = run_simulation(dose, P, months * 30)
days_axis = np.linspace(0, months*30, len(sim))

# Metrics
m_ss = steady_state[2]
m_conc = sim[:, 2]
b_inc_pct = (sim[:, 3] - 1.0) * 5.0 

# Thresholds
day_benefit_m = next((d for d, s in zip(days_axis, m_conc) if s >= 140), None)
day_benefit_b = next((d for d, s in zip(days_axis, b_inc_pct) if s >= 3.0), None)

st.subheader("📊 Systemic Saturation Forecast")
c1, c2 = st.columns([1, 3])

with c1:
    st.markdown("### 🧪 Baseline Status")
    st.write(f"**Steady State Plasma:** {steady_state[1]*1000:.1f} µmol/L")
    st.write(f"**Muscle Baseline:** {m_ss:.1f} mmol/kg")
    
    st.markdown("### 🎯 Forecast")
    if dose == 0:
        st.info("System is at endogenous steady state.")
    else:
        if day_benefit_m: st.success(f"**Muscle Efficacy:** Day {int(day_benefit_m)}")
        else: st.warning("Muscle target (140 mmol/kg) not reached.")
        
        if day_benefit_b: st.info(f"**Brain Efficacy:** Day {int(day_benefit_b)}")
        else: st.error("Brain target (3% inc) not reached.")

with c2:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Plasma (mmol/L)", "Muscle Content (mmol/kg dm)", "Brain Increase (%)"))
    
    fig.add_trace(go.Scatter(x=days_axis, y=sim[:, 1], name="Plasma", line=dict(color='#FFA15A')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=days_axis, y=m_conc, name="Muscle", line=dict(color='#00CC96', width=3)), row=2, col=1)
    fig.add_hrect(y0=140, y1=160, fillcolor="green", opacity=0.1, annotation_text="Ergogenic Zone", row=2, col=1)
    
    fig.add_trace(go.Scatter(x=days_axis, y=b_inc_pct, name="Brain", line=dict(color='#636EFA')), row=3, col=1)
    fig.add_hrect(y0=3, y1=10, fillcolor="blue", opacity=0.1, annotation_text="Cognitive Zone", row=3, col=1)
    
    fig.update_xaxes(title_text="Time (Days)", row=3, col=1)
    fig.update_layout(height=700, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📚 Model Rationale (References)"):
    st.markdown("""
    1. **Renal Threshold:** The kidney reabsorbs creatine efficiently. Excretion only becomes significant when plasma concentration exceeds ~60 µmol/L (0.06 mmol/L). This ensures steady state is maintained at 0g dose [Persky et al., 2003].
    2. **Hultman et al. (1996):** Muscle creatine saturation kinetics. DOI: [10.1152/jappl.1996.81.1.232](https://doi.org/10.1152/jappl.1996.81.1.232)
    3. **Dolan et al. (2019):** Brain creatine uptake kinetics. DOI: [10.1007/s00421-019-04146-x](https://doi.org/10.1007/s00421-019-04146-x)
    """)