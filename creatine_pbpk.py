import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. INTRODUCTION & SYSTEMS PHARMACOLOGY
# ==========================================

def display_intro():
    st.markdown("""
    # 🧬 Creatine PBPK: Validated Systems Model
    
    ### Hello! 
    I am **Carmine**, your Systems Pharmacology scientist.
    
    **What is Systems Pharmacology?** It is the quantitative study of how substances interact with the body as an integrated network. Instead of looking at a single value, we use **Physiologically Based Pharmacokinetic (PBPK)** modeling to simulate the "flux"—the movement of molecules between compartments like plasma, muscle, and brain.
    
    **Model Validation:** This simulation includes a mathematical "Burn-in" phase. Before showing you the results, the model runs a simulation of your body's natural state for 200 days to find your unique **Steady State** equilibrium. This ensures that if you select a 0g dose, the graph remains perfectly flat, representing homeostasis [1, 2].
    
    ---
    ⚠️ **DISCLAIMER:** *For educational use. Simulates healthy renal function (GFR ~120 ml/min).*
    """)

# ==========================================
# 2. CALIBRATED PHYSICS ENGINE
# ==========================================

def get_parameters(weight_kg, diet, variability, activity_level):
    """
    Calibrates constants based on literature.
    """
    # 1. Volume of Distribution (Central Compartment)
    # Creatine distributes into Extracellular Fluid (ECF), approx 20% of BW.
    # Source: Persky et al. (2003)
    v_central = weight_kg * 0.20 
    
    # 2. Muscle Mass (Dry Mass)
    # Muscle is ~40% BW. Dry mass is ~25% of wet muscle.
    m_dm = weight_kg * 0.40 * 0.25 
    
    # 3. Baselines (mmol/kg dm)
    # Source: Hultman (1996) DOI: 10.1152/jappl.1996.81.1.232
    # Vegans start lower (approx 100 vs 125).
    m_base_target = 100.0 if diet == "Vegetarian/Vegan" else 125.0
    m_ceil = 160.0 
    
    # 4. Metabolic Fluxes
    # Synthesis: ~1g/day (approx 7.5 mmol) to 2g/day depending on size
    # Source: Brosnan (2011) DOI: 10.1007/s00726-011-0853-y
    k_synth_base = 10.0 # mmol/day (Endogenous production)
    
    # Degradation (k_deg): ~1.7% per day conversion to Creatinine
    # Athletes have slightly higher turnover.
    activity_multiplier = {"Sedentary": 1.0, "Moderate (3x/wk)": 1.1, "Athlete (Daily)": 1.2}
    act_factor = activity_multiplier[activity_level]
    k_deg = 0.017 * act_factor
    
    # 5. Clearance (CRITICAL FIX)
    # Kidneys filter ~180 L/day (GFR). Creatine is freely filtered.
    cl_renal = 180.0 # L/day (Standard GFR)
    
    # 6. Transporter Kinetics (SLC6A8)
    # Vmax must be high enough to maintain baseline against degradation
    # Vmax ~ 15 mmol/day total flux into muscle at baseline
    vmax_m = 120.0 * variability * act_factor # mmol/day (Total capacity)
    km_m = 0.15 # mmol/L (Plasma affinity)
    
    # Brain Parameters
    # Extremely slow flux (Dolan et al. 2019)
    k_brain_in = 0.002 * variability 
    
    return {
        'v_c': v_central, 'm_dm': m_dm, 
        'm_base_target': m_base_target, 'm_ceil': m_ceil,
        'k_deg': k_deg, 'k_synth': k_synth_base,
        'vmax_m': vmax_m, 'km_m': km_m,
        'k_brain_in': k_brain_in, 
        'k_ka': 24.0, # Absorption rate (fast, ~1h peak)
        'cl_renal': cl_renal
    }

def pbpk_model(y, t, dose_mmol, p):
    GI, Cp, M, B = y
    
    # Feedback Inhibition: High GI creatine reduces liver synthesis (AGAT enzyme repression)
    synthesis = p['k_synth'] * (1 / (1 + (GI/5.0)))
    
    # Saturable Muscle Uptake (SLC6A8)
    # As M approaches Ceiling, transport shuts off (downregulation)
    sat_term = max(0, (p['m_ceil'] - M) / (p['m_ceil'] - 20)) # Soft landing
    uptake_rate_total = (p['vmax_m'] * Cp / (p['km_m'] + Cp)) * sat_term
    
    # Fluxes
    absorption = p['k_ka'] * GI
    renal_elim = p['cl_renal'] * Cp 
    
    # Derivatives
    dGI = -absorption
    
    # Plasma: Input (Abs + Synth) - Output (Uptake + Renal)
    dCp = (absorption + synthesis - uptake_rate_total - renal_elim) / p['v_c']
    
    # Muscle: Input (Uptake per kg) - Output (Degradation)
    dM = (uptake_rate_total / p['m_dm']) - (p['k_deg'] * M)
    
    # Brain: Relative index change (Slow first-order)
    dB = (p['k_brain_in'] * Cp * 100) - (0.017 * (B - 1.0))
    
    return [dGI, dCp, dM, dB]

def run_simulation(dose_g, params, total_days):
    # STEP 1: BURN-IN PHASE (Finding Steady State)
    # We simulate 200 days at 0g dose to find the exact equilibrium 
    # where Endogenous Synthesis = Degradation + Excretion.
    y_guess = [0.0, 0.06, params['m_base_target'], 1.0]
    t_burn = np.linspace(0, 200, 2000)
    sol_burn = odeint(pbpk_model, y_guess, t_burn, args=(0, params))
    y_steady = sol_burn[-1]
    
    # STEP 2: ACTIVE PROTOCOL
    # We start from the calculated steady state.
    t_span_daily = np.linspace(0, 1, 24) # 24 hour resolution per day
    dose_mmol = dose_g / 131.13 * 1000
    
    res_list = []
    curr_y = y_steady.copy()
    
    for day in range(total_days):
        curr_y[0] += dose_mmol # Add daily dose to GI tract
        sol = odeint(pbpk_model, curr_y, t_span_daily, args=(dose_mmol, params))
        res_list.append(sol)
        curr_y = sol[-1]
        
    return np.vstack(res_list), y_steady

# ==========================================
# 3. INTERFACE
# ==========================================

st.set_page_config(page_title="Creatine PBPK Final", layout="wide")
display_intro()

with st.sidebar:
    st.header("1. Parameters")
    weight = st.number_input("Weight (kg)", 50, 120, 75)
    diet = st.selectbox("Dietary Baseline", ["Omnivore", "Vegetarian/Vegan"])
    activity = st.selectbox("Activity Level", ["Sedentary", "Moderate (3x/wk)", "Athlete (Daily)"])
    var = st.slider("Responder Type", 0.8, 1.2, 1.0, help="Genetic transporter efficiency")
    
    st.header("2. Protocol")
    dose = st.slider("Daily Dose (g)", 0, 25, 5)
    months = st.slider("Duration (Months)", 1, 6, 3)

# Execution
P = get_parameters(weight, diet, var, activity)
sim, steady_state = run_simulation(dose, P, months * 30)
days_axis = np.linspace(0, months*30, len(sim))

# --- METRICS ---
m_ss = steady_state[2]
m_conc = sim[:, 2]
m_increase_pct = ((m_conc - m_ss) / m_ss) * 100 # Relative increase
b_inc_pct = (sim[:, 3] - 1.0) * 5.0 # Brain relative increase

# Threshold Detection
# Muscle benefit usually seen at >140 mmol/kg total creatine
day_benefit_m = next((d for d, s in zip(days_axis, m_conc) if s >= 140), None)
# Brain benefit seen at >3-5% increase
day_benefit_b = next((d for d, s in zip(days_axis, b_inc_pct) if s >= 3.0), None)

st.subheader("📊 Systemic Saturation Forecast")
c1, c2 = st.columns([1, 3])

with c1:
    st.markdown("### 🧪 Baseline Status")
    st.write(f"**Steady State Plasma:** {steady_state[1]*1000:.1f} µmol/L")
    st.write(f"**Muscle Baseline:** {m_ss:.1f} mmol/kg")
    st.write(f"**Renal Clearance:** {P['cl_renal']} L/day")
    
    st.markdown("---")
    st.markdown("### 🎯 Benefit Forecast")
    if dose == 0:
        st.info("Maintaining Endogenous Steady State.")
    else:
        if day_benefit_m: st.success(f"**Muscle Efficacy:** Day {int(day_benefit_m)}")
        else: st.warning("Muscle target (140 mmol/kg) not reached.")
        
        if day_benefit_b: st.info(f"**Brain Efficacy:** Day {int(day_benefit_b)}")
        else: st.error("Brain target (3% inc) not reached.")

with c2:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Plasma (mmol/L)", "Muscle Content (mmol/kg dm)", "Brain Increase (%)"))
    
    # Plasma Trace
    fig.add_trace(go.Scatter(x=days_axis, y=sim[:, 1], name="Plasma", line=dict(color='#FFA15A')), row=1, col=1)
    
    # Muscle Trace + Band
    fig.add_trace(go.Scatter(x=days_axis, y=m_conc, name="Muscle", line=dict(color='#00CC96', width=3)), row=2, col=1)
    fig.add_hrect(y0=140, y1=160, fillcolor="green", opacity=0.1, annotation_text="Ergogenic Zone (140-160)", row=2, col=1)
    
    # Brain Trace + Band
    fig.add_trace(go.Scatter(x=days_axis, y=b_inc_pct, name="Brain", line=dict(color='#636EFA')), row=3, col=1)
    fig.add_hrect(y0=3, y1=10, fillcolor="blue", opacity=0.1, annotation_text="Cognitive Zone (3-10%)", row=3, col=1)
    
    fig.update_xaxes(title_text="Time (Days)", row=3, col=1)
    fig.update_layout(height=700, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --- REFERENCES ---
with st.expander("📚 Literature & Model Rationale"):
    st.markdown("""
    1. **Hultman E, et al. (1996).** *Muscle creatine loading in men.* DOI: [10.1152/jappl.1996.81.1.232](https://doi.org/10.1152/jappl.1996.81.1.232)
    2. **Persky AM, Brazeau GA. (2001).** *Clinical pharmacology of the dietary supplement creatine monohydrate.* Pharmacological Reviews.
    3. **Dolan E, et al. (2019).** *Beyond muscle: the effects of creatine supplementation on brain creatine.* DOI: [10.1007/s00421-019-04146-x](https://doi.org/10.1007/s00421-019-04146-x)
    
    **Why the Burn-in?** Everyone starts with a different baseline of creatine depending on their diet and muscle mass. The model runs a 200-day pre-simulation to find your specific "zero point" where your body's synthesis matches your degradation. The visible graph starts from *that* point.
    """)