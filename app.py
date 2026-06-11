import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- 1. PHYSICAL SYSTEM MODEL (THE PLANT) ---
def tank_dynamics(h, t, u, Area, C_out):
    h = max(0.0, float(h[0]))
    inflow = max(0.0, min(10.0, u))
    outflow = C_out * np.sqrt(h) if h > 0 else 0.0
    dhdt = (inflow - outflow) / Area
    return [dhdt]

# --- 2. SIMULATION RUNNER ---
def simulate_system(pid_params, setpoint, sim_time, Area, C_out, apply_disturbance=False):
    Kp, Ki, Kd = pid_params
    dt = 0.1
    t_steps = np.arange(0, sim_time, dt)
    
    h_history = []
    u_history = []
    h_current = [0.0] 
    integral = 0.0
    last_error = 0.0
    
    for t in t_steps:
        # THE DISTURBANCE: At exactly 20 seconds, drain 1.5m of water instantly
        if apply_disturbance and abs(t - 20.0) < 0.05:
            h_current[0] = max(0.0, h_current[0] - 1.5)
            integral = 0.0 # Reset integral to simulate the system being "surprised"

        error = setpoint - h_current[0]
        integral += error * dt
        derivative = (error - last_error) / dt if t > 0 else 0.0
        
        u = (Kp * error) + (Ki * integral) + (Kd * derivative)
        u = max(0.0, min(10.0, u)) 
        
        t_span = [t, t + dt]
        h_next = odeint(tank_dynamics, h_current, t_span, args=(u, Area, C_out))
        
        h_history.append(h_current[0])
        u_history.append(u)
        
        h_current = h_next[-1]
        last_error = error
        
    return t_steps, np.array(h_history), np.array(u_history)

# --- 3. OPTIMIZATION OBJECTIVE FUNCTION (UPDATED) ---
def compute_fitness(pid_params, setpoint, sim_time, Area, C_out):
    # Train WITHOUT the disturbance
    _, h, u = simulate_system(pid_params, setpoint, sim_time, Area, C_out, apply_disturbance=False)
    
    # 1. Level Tracking Error
    error = np.abs(setpoint - h)
    iae = np.sum(error)
    
    # 2. Overshoot Penalty
    overshoot = max(0.0, np.max(h) - setpoint)
    
    # 3. THE NEW MECHANICAL WEAR PENALTY
    # np.diff calculates the difference between consecutive valve movements. 
    # Adding absolute values means ALL movement (up or down) gets penalized if it's excessive.
    valve_chatter = np.sum(np.abs(np.diff(u)))
    
    # Total Cost = Error + (Heavy Overshoot Penalty) + (Moderate Wear Penalty)
    return iae + (10.0 * overshoot) + (2.0 * valve_chatter)

# --- 4. GENETIC ALGORITHM (THE TUNER) ---
def run_genetic_algorithm(setpoint, sim_time, Area, C_out, generations=10, pop_size=12):
    population = np.random.rand(pop_size, 3) * 15.0
    best_param = None
    best_score = float('inf')
    
    for gen in range(generations):
        scores = [compute_fitness(ind, setpoint, sim_time, Area, C_out) for ind in population]
        
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_param = population[min_idx]
            
        sorted_indices = np.argsort(scores)
        parents = population[sorted_indices[:pop_size // 2]]
        
        new_pop = list(parents)
        while len(new_pop) < pop_size:
            p1, p2 = parents[np.random.randint(0, len(parents))], parents[np.random.randint(0, len(parents))]
            child = (p1 + p2) / 2.0  
            child += np.random.normal(0, 0.5, 3) 
            child = np.clip(child, 0.0, 15.0)
            new_pop.append(child)
            
        population = np.array(new_pop)
        
    return best_param

# --- 5. STREAMLIT WEB INTERFACE ---
st.set_page_config(layout="wide")
st.title("Digital Twin: ML-Optimized PID Control System")
st.markdown("This dashboard simulates a non-linear fluid tank dynamics environment and uses a Genetic Algorithm to discover optimal PID coefficients.")

# Sidebar Controls
st.sidebar.header("Plant Configurations")
Area = st.sidebar.slider("Tank Cross-Sectional Area", 1.0, 10.0, 5.0)
C_out = st.sidebar.slider("Outflow Valve Coefficient", 0.1, 2.0, 0.5)
setpoint = st.sidebar.slider("Target Liquid Level (m)", 1.0, 5.0, 3.0)

st.sidebar.header("Manual Tuning Controls")
manual_kp = st.sidebar.slider("Manual Kp", 0.0, 15.0, 1.0)
manual_ki = st.sidebar.slider("Manual Ki", 0.0, 5.0, 0.1)
manual_kd = st.sidebar.slider("Manual Kd", 0.0, 5.0, 0.0)

st.sidebar.divider() 
st.sidebar.subheader("Chaos Testing")
inject_disturbance = st.sidebar.toggle("💥 Simulate Pipe Burst at t=20s", value=False)

st.sidebar.divider() 
show_advanced = st.sidebar.toggle("⚙️ Developer Mode: Show Analytics", value=False)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Manual vs. AI-Optimized System Tracking")
    if st.button("🧬 Run Genetic Algorithm Optimizer"):
        with st.spinner("Executing Genetic Search Optimization..."):
            best_pid = run_genetic_algorithm(setpoint, 40, Area, C_out)
            st.session_state['optimized_pid'] = best_pid
            st.success(f"Optimal Parameters Found: Kp={best_pid[0]:.2f}, Ki={best_pid[1]:.2f}, Kd={best_pid[2]:.2f}")

    t, h_man, u_man = simulate_system([manual_kp, manual_ki, manual_kd], setpoint, 40, Area, C_out, apply_disturbance=inject_disturbance)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot Level
    ax[0].plot(t, h_man, label="Manual Tuning Profile", color="orange", linestyle="--")
    ax[0].axhline(y=setpoint, color="red", linestyle=":", label="Target Setpoint")
    if 'optimized_pid' in st.session_state:
        _, h_opt, _ = simulate_system(st.session_state['optimized_pid'], setpoint, 40, Area, C_out, apply_disturbance=inject_disturbance)
        ax[0].plot(t, h_opt, label="Genetic Algorithm Optimized Profile", color="cyan")
    ax[0].set_ylabel("Liquid Level (meters)")
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot Valve Output
    ax[1].plot(t, u_man, label="Manual Valve Action", color="orange", linestyle="--")
    if 'optimized_pid' in st.session_state:
        _, _, u_opt = simulate_system(st.session_state['optimized_pid'], setpoint, 40, Area, C_out, apply_disturbance=inject_disturbance)
        ax[1].plot(t, u_opt, label="Optimized Valve Action", color="cyan")
    ax[1].set_ylabel("Valve Opening (U)")
    ax[1].set_xlabel("Time (seconds)")
    ax[1].legend()
    ax[1].grid(True)
    
    st.pyplot(fig)

with col2:
    if show_advanced:
        st.subheader("Advanced Engineering Diagnostics")
        
        if 'optimized_pid' in st.session_state:
            _, h_opt, u_opt = simulate_system(st.session_state['optimized_pid'], setpoint, 40, Area, C_out, apply_disturbance=inject_disturbance)
            
            max_height = np.max(h_opt)
            overshoot = max(0.0, ((max_height - setpoint) / setpoint) * 100)
            steady_state_error = abs(setpoint - h_opt[-1])
            valve_chatter = np.sum(np.abs(np.diff(u_opt)))
            
            st.markdown("### Model Performance")
            st.write(f"**Maximum Overshoot:** {overshoot:.2f}%")
            st.write(f"**Steady-State Error:** {steady_state_error:.4f} meters")
            st.write(f"**Valve Chatter Profile:** {valve_chatter:.2f} total units")
            
            if steady_state_error < 0.05 and overshoot < 5.0 and valve_chatter < 200:
                st.success("✅ System Stable: Controller maintained target with acceptable control effort.")
            else:
                st.warning("⚠️ Warning: System struggling or valve chatter is too high.")
                
            st.divider()

        st.markdown("### Underlying Plant Mathematics")
        st.markdown(r"""
        **Dynamic Non-Linear ODE:**
        $$\frac{dh}{dt} = \frac{U(t) - C_{out}\sqrt{h(t)}}{Area}$$
        
        **Algorithm Cost Function (IAE + Overhoot + Control Effort):**
        $$\text{Cost} = \int_{0}^{T} |r(t) - h(t)|\,dt + w_1 \cdot \max(0, h_{max} - r) + w_2 \sum |\Delta U|$$
        """)
    else:
        st.subheader("System Status")
        st.info("The digital twin is running in Operator Mode. All complex plant mathematics and ML cost functions are hidden to prioritize critical level-tracking visuals.")