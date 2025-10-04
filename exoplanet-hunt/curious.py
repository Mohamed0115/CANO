# ========================== ✅
# curious.py
# ==========================
import streamlit as st
import numpy as np
import plotly.graph_objects as go

def simulate_light_curve(num_planets=1, radii=None, periods=None, steps=300):
    """
    Simulate normalized light curve for 1–N planets,
    synced with orbital positions.
    """
    time = np.linspace(0, 30, steps)  # 30 days observation
    flux = np.ones_like(time)

    if radii is None:
        radii = [1.0] * num_planets
    if periods is None:
        periods = [5.0] * num_planets

    for i in range(num_planets):
        radius = radii[i]
        period = periods[i]
        depth = radius * 0.01  # bigger planet = deeper dip

        # orbital angle for each timestep
        theta = 2 * np.pi * (time / period)  # angle along orbit
        x = np.cos(theta)  # planet x-position (edge-on view)

        # Transit when x ≈ 0 (planet in front of star)
        in_transit = np.abs(x) < 0.05  # tolerance threshold
        flux[in_transit] -= depth

    return time, flux


def show_curious():
    # --------------------------
    # Page Setup
    # --------------------------
    st.title("🌍 Curious Explorer")
    st.write("Learn how scientists detect exoplanets using the transit method.")

    # --------------------------
    # Part 1: Transit Game
    # --------------------------
    st.subheader("✨ Transit Game (Interactive Demo)")

    st.write("Adjust the sliders to create your own planetary system and see the light curve dips!")

    # Controls
    num_planets = st.slider("Number of Planets", 1, 3, 1)

    radii = []
    periods = []
    for i in range(num_planets):
        st.markdown(f"**Planet {i+1} settings**")
        col1, col2 = st.columns(2)
        with col1:
            radius = st.slider(f"Planet {i+1} Radius (Earth units)", 0.1, 2.0, 1.0, 0.1, key=f"radius_{i}")
        with col2:
            period = st.slider(f"Planet {i+1} Orbital Period (days)", 2.0, 15.0, 5.0, 0.5, key=f"period_{i}")
        radii.append(radius)
        periods.append(period)

    # Simulate curve
    time, flux = simulate_light_curve(num_planets, radii, periods)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=flux, mode="lines", name="Light Curve"))
    fig.update_layout(
        title="Simulated Light Curve",
        xaxis_title="Time (days)",
        yaxis_title="Normalized Brightness",
        yaxis=dict(range=[0.9, 1.01]),
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("👉 When a planet crosses in front of its star, brightness dips. Bigger planets create deeper dips.")
        # --------------------------
    # Part 1b: Orbit Animation
    # --------------------------
    st.subheader("🌌 Orbit Simulation")

    st.write("Watch the planets orbit their star (edge-on view). When aligned, the light curve dips!")

    steps = 100
    theta = np.linspace(0, 2*np.pi, steps)

    # Planet orbits
    orbit_traces = []
    frames = []

    for i in range(num_planets):
        r = (periods[i]) * 0.3  # orbit size scaling
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        orbit_traces.append((x, y))

    # Base figure
    fig_orbit = go.Figure()

    # Add star
    fig_orbit.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers", marker=dict(size=20, color="yellow"),
        name="Star"
    ))

    # Add orbits (faint circles)
    for i, (x, y) in enumerate(orbit_traces):
        fig_orbit.add_trace(go.Scatter(
            x=x, y=y, mode="lines", line=dict(dash="dot"),
            name=f"Planet {i+1} orbit", showlegend=False
        ))

    # Add initial planet positions
    planet_traces = []
    for i, (x, y) in enumerate(orbit_traces):
        trace = go.Scatter(
            x=[x[0]], y=[y[0]], mode="markers",
            marker=dict(size=10*radii[i]*5, color="cyan"),
            name=f"Planet {i+1}"
        )
        fig_orbit.add_trace(trace)
        planet_traces.append(trace)

    # Animation frames
    for t in range(steps):
        frame_data = []
        for i, (x, y) in enumerate(orbit_traces):
            frame_data.append(go.Scatter(
                x=[x[t]], y=[y[t]], mode="markers",
                marker=dict(size=10*radii[i]*5, color="cyan"),
                name=f"Planet {i+1}"
            ))
        frames.append(go.Frame(data=frame_data, name=str(t)))

    fig_orbit.frames = frames
    fig_orbit.update_layout(
        title="Planetary Orbits (Top View)",
        xaxis=dict(scaleanchor="y", showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        template="plotly_dark",
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 80}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
            ]
        }]
    )

    st.plotly_chart(fig_orbit, use_container_width=True)

    # --------------------------
    # Part 2: Explainer Hub
    # --------------------------
    st.subheader("🎥 Explainer Hub")
    st.write("Curated resources to understand exoplanets:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📺 Videos**")
        st.markdown("- [How NASA Finds Exoplanets](https://www.youtube.com/watch?v=BFi4HBUdWkk)")
        st.markdown("- [Transit Method Explained](https://www.youtube.com/watch?v=V9f2rS1QbbQ)")

    with col2:
        st.markdown("**📚 Articles & Podcasts**")
        st.markdown("- [NASA Exoplanet Exploration](https://exoplanets.nasa.gov/)")
        st.markdown("- [Exoplanet Podcasts](https://exoplanets.nasa.gov/resources/)")
    # Back to Home Button
    if st.button("🏠 Back to Home"):
        st.session_state["mode"] = "home"
        st.rerun()
