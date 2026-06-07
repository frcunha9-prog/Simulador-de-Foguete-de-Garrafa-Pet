"""
Streamlit app — Simulador de Foguete de Garrafa PET.
Calcula o ângulo de lançamento ótimo para máximo alcance horizontal.
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

# ============================================================
# Constantes físicas
# ============================================================
G         = 9.81
RHO_W     = 1000.0
RHO_A     = 1.225
P_ATM     = 101_325.0
GAMMA     = 1.4
R_AIR     = 287.05
T_AMB     = 293.15
PSI_TO_PA = 6894.757


# ============================================================
# Núcleo de simulação
# ============================================================
def _simulate_once(angle_deg, params):
    """Simula uma trajetória e retorna (alcance, apogeu, t, x, z)."""
    (V_bottle, V_water, D_nozzle, D_body, m_dry,
     Cd, P0, Cd_nozzle, L_tube) = params

    A_throat = np.pi * (D_nozzle / 2) ** 2
    A_eff    = A_throat * Cd_nozzle
    A_body   = np.pi * (D_body / 2) ** 2

    V0_air = V_bottle - V_water
    if V0_air <= 0:
        raise ValueError("Volume de água deve ser menor que o da garrafa.")

    m_water0 = V_water * RHO_W
    m_air0   = P0 * V0_air / (R_AIR * T_AMB)
    p_burn   = P0    * (V0_air / V_bottle) ** GAMMA
    T_burn   = T_AMB * (V0_air / V_bottle) ** (GAMMA - 1)

    theta  = np.radians(angle_deg)
    u      = np.array([np.cos(theta), np.sin(theta)])
    sin_th = u[1]
    y0     = [0.0, 0.0, 0.0, 0.0, m_water0, m_air0]

    def deriv(t, y):
        x, z, vx, vz, m_w, m_a = y
        v_mag   = np.hypot(vx, vz)
        m_tot   = m_dry + max(m_w, 0.0)
        on_tube = np.hypot(x, z) < L_tube

        F_thrust = 0.0
        dm_w     = 0.0
        dm_a     = 0.0

        if m_w > 1e-7:
            V_air = V_bottle - m_w / RHO_W
            p_in  = P0 * (V0_air / V_air) ** GAMMA
            if p_in > P_ATM:
                V_e      = np.sqrt(2 * (p_in - P_ATM) / RHO_W)
                dm_w     = -RHO_W * A_eff * V_e
                F_thrust = -dm_w * V_e
        elif m_a > 1e-7:
            ratio = m_a / m_air0
            p_in  = p_burn * ratio ** GAMMA
            T_in  = T_burn * ratio ** (GAMMA - 1)
            if p_in > P_ATM and T_in > 0:
                pr   = P_ATM / p_in
                crit = (2 / (GAMMA + 1)) ** (GAMMA / (GAMMA - 1))
                if pr < crit:
                    dm_a = -A_eff * p_in * np.sqrt(GAMMA / (R_AIR * T_in)) * \
                           (2 / (GAMMA + 1)) ** ((GAMMA + 1) / (2 * (GAMMA - 1)))
                    V_e  = np.sqrt(GAMMA * R_AIR * T_in * 2 / (GAMMA + 1))
                    p_e  = p_in * crit
                else:
                    dm_a = -A_eff * p_in * np.sqrt(
                        (2 * GAMMA) / (R_AIR * T_in * (GAMMA - 1)) *
                        (pr ** (2 / GAMMA) - pr ** ((GAMMA + 1) / GAMMA))
                    )
                    V_e  = np.sqrt(2 * GAMMA * R_AIR * T_in / (GAMMA - 1) *
                                   (1 - pr ** ((GAMMA - 1) / GAMMA)))
                    p_e  = P_ATM
                F_thrust = -dm_a * V_e + (p_e - P_ATM) * A_throat

        F_drag = 0.5 * RHO_A * Cd * A_body * v_mag ** 2

        if on_tube:
            a_along = (F_thrust - F_drag) / m_tot - G * sin_th
            ax, az  = a_along * u[0], a_along * u[1]
        else:
            if v_mag > 1e-6:
                vhat = np.array([vx, vz]) / v_mag
            else:
                vhat = u
            ax = (F_thrust - F_drag) * vhat[0] / m_tot
            az = (F_thrust - F_drag) * vhat[1] / m_tot - G

        return [vx, vz, ax, az, dm_w, dm_a]

    def water_burnout(t, y):
        return y[4] - 1e-7
    water_burnout.terminal  = False
    water_burnout.direction = -1

    def hit_ground(t, y):
        return y[1] if t > 0.05 else 1.0
    hit_ground.terminal  = True
    hit_ground.direction = -1

    sol = solve_ivp(deriv, [0, 30], y0, events=[water_burnout, hit_ground],
                    method='RK45', max_step=5e-3, rtol=1e-7, atol=1e-10)
    x, z = sol.y[0], sol.y[1]

    # Estado no instante de esgotamento da água
    if sol.y_events[0].shape[0] > 0:
        yb   = sol.y_events[0][0]
        t_b  = float(sol.t_events[0][0])
        v_b  = float(np.hypot(yb[2], yb[3]))
        incl = float(np.degrees(np.arctan2(yb[3], yb[2])))
        x_b  = float(yb[0])
        z_b  = float(yb[1])
    else:
        t_b = v_b = incl = x_b = z_b = float('nan')

    burnout = {
        'p_burn_gauge': p_burn - P_ATM,   # Pa manométrico
        't_burn':       t_b,              # s
        'v_burn':       v_b,              # m/s
        'incl_burn':    incl,             # graus
        'x_burn':       x_b,              # m
        'z_burn':       z_b,              # m
    }
    return float(x[-1]), float(z.max()), sol.t, x, z, burnout


def best_angle(V_bottle_L, V_water_L, D_nozzle_mm, D_body_mm,
               m_dry_g, Cd, P0_psi, Cd_nozzle=0.97, L_tube_cm=100.0):
    """Encontra o ângulo ótimo e devolve resultados + curva alcance×ângulo."""
    params = (
        V_bottle_L * 1e-3,
        V_water_L  * 1e-3,
        D_nozzle_mm * 1e-3,
        D_body_mm   * 1e-3,
        m_dry_g     * 1e-3,
        Cd,
        P0_psi * PSI_TO_PA + P_ATM,
        Cd_nozzle,
        L_tube_cm * 1e-2,
    )

    # Varredura grosseira + refino com Brent
    angles = np.arange(15.0, 75.5, 1.0)
    ranges = np.array([_simulate_once(a, params)[0] for a in angles])
    j      = int(np.argmax(ranges))
    lo     = angles[max(j - 3, 0)]
    hi     = angles[min(j + 3, len(angles) - 1)]
    res    = minimize_scalar(lambda a: -_simulate_once(a, params)[0],
                             bounds=(lo, hi), method='bounded',
                             options={'xatol': 0.05})
    ang_opt = float(res.x)
    rng_opt, ap_opt, t_opt, x_opt, z_opt, b = _simulate_once(ang_opt, params)

    return {
        'angle_deg':   round(ang_opt, 2),
        'range_m':     round(rng_opt, 2),
        'apogee_m':    round(ap_opt, 2),
        'p_burnout_psi':    round(b['p_burn_gauge'] / PSI_TO_PA, 1),
        'v_burnout_kmh':    round(b['v_burn'] * 3.6, 1),
        'incl_burnout_deg': round(b['incl_burn'], 2),
        't_burnout_s':      round(b['t_burn'], 3),
        'x_burnout_m':      b['x_burn'],
        'z_burnout_m':      b['z_burn'],
        'traj_x':      x_opt,
        'traj_z':      z_opt,
    }


# ============================================================
# Interface
# ============================================================
st.set_page_config(page_title="Foguete PET", page_icon="🚀", layout="centered")

st.title("🚀 Simulador de Foguete PET")
st.caption("Calcula o ângulo de lançamento ótimo para o maior alcance horizontal.")

with st.form("params"):
    st.subheader("Parâmetros do foguete")

    col1, col2 = st.columns(2)
    with col1:
        V_bottle_L  = st.number_input("Volume da garrafa (L)",   value=6.0)
        V_water_L   = st.number_input("Volume de água (L)",      value=2.0)
        D_nozzle_mm = st.number_input("Diâmetro do bocal (mm)",  value=10)
        D_body_mm   = st.number_input("Diâmetro do foguete (mm)", value=106.0)
        m_dry_g     = st.number_input("Massa seca (g)",          value=580.0)
    with col2:
        Cd          = st.number_input("Cd do corpo",             value=0.35)
        P0_psi      = st.number_input("Pressão inicial (PSI)",   value=200.0)
        Cd_nozzle   = st.number_input("Cd do bocal",             value=0.77)
        L_tube_cm   = st.number_input("Comprimento do tubo (cm)", value=100.0)

    submitted = st.form_submit_button("Simular Lançamento", type="primary",
                                      use_container_width=True)

if submitted:
    if V_water_L >= V_bottle_L:
        st.error("⚠️ O volume de água precisa ser menor que o volume da garrafa.")
    else:
        with st.spinner("Simulando trajetórias..."):
            try:
                r = best_angle(V_bottle_L, V_water_L, D_nozzle_mm, D_body_mm,
                               m_dry_g, Cd, P0_psi, Cd_nozzle, L_tube_cm)
            except Exception as e:
                st.error(f"Erro na simulação: {e}")
                st.stop()

        st.success("Resultado calculado!")

        c1, c2, c3 = st.columns(3)
        c1.metric("Ângulo ótimo", f"{r['angle_deg']}°")
        c2.metric("Alcance",      f"{r['range_m']} m")
        c3.metric("Apogeu",       f"{r['apogee_m']} m")

        st.markdown("**No esgotamento da água (burnout)**")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Pressão interna", f"{r['p_burnout_psi']} PSI")
        b2.metric("Velocidade", f"{r['v_burnout_kmh']} km/h")
        b3.metric("Inclinação", f"{r['incl_burnout_deg']}°",
                  delta=f"{round(r['incl_burnout_deg'] - r['angle_deg'], 1)}° vs lançamento",
                  delta_color="off")
        b4.metric("Instante", f"{r['t_burnout_s']} s")

        st.subheader("Trajetória")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(r['traj_x'], r['traj_z'], color='tab:blue', lw=2)
        ax.fill_between(r['traj_x'], 0, r['traj_z'], alpha=0.1, color='tab:blue')

        # Marca o momento em que a água se esgota
        ax.scatter([r['x_burnout_m']], [r['z_burnout_m']],
                   color='red', s=90, zorder=5)
        ax.annotate(f"água esgota\n(t = {r['t_burnout_s']} s)",
                    xy=(r['x_burnout_m'], r['z_burnout_m']),
                    xytext=(20, 28), textcoords='offset points',
                    fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_xlabel('x (m)')
        ax.set_ylabel('z (m)')
        ax.set_title(f"Trajetória ótima @ {r['angle_deg']}°")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'box')

        plt.tight_layout()
        st.pyplot(fig)

with st.expander("ℹ️ Sobre o modelo"):
    st.markdown("""
    **Física do simulador**

    Três fases consecutivas:

    1. **Empuxo por água** — o ar comprimido empurra a água pelo bocal.
       Expansão adiabática $pV^\\gamma = \\mathrm{const}$ + Bernoulli para a
       velocidade de saída da água.
    2. **Empuxo por ar** — após o esgotamento da água, o ar residual escapa
       pelo bocal. Escoamento isentrópico compressível, com checagem
       automática de choke ($p_{atm}/p_{in} < 0{,}528$).
    3. **Voo balístico** — gravidade + arrasto aerodinâmico em 2D.

    O **tubo de lançamento** é modelado explicitamente: enquanto o foguete
    estiver dentro do tubo, o movimento é confinado 1D na direção do tubo.

    **Otimização**: varredura grosseira de 15° a 75° em passos de 1°,
    seguida de refino com Brent (`scipy.optimize.minimize_scalar`) com
    precisão de 0,05°.

    **Calibração**: o parâmetro mais incerto é o $C_d$ do corpo. Para
    resultados confiáveis, faça lançamentos verticais reais, meça o apogeu
    com câmera, e ajuste o $C_d$ até o simulador prever o apogeu observado.
    """)
