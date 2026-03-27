import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from io import BytesIO
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

st.set_page_config(page_title="TEFSM HydroPick", layout="wide", page_icon="🧪")

# ====================== AUTENTICACIÓN ======================
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login()

if authentication_status == False:
    st.error("Usuario o contraseña incorrectos")
elif authentication_status == None:
    st.warning("Por favor ingresa tu usuario y contraseña")
elif authentication_status:
    authenticator.logout("Cerrar sesión", "sidebar")
    st.title("🧪 TEFSM HydroPick - Post-procesador PQWT")
    st.markdown("**Análisis automático de anomalías TEFSM**")

    # ====================== SUBIR CSV ======================
    uploaded_file = st.file_uploader("Sube tu archivo CSV del PQWT", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.sidebar.header("⚙️ Configuración")
        line_length = st.sidebar.number_input("Longitud total de la línea (m)", value=300.0, step=10.0)
        max_depth = st.sidebar.number_input("Profundidad máxima (m)", value=200.0, step=10.0)
        anomaly_width = st.sidebar.slider("Ancho de anomalía (m)", 5, 30, 15)

        n_points = len(df)
        spacing = line_length / (n_points - 1) if n_points > 1 else 10
        df['Distance'] = (df['N'] - 1) * spacing

        freq_cols = [col for col in df.columns if col.startswith('freq')]

        # Análisis automático
        df['avg_potential'] = df[freq_cols].mean(axis=1)
        anomaly_idx = df['avg_potential'].idxmin()
        anomaly_dist = df.loc[anomaly_idx, 'Distance']

        st.success(f"✅ Anomalía principal detectada en **{anomaly_dist:.1f} m** (punto N={df.loc[anomaly_idx, 'N']})")

        tab1, tab2, tab3 = st.tabs(["📈 Curvas de Frecuencia", "🗺️ Sección 2D", "📉 Curva SP"])

        with tab1:
            fig1, ax1 = plt.subplots(figsize=(14, 6))
            for col in freq_cols:
                ax1.plot(df['Distance'], df[col], lw=1.2, alpha=0.75, label=col)
            ax1.axvspan(anomaly_dist - anomaly_width/2, anomaly_dist + anomaly_width/2, alpha=0.3, color='red')
            ax1.axvline(anomaly_dist, color='red', linestyle='--', linewidth=2)
            ax1.set_title("PQWT Frequency Curves - Main Anomaly")
            ax1.set_xlabel("Distance (m)")
            ax1.set_ylabel("Potential (mV)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), fontsize=8)
            st.pyplot(fig1)
            buf = BytesIO()
            fig1.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            st.download_button("⬇️ Descargar Curvas", buf.getvalue(), "frequency_curves.png", "image/png")

        with tab2:
            depth_levels = np.linspace(0, max_depth, len(freq_cols))
            n_freq = len(freq_cols)
            x_points = np.repeat(df['Distance'].values, n_freq)
            z_points = np.tile(depth_levels, n_points)
            values = df[freq_cols].values.ravel()

            xi = np.linspace(0, line_length, 400)
            zi = np.linspace(0, max_depth, 200)
            xi, zi = np.meshgrid(xi, zi)
            vi = griddata((x_points, z_points), values, (xi, zi), method='cubic')

            fig2, ax2 = plt.subplots(figsize=(14, 8))
            im = ax2.imshow(vi, extent=[0, line_length, max_depth, 0], aspect='auto', cmap='jet', origin='upper')
            plt.colorbar(im, ax=ax2, label='Potencial (mV)')
            ax2.axvline(anomaly_dist, color='red', linewidth=3, label='Anomalía')
            ax2.axvspan(anomaly_dist - anomaly_width/2, anomaly_dist + anomaly_width/2, alpha=0.3, color='red')
            ax2.set_title("PQWT 2D Section")
            ax2.set_xlabel("Distance (m)")
            ax2.set_ylabel("Depth (m)")
            ax2.invert_yaxis()
            ax2.legend()
            st.pyplot(fig2)
            buf2 = BytesIO()
            fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
            st.download_button("⬇️ Descargar Sección 2D", buf2.getvalue(), "2D_section.png", "image/png")

        with tab3:
            sp_values = df.loc[anomaly_idx, freq_cols].values
            fig3, ax3 = plt.subplots(figsize=(6, 9))
            ax3.plot(sp_values, depth_levels, 'b-', linewidth=2.5)
            ax3.fill_betweenx(depth_levels, sp_values, sp_values.mean(), color='red', alpha=0.15)
            ax3.set_title(f"SP Curve at {anomaly_dist:.1f} m")
            ax3.set_xlabel("Potential (mV)")
            ax3.set_ylabel("Depth (m)")
            ax3.grid(True, alpha=0.3)
            ax3.invert_yaxis()
            rect = plt.Rectangle((0, 80), 0.6, 60, fill=False, edgecolor='red', linewidth=3)
            ax3.add_patch(rect)
            st.pyplot(fig3)
            buf3 = BytesIO()
            fig3.savefig(buf3, format="png", dpi=300, bbox_inches="tight")
            st.download_button("⬇️ Descargar Curva SP", buf3.getvalue(), "SP_curve.png", "image/png")

# ====================== CONFIGURACIÓN DE USUARIOS ======================