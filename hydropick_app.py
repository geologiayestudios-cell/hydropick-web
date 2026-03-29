import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from io import BytesIO
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from datetime import datetime

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

authenticator.login()

if st.session_state.get("authentication_status") == True:
    authenticator.logout("Cerrar sesión", "sidebar")
    
    st.title("🧪 TEFSM HydroPick - Post-procesador PQWT")
    st.markdown("**Análisis automático generalizado para cualquier geología**")

    uploaded_file = st.file_uploader("Sube tu archivo CSV del PQWT", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error al leer el CSV: {e}")
            st.stop()

        # Validación de columnas esenciales
        if 'N' not in df.columns:
            st.error("El CSV debe contener la columna 'N'")
            st.stop()
        freq_cols = [col for col in df.columns if col.startswith('freq')]
        if len(freq_cols) == 0:
            st.error("No se encontraron columnas freq01, freq02...")
            st.stop()

        # ====================== CONFIGURACIÓN GEOLÓGICA ======================
        st.sidebar.header("🪨 Configuración Geológica")
        geology_type = st.sidebar.selectbox(
            "Tipo de terreno principal", 
            ["Granito / Ígnea dura", "Sedimentaria", "Metamórfica / Mixta", "Volcánica", "Personalizado"]
        )
        if geology_type == "Granito / Ígnea dura": 
            default_rho, default_th = 200, 0.085
        elif geology_type == "Sedimentaria": 
            default_rho, default_th = 50, 0.15
        elif geology_type == "Metamórfica / Mixta": 
            default_rho, default_th = 150, 0.10
        elif geology_type == "Volcánica": 
            default_rho, default_th = 80, 0.12
        else: 
            default_rho, default_th = 100, 0.10

        rho = st.sidebar.number_input("Resistividad asumida (Ω·m)", value=default_rho, min_value=10, max_value=2000, step=10)
        low_epd_threshold = st.sidebar.number_input("Umbral EPD máximo para anomalía (mV)", 
                                                   value=default_th, step=0.01)

        line_length = st.sidebar.number_input("Longitud total de la línea (m)", value=300.0, step=10.0)
        max_depth = st.sidebar.number_input("Profundidad máxima (m)", value=200.0, step=10.0)
        anomaly_width = st.sidebar.slider("Ancho de zona de anomalía (m)", 5, 30, 15)

        # ====================== CÁLCULOS ======================
        n_points = len(df)
        spacing = line_length / (n_points - 1) if n_points > 1 else 10
        df['Distance'] = (df['N'] - 1) * spacing

        # Profundidad física real (solo hasta las frecuencias disponibles)
        freq_values = np.array([2520,2000,1600,1250,1000,800,630,500,400,315,250,200,160,125,100,
                                80,63,50,40,31.5,25,20,16,12.5,10,8,6.3,5,4,3.15,2.5,2,1.6,1.25,1,
                                0.8,0.63,0.5,0.4,0.315])[:len(freq_cols)]
        depth_levels = 0.40 * 503.3 * np.sqrt(rho / freq_values)

        df['avg_potential'] = df[freq_cols].mean(axis=1)
        percentile_20 = np.percentile(df[freq_cols].values, 20)
        df['anomaly_score'] = ((df[freq_cols] < low_epd_threshold) & 
                               (df[freq_cols] < percentile_20)).sum(axis=1)

        anomaly_candidates = df[df['anomaly_score'] >= 3]
        if not anomaly_candidates.empty:
            anomaly_idx = anomaly_candidates['avg_potential'].idxmin()
        else:
            anomaly_idx = df['avg_potential'].idxmin()

        anomaly_dist = df.loc[anomaly_idx, 'Distance']

        st.success(f"✅ **Anomalía principal detectada** en **{anomaly_dist:.1f} m** (punto N={df.loc[anomaly_idx, 'N']})")

        # ====================== PESTAÑAS ======================
        tab1, tab2, tab3 = st.tabs(["📈 Curvas de Frecuencia", "🗺️ Sección 2D", "📉 Curva SP Vertical"])

        with tab1:
            fig1, ax1 = plt.subplots(figsize=(14, 6))
            for col in freq_cols:
                ax1.plot(df['Distance'], df[col], lw=1.2, alpha=0.75, label=col)
            ax1.axvspan(anomaly_dist - anomaly_width/2, anomaly_dist + anomaly_width/2, alpha=0.3, color='red')
            ax1.axvline(anomaly_dist, color='red', linestyle='--', linewidth=2)
            ax1.set_title("Curvas de Frecuencia PQWT")
            ax1.set_xlabel("Distancia (m)")
            ax1.set_ylabel("Potencial (mV)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), fontsize=8)
            st.pyplot(fig1)

        with tab2:
            st.subheader("Sección Geofísica 2D")
            n_freq = len(freq_cols)
            x_points = np.repeat(df['Distance'].values, n_freq)
            z_points = np.tile(depth_levels, n_points)
            values = df[freq_cols].values.ravel()

            actual_max_d = min(max_depth, depth_levels.max())
            xi = np.linspace(0, line_length, 400)
            zi = np.linspace(0, actual_max_d, 200)
            xi, zi = np.meshgrid(xi, zi)
            
            vi = griddata((x_points, z_points), values, (xi, zi), method='linear')
            vi = np.nan_to_num(vi, nan=np.nanmin(vi))   # elimina cualquier banda superior

            fig2, ax2 = plt.subplots(figsize=(14, 8))
            im = ax2.imshow(vi, extent=[0, line_length, actual_max_d, 0], 
                           aspect='auto', cmap='jet', origin='upper')
            plt.colorbar(im, ax=ax2, label='Potencial (mV)')
            ax2.axvline(anomaly_dist, color='red', linewidth=3, label='Anomalía')
            ax2.axvspan(anomaly_dist - anomaly_width/2, anomaly_dist + anomaly_width/2, alpha=0.3, color='red')
            ax2.set_title("Sección Geofísica 2D")
            ax2.set_xlabel("Distancia (m)")
            ax2.set_ylabel("Profundidad (m)")
            ax2.set_ylim(actual_max_d, 0)   # ← CORREGIDO: solo este comando
            ax2.legend()
            st.pyplot(fig2)

        with tab3:
            st.subheader(f"Curva SP en la anomalía ({anomaly_dist:.1f} m)")
            mask = depth_levels <= max_depth
            sp_plot = df.loc[anomaly_idx, freq_cols].values[mask]
            depth_plot = depth_levels[mask]

            fig3, ax3 = plt.subplots(figsize=(6, 9))
            ax3.plot(sp_plot, depth_plot, 'b-', linewidth=2.5)
            ax3.fill_betweenx(depth_plot, sp_plot, sp_plot.mean(), color='red', alpha=0.15)
            
            # Rectángulo rojo DINÁMICO
            x_min, x_max = sp_plot.min(), sp_plot.max()
            rect_width = (x_max - x_min) * 0.8
            rect_y = max_depth * 0.4
            rect_h = max_depth * 0.3
            rect = plt.Rectangle((x_min, rect_y), rect_width, rect_h, 
                                 fill=False, edgecolor='red', linewidth=3)
            ax3.add_patch(rect)

            ax3.set_title(f"Curva SP Vertical en {anomaly_dist:.1f} m")
            ax3.set_xlabel("Potencial (mV)")
            ax3.set_ylabel("Profundidad (m)")
            ax3.grid(True, alpha=0.3)
            ax3.invert_yaxis()
            st.pyplot(fig3)

        # ====================== INFORME PDF ======================
        if st.button("📄 Generar Informe PDF (1 hoja profesional)", type="primary", use_container_width=True):
            with st.spinner("Creando informe PDF..."):
                fig_pdf = plt.figure(figsize=(14, 11))
                gs = fig_pdf.add_gridspec(3, 1, height_ratios=[1.2, 2.2, 1.2], hspace=0.35)
                fig_pdf.suptitle(f"INFORME TEFSM HYDROPICK - {geology_type}\n"
                                 f"Anomalía principal en {anomaly_dist:.1f} m | {datetime.now().strftime('%d/%m/%Y')}", 
                                 fontsize=16, fontweight='bold')

                ax1p = fig_pdf.add_subplot(gs[0])
                for col in freq_cols:
                    ax1p.plot(df['Distance'], df[col], lw=1.2, alpha=0.75)
                ax1p.axvspan(anomaly_dist - anomaly_width/2, anomaly_dist + anomaly_width/2, alpha=0.3, color='red')
                ax1p.axvline(anomaly_dist, color='red', linestyle='--', linewidth=2)
                ax1p.set_title("Curvas de Frecuencia PQWT")
                ax1p.set_ylabel("Potencial (mV)")
                ax1p.grid(True, alpha=0.3)

                ax2p = fig_pdf.add_subplot(gs[1])
                im = ax2p.imshow(vi, extent=[0, line_length, actual_max_d, 0], 
                                aspect='auto', cmap='jet', origin='upper')
                fig_pdf.colorbar(im, ax=ax2p, label='Potencial (mV)')
                ax2p.axvline(anomaly_dist, color='red', linewidth=3)
                ax2p.axvspan(anomaly_dist - anomaly_width/2, anomaly_dist + anomaly_width/2, alpha=0.3, color='red')
                ax2p.set_title("Sección Geofísica 2D")
                ax2p.set_xlabel("Distancia (m)")
                ax2p.set_ylabel("Profundidad (m)")
                ax2p.set_ylim(actual_max_d, 0)

                ax3p = fig_pdf.add_subplot(gs[2])
                ax3p.plot(sp_plot, depth_plot, 'b-', linewidth=2.5)
                ax3p.fill_betweenx(depth_plot, sp_plot, sp_plot.mean(), color='red', alpha=0.15)
                ax3p.add_patch(plt.Rectangle((x_min, rect_y), rect_width, rect_h, 
                                            fill=False, edgecolor='red', linewidth=3))
                ax3p.set_title(f"Curva SP Vertical en {anomaly_dist:.1f} m")
                ax3p.set_xlabel("Potencial (mV)")
                ax3p.set_ylabel("Profundidad (m)")
                ax3p.grid(True, alpha=0.3)
                ax3p.invert_yaxis()

                buf_pdf = BytesIO()
                fig_pdf.savefig(buf_pdf, format="pdf", dpi=300, bbox_inches="tight")
                buf_pdf.seek(0)

                st.download_button(
                    label="⬇️ Descargar Informe PDF",
                    data=buf_pdf,
                    file_name=f"TEFSM_Informe_{anomaly_dist:.1f}m_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        st.info("🔬 Sección 2D corregida • Sin bandas superiores • Rectángulo SP dinámico")

else:
    st.warning("Por favor inicia sesión")

st.caption("TEFSM HydroPick Generalizado • Versión corregida")