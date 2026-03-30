import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from io import BytesIO
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from datetime import datetime

st.set_page_config(page_title="TEFSM", layout="wide", page_icon="🧪")

# ====================== AUTENTICACIÓN ======================
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("❌ Archivo 'config.yaml' no encontrado. Verifica que exista en el directorio raíz.")
    st.stop()

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

authenticator.login()

if st.session_state.get("authentication_status") == True:
    authenticator.logout("Cerrar sesión", "sidebar")

    st.title("🧪 TEFSM - Post-procesador")
    st.markdown("**Análisis automático generalizado para cualquier geología**")

    uploaded_file = st.file_uploader("Sube tu archivo CSV del PQWT", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # ====================== VALIDACIÓN DE COLUMNAS ======================
        freq_cols = [col for col in df.columns if col.startswith('freq')]

        if 'N' not in df.columns:
            st.error("❌ CSV inválido: falta la columna 'N' (número de punto).")
            st.stop()
        if not freq_cols:
            st.error("❌ CSV inválido: no se encontraron columnas 'freq*'.")
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

        rho               = st.sidebar.number_input("Resistividad asumida (Ω·m)", value=default_rho, min_value=10, max_value=2000, step=10)
        low_epd_threshold = st.sidebar.number_input("Umbral EPD máximo para anomalía (mV)", value=default_th, step=0.01)
        line_length       = st.sidebar.number_input("Longitud total de la línea (m)", value=18.0, step=1.0)
        max_depth         = st.sidebar.number_input("Profundidad máxima (m)", value=300.0, step=10.0)
        anomaly_width     = st.sidebar.slider("Ancho de zona de anomalía (m)", 5, 30, 5)

        # ====================== CÁLCULOS ======================
        n_points = len(df)

        if 'Distance' in df.columns:
            st.info("ℹ️ Se usará la columna 'Distance' del CSV en lugar de calcularla.")
        else:
            spacing = line_length / (n_points - 1) if n_points > 1 else 10
            df['Distance'] = (df['N'] - 1) * spacing

        # ------------------------------------------------------------------
        # TABLA DE FRECUENCIAS DINÁMICA — cubre 40 (150 m), 56 (500 m) y más.
        # Sigue la escala 1/3 de octava estándar del instrumento PQWT.
        # ------------------------------------------------------------------
        _freq_base = np.array([
            2520,    2000,    1600,    1250,    1000,    800,     630,     500,     400,     315,
            250,     200,     160,     125,     100,     80,      63,      50,      40,      31.5,
            25,      20,      16,      12.5,    10,      8,       6.3,     5,       4,       3.15,
            2.5,     2,       1.6,     1.25,    1,       0.8,     0.63,    0.5,     0.4,     0.315,
            0.25,    0.2,     0.16,    0.125,   0.1,     0.08,    0.063,   0.05,    0.04,    0.0315,
            0.025,   0.02,    0.016,   0.0125,  0.01,    0.008,   0.0063,  0.005,   0.004,   0.00315,
            0.0025,  0.002,   0.0016,  0.00125, 0.001,   0.0008,  0.00063, 0.0005,  0.0004,  0.000315,
            0.00025, 0.0002,  0.00016, 0.000125,0.0001,  0.00008, 0.000063,0.00005, 0.00004, 0.0000315,
        ])

        n_cols = len(freq_cols)
        if n_cols > len(_freq_base):
            # Extensión automática si el CSV supera los 80 niveles predefinidos
            extra     = n_cols - len(_freq_base)
            ratio     = _freq_base[-2] / _freq_base[-1]
            extension = _freq_base[-1] / (ratio ** np.arange(1, extra + 1))
            _freq_base = np.concatenate([_freq_base, extension])

        freq_values = _freq_base[:n_cols]
        n_usable    = n_cols

        # Profundidad física por nivel de frecuencia (fórmula skin-depth AMT)
        depth_levels = 0.40 * 503.3 * np.sqrt(rho / freq_values)

        df['avg_potential'] = df[freq_cols].mean(axis=1)
        percentile_20 = np.percentile(df[freq_cols].values, 20)
        df['anomaly_score'] = (
            (df[freq_cols] < low_epd_threshold) &
            (df[freq_cols] < percentile_20)
        ).sum(axis=1)

        anomaly_candidates = df[df['anomaly_score'] >= 3]
        anomaly_idx  = anomaly_candidates['avg_potential'].idxmin() if not anomaly_candidates.empty else df['avg_potential'].idxmin()
        anomaly_dist = df.loc[anomaly_idx, 'Distance']

        st.success(f"✅ **Anomalía principal detectada** en **{anomaly_dist:.1f} m** (punto N={df.loc[anomaly_idx, 'N']})")

        # ====================== INTERPOLACIÓN 2D ======================
        x_points = np.repeat(df['Distance'].values, n_usable)
        z_points = np.tile(depth_levels, n_points)
        values   = df[freq_cols].values.ravel()

        actual_max_d = min(max_depth, depth_levels.max())
        xi = np.linspace(0, line_length, 400)
        zi = np.linspace(0, actual_max_d, 200)
        xi, zi = np.meshgrid(xi, zi)

        # Paso 1: cúbico → máxima suavidad en zonas con datos
        # Paso 2: nearest → rellena NaNs de borde (zona sin datos) con el vecino más cercano
        # Paso 3: gaussian_filter → suavizado final para eliminar artefactos residuales
        try:
            vi = griddata((x_points, z_points), values, (xi, zi), method='cubic')
        except Exception:
            vi = griddata((x_points, z_points), values, (xi, zi), method='linear')

        nan_mask = np.isnan(vi)
        if nan_mask.any():
            vi_nearest   = griddata((x_points, z_points), values, (xi, zi), method='nearest')
            vi[nan_mask] = vi_nearest[nan_mask]

        vi = gaussian_filter(vi, sigma=2)   # sigma=2 px — ajustable si se quiere más/menos suavidad

        # ====================== CURVA SP ======================
        # Ordenar por profundidad ascendente — garantiza curva coherente sin saltos
        sort_idx            = np.argsort(depth_levels)
        depth_levels_sorted = depth_levels[sort_idx]
        freq_cols_sorted    = [freq_cols[i] for i in sort_idx]

        mask       = depth_levels_sorted <= max_depth
        sp_plot    = df.loc[anomaly_idx, freq_cols_sorted].values[mask]
        depth_plot = depth_levels_sorted[mask]

        min_sp_idx    = np.argmin(sp_plot)
        aquifer_depth = depth_plot[min_sp_idx]

        # ====================== PESTAÑAS ======================
        tab1, tab2, tab3 = st.tabs(["📈 Curvas de Frecuencia", "🗺️ Sección 2D", "📉 Curva SP Vertical"])

        # --- Tab 1: Curvas de Frecuencia ---
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(14, 6))
            for col in freq_cols:
                ax1.plot(df['Distance'], df[col], lw=1.2, alpha=0.75, label=col)
            ax1.axvspan(anomaly_dist - anomaly_width / 2, anomaly_dist + anomaly_width / 2, alpha=0.3, color='red')
            ax1.axvline(anomaly_dist, color='red', linestyle='--', linewidth=2)
            ax1.set_title("Curvas de Frecuencia PQWT")
            ax1.set_xlabel("Distancia (m)")
            ax1.set_ylabel("Potencial (mV)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), fontsize=8)
            st.pyplot(fig1)

        # --- Tab 2: Sección Geofísica 2D ---
        with tab2:
            st.subheader("Sección Geofísica 2D")
            fig2, ax2 = plt.subplots(figsize=(14, 8))
            im = ax2.imshow(
                vi,
                extent=[0, line_length, actual_max_d, 0],
                aspect='auto', cmap='jet', origin='upper',
                vmin=np.percentile(vi, 2), vmax=np.percentile(vi, 98)   # recorte de outliers
            )
            plt.colorbar(im, ax=ax2, label='Potencial (mV)')
            ax2.axvline(anomaly_dist, color='red', linewidth=3, label='Anomalía')
            ax2.axvspan(anomaly_dist - anomaly_width / 2, anomaly_dist + anomaly_width / 2, alpha=0.3, color='red')
            ax2.set_title("Sección Geofísica 2D")
            ax2.set_xlabel("Distancia (m)")
            ax2.set_ylabel("Profundidad (m)")
            ax2.set_ylim(actual_max_d, 0)
            ax2.legend()
            st.pyplot(fig2)

        # --- Tab 3: Curva SP Vertical ---
        with tab3:
            st.subheader(f"Curva SP en la anomalía ({anomaly_dist:.1f} m)")
            fig3, ax3 = plt.subplots(figsize=(6, 9))
            ax3.plot(sp_plot, depth_plot, 'b-o', linewidth=2.5, markersize=3)
            ax3.fill_betweenx(depth_plot, sp_plot, sp_plot.mean(), color='cyan', alpha=0.20)
            ax3.set_title(f"Curva SP Vertical en {anomaly_dist:.1f} m")
            ax3.set_xlabel("Potencial (mV)")
            ax3.set_ylabel("Profundidad (m)")
            ax3.grid(True, alpha=0.3)

            x_margin = (sp_plot.max() - sp_plot.min()) * 0.15 if sp_plot.max() != sp_plot.min() else 0.01
            ax3.set_xlim(sp_plot.min() - x_margin, sp_plot.max() + x_margin)
            ax3.set_ylim(depth_plot.max() * 1.05, 0)

            rect_h     = depth_plot.max() * 0.12
            rect_y_top = max(0, aquifer_depth - rect_h / 2)
            x_range    = sp_plot.max() - sp_plot.min() if sp_plot.max() != sp_plot.min() else 1.0
            rect_w     = x_range * 0.80
            rect_x     = sp_plot.min() - x_margin * 0.5
            rect = plt.Rectangle(
                (rect_x, rect_y_top), rect_w, rect_h,
                fill=True, facecolor='yellow', edgecolor='red',
                linewidth=2, alpha=0.25, zorder=3
            )
            ax3.add_patch(rect)
            ax3.annotate(
                f"↓ Posible acuífero\n  ~{aquifer_depth:.0f} m",
                xy=(sp_plot[min_sp_idx], aquifer_depth),
                xytext=(sp_plot.mean(), aquifer_depth + depth_plot.max() * 0.08),
                fontsize=9, color='darkred',
                arrowprops=dict(arrowstyle='->', color='red')
            )
            st.pyplot(fig3)

        # ====================== INFORME PDF ======================
        if st.button("📄 Generar Informe PDF (1 hoja profesional)", type="primary", use_container_width=True):
            with st.spinner("Creando informe PDF..."):
                fig_pdf = plt.figure(figsize=(14, 11))
                gs = fig_pdf.add_gridspec(3, 1, height_ratios=[1.2, 2.2, 1.2], hspace=0.35)
                fig_pdf.suptitle(
                    f"INFORME TEFSM - {geology_type}\n"
                    f"Anomalía principal en {anomaly_dist:.1f} m | {datetime.now().strftime('%d/%m/%Y')}",
                    fontsize=16, fontweight='bold'
                )

                ax1p = fig_pdf.add_subplot(gs[0])
                for col in freq_cols:
                    ax1p.plot(df['Distance'], df[col], lw=1.2, alpha=0.75)
                ax1p.axvspan(anomaly_dist - anomaly_width / 2, anomaly_dist + anomaly_width / 2, alpha=0.3, color='red')
                ax1p.axvline(anomaly_dist, color='red', linestyle='--', linewidth=2)
                ax1p.set_title("Curvas de Frecuencia PQWT")
                ax1p.set_ylabel("Potencial (mV)")
                ax1p.grid(True, alpha=0.3)

                ax2p = fig_pdf.add_subplot(gs[1])
                im = ax2p.imshow(
                    vi,
                    extent=[0, line_length, actual_max_d, 0],
                    aspect='auto', cmap='jet', origin='upper',
                    vmin=np.percentile(vi, 2), vmax=np.percentile(vi, 98)
                )
                fig_pdf.colorbar(im, ax=ax2p, label='Potencial (mV)')
                ax2p.axvline(anomaly_dist, color='red', linewidth=3)
                ax2p.axvspan(anomaly_dist - anomaly_width / 2, anomaly_dist + anomaly_width / 2, alpha=0.3, color='red')
                ax2p.set_title("Sección Geofísica 2D")
                ax2p.set_xlabel("Distancia (m)")
                ax2p.set_ylabel("Profundidad (m)")
                ax2p.set_ylim(actual_max_d, 0)

                ax3p = fig_pdf.add_subplot(gs[2])
                ax3p.plot(sp_plot, depth_plot, 'b-o', linewidth=2.5, markersize=3)
                ax3p.fill_betweenx(depth_plot, sp_plot, sp_plot.mean(), color='cyan', alpha=0.20)
                ax3p.set_title(f"Curva SP Vertical en {anomaly_dist:.1f} m")
                ax3p.set_xlabel("Potencial (mV)")
                ax3p.set_ylabel("Profundidad (m)")
                ax3p.grid(True, alpha=0.3)
                x_margin = (sp_plot.max() - sp_plot.min()) * 0.15 if sp_plot.max() != sp_plot.min() else 0.01
                ax3p.set_xlim(sp_plot.min() - x_margin, sp_plot.max() + x_margin)
                ax3p.set_ylim(depth_plot.max() * 1.05, 0)
                rect_h     = depth_plot.max() * 0.12
                rect_y_top = max(0, aquifer_depth - rect_h / 2)
                x_range    = sp_plot.max() - sp_plot.min() if sp_plot.max() != sp_plot.min() else 1.0
                rect_w     = x_range * 0.80
                rect_x     = sp_plot.min() - x_margin * 0.5
                rect_p = plt.Rectangle(
                    (rect_x, rect_y_top), rect_w, rect_h,
                    fill=True, facecolor='yellow', edgecolor='red',
                    linewidth=2, alpha=0.25, zorder=3
                )
                ax3p.add_patch(rect_p)
                ax3p.annotate(
                    f"↓ Posible acuífero\n  ~{aquifer_depth:.0f} m",
                    xy=(sp_plot[min_sp_idx], aquifer_depth),
                    xytext=(sp_plot.mean(), aquifer_depth + depth_plot.max() * 0.08),
                    fontsize=8, color='darkred',
                    arrowprops=dict(arrowstyle='->', color='red')
                )

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

        st.info(f"🔬 {n_cols} canales detectados • Profundidad máxima calculada: {depth_levels.max():.0f} m")

else:
    st.warning("Por favor inicia sesión")

st.caption("TEFSM HydroPick Generalizado")