"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Analítica Descriptiva — Sector Real Colombiano 2021–2024                  ║
║  Trabajo de Grado | Maestría en Analítica e Inteligencia de Negocios       ║
║  Stack: Streamlit · Plotly · scikit-learn · pandas                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analítica Financiera Colombia 2021–2024",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta corporativa
# ── Paleta institucional Universidad del Valle ────────────────────────────
# Fuente: Manual de Identidad Visual Corporativa Univalle
CLR = {
    # Sistema cromático institucional
    "rojo":         "#C20E1A",  # Rojo Univalle — color principal de marca
    "granate":      "#841F1C",  # Granate oscuro — Z-Score / riesgo
    "gris_oscuro":  "#636363",  # Gris funcional — endeudamiento / texto
    "gris_medio":   "#555555",  # Gris medio — texto secundario (legible sobre blanco)
    "gris_claro":   "#CFCFCF",  # Gris claro — bordes, separadores
    "fondo":        "#F5F5F5",  # Fondo general — gris institucional muy claro
    "tarjeta":      "#FFFFFF",  # Tarjetas — blanco puro
    # Colores funcionales para datos (baja saturación)
    "verde":        "#3D7A3D",  # Verde institucional — rentabilidad positiva
    "verde_claro":  "#5A9A5A",  # Verde suave — valores positivos
    "amarillo":     "#C8581A",  # Naranja — advertencias / alertas
    "azul_claro":   "#3A6080",  # Azul grisáceo — clustering / datos neutros
    "azul_medio":   "#CFCFCF",  # Usado para bordes en dark; ahora borde gris claro
    "azul_oscuro":  "#1A1A1A",  # Texto principal — negro tipográfico
}

st.markdown(f"""
<style>
  /* ── SISTEMA GRÁFICO — UNIVERSIDAD DEL VALLE ──────────────────────────── */
  /* Manual de Identidad Visual Corporativa | MAIN – Maestría en Analítica  */

  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

  /* Base */
  .stApp {{
      background-color:{CLR['fondo']};
      font-family:Arial, 'Helvetica Neue', sans-serif;
  }}
  .block-container {{
      padding-top:0;
      padding-bottom:2rem;
      max-width:1400px;
  }}

  /* ── Sidebar institucional ──────────────────────────────────────────── */
  section[data-testid="stSidebar"] {{
      background-color:#FFFFFF;
      border-right:1px solid {CLR['gris_claro']};
  }}
  /* Solo texto del sidebar — NO usar * !important (bloquea todo) */
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] span:not([data-baseweb="tag"] span),
  section[data-testid="stSidebar"] div.stMarkdown {{
      color:{CLR['gris_oscuro']} !important;
  }}
  section[data-testid="stSidebar"] .stSelectbox > div > div,
  section[data-testid="stSidebar"] .stMultiSelect > div > div {{
      background:#FFFFFF !important;
      border:1px solid {CLR['rojo']} !important;
      border-radius:4px !important;
  }}

  /* ── FIX multiselect tags — globales (sidebar y tabs) ───────────────── */
  /* Los tags heredan primaryColor de Streamlit; se fuerza a institucional */
  [data-baseweb="tag"] {{
      background-color:{CLR['fondo']} !important;
      border:1px solid {CLR['rojo']} !important;
      border-radius:3px !important;
  }}
  [data-baseweb="tag"] span {{
      color:{CLR['gris_oscuro']} !important;
      font-size:.78rem !important;
  }}
  [data-baseweb="tag"] svg,
  [data-baseweb="tag"] button {{
      fill:{CLR['gris_oscuro']} !important;
      color:{CLR['gris_oscuro']} !important;
  }}
  /* Override del dropdown interno del multiselect */
  [data-baseweb="select"] > div {{
      background-color:#FFFFFF !important;
      border-color:{CLR['gris_claro']} !important;
  }}
  [data-baseweb="menu"] {{
      background:#FFFFFF !important;
  }}
  [data-baseweb="option"] {{
      color:{CLR['gris_oscuro']} !important;
      background:#FFFFFF !important;
  }}
  [data-baseweb="option"]:hover {{
      background:{CLR['fondo']} !important;
  }}

  /* ── Franja superior roja institucional (sistema gráfico Univalle) ──── */
  /* Aplicada mediante el encabezado dash-header */
  .dash-header {{
      background-color:#FFFFFF;
      padding:0;
      margin-bottom:1.4rem;
      border-bottom:1px solid {CLR['gris_claro']};
  }}
  .dash-header-stripe {{
      background-color:{CLR['rojo']};
      height:6px;
      width:100%;
  }}
  .dash-header-body {{
      padding:1.1rem 1.5rem 1rem;
      display:flex;
      align-items:flex-start;
      gap:1.4rem;
  }}
  .dash-header-logo {{
      flex-shrink:0;
  }}
  .dash-header-logo-mark {{
      width:36px;height:36px;
      background:{CLR['rojo']};
      display:flex;align-items:center;justify-content:center;
      position:relative;
  }}
  .dash-header-logo-mark::after {{
      content:'';
      position:absolute;
      bottom:6px;left:50%;
      transform:translateX(-50%);
      border-left:8px solid transparent;
      border-right:8px solid transparent;
      border-top:10px solid #FFFFFF;
  }}
  .dash-header-texts {{flex:1;}}
  .dash-header-pretitle {{
      font-size:.72rem;
      color:{CLR['rojo']};
      font-family:'Montserrat',Arial,sans-serif;
      font-weight:600;
      letter-spacing:.08em;
      text-transform:uppercase;
      margin-bottom:.25rem;
  }}
  .dash-header h1 {{
      font-family:'Montserrat',Arial,sans-serif;
      font-size:1.35rem;
      font-weight:700;
      color:{CLR['gris_oscuro']};
      margin:0 0 .2rem;
      line-height:1.3;
  }}
  .dash-header h1 span.titulo-rojo {{color:{CLR['rojo']};}}
  .dash-header p {{
      font-size:.8rem;
      color:#444444;
      margin:0;
      line-height:1.5;
  }}
  .dash-header-meta {{
      display:flex;
      gap:1.5rem;
      margin-top:.5rem;
      padding-top:.5rem;
      border-top:1px solid {CLR['gris_claro']};
  }}
  .dash-header-meta-item {{
      font-size:.72rem;
      color:{CLR['gris_oscuro']};
  }}
  .dash-header-meta-item span {{
      color:{CLR['rojo']};
      font-weight:600;
  }}

  /* ── Tarjetas KPI ───────────────────────────────────────────────────── */
  .kpi-card {{
      background:#FFFFFF;
      border:1px solid {CLR['gris_claro']};
      border-radius:6px;
      padding:1rem 1.1rem;
      text-align:left;
      box-shadow:0 1px 3px rgba(0,0,0,.06);
  }}
  .kpi-title {{
      font-size:.72rem;
      color:#444444;
      font-family:'Montserrat',Arial,sans-serif;
      font-weight:600;
      letter-spacing:.04em;
      text-transform:uppercase;
      margin-bottom:.4rem;
  }}
  .kpi-value {{
      font-size:1.9rem;
      font-weight:700;
      color:{CLR['gris_oscuro']};
      font-family:'Montserrat',Arial,sans-serif;
      line-height:1;
  }}
  .kpi-sub {{
      font-size:.72rem;
      color:#555555;
      margin-top:.3rem;
  }}
  /* Indicadores de estado — borde izquierdo fino */
  .kpi-safe   {{border-left:3px solid {CLR['verde']} !important;}}
  .kpi-warn   {{border-left:3px solid {CLR['amarillo']} !important;}}
  .kpi-danger {{border-left:3px solid {CLR['rojo']} !important;}}
  .kpi-info   {{border-left:3px solid {CLR['gris_oscuro']} !important;}}

  /* ── Encabezados de sección ─────────────────────────────────────────── */
  .section-header {{
      font-family:'Montserrat',Arial,sans-serif;
      font-size:.85rem;
      font-weight:600;
      color:#222222;
      border-bottom:2px solid {CLR['rojo']};
      padding-bottom:.35rem;
      margin-bottom:.9rem;
      letter-spacing:.01em;
  }}

  /* Separador de sección (línea roja delgada) */
  .sep-rojo {{
      border:none;
      border-top:1px solid {CLR['rojo']};
      margin:1.5rem 0;
      opacity:.35;
  }}

  /* ── Caja de interpretación / insight ─────────────────────────────── */
  .interp-box {{
      background:#FFFFFF;
      border-radius:4px;
      padding:.7rem 1rem;
      border-left:3px solid {CLR['gris_claro']};
      margin:.5rem 0;
      font-size:.8rem;
      color:{CLR['gris_oscuro']};
      line-height:1.65;
      box-shadow:0 1px 2px rgba(0,0,0,.04);
  }}
  .interp-box b {{color:{CLR['gris_oscuro']};}}

  /* ── Pestañas ───────────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {{
      background:#FFFFFF;
      border-bottom:2px solid {CLR['rojo']};
      border-radius:0;
      padding:0;
      gap:0;
  }}
  .stTabs [data-baseweb="tab"] {{
      font-family:'Montserrat',Arial,sans-serif;
      font-size:.8rem;
      font-weight:600;
      color:#841F1C !important;
      border-radius:0;
      padding:.55rem 1.1rem;
      border-bottom:2px solid transparent;
      margin-bottom:-2px;
      background:transparent !important;
      opacity:1 !important;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
      opacity:.85 !important;
      background:transparent !important;
  }}
  .stTabs [aria-selected="true"] {{
      color:{CLR['rojo']} !important;
      font-weight:700 !important;
      border-bottom:2px solid {CLR['rojo']} !important;
      background:transparent !important;
      opacity:1 !important;                  /* activa: rojo completo */
  }}
  /* Fondo del contenido de las pestañas */
  .stTabs [data-baseweb="tab-panel"] {{
      background:{CLR['fondo']};
      padding-top:1rem;
  }}

  /* ── Metrica override ───────────────────────────────────────────────── */
  [data-testid="metric-container"] {{
      background:#FFFFFF;
      border-radius:6px;
      padding:.6rem;
      border:1px solid {CLR['gris_claro']};
      box-shadow:0 1px 2px rgba(0,0,0,.04);
  }}

  /* ── Scrollbar discreta ─────────────────────────────────────────────── */
  ::-webkit-scrollbar {{width:4px;}}
  ::-webkit-scrollbar-thumb {{background:{CLR['gris_claro']};border-radius:2px;}}

  /* ── Ficha metodológica de indicador ───────────────────────────────── */
  .ficha-ind {{
      background:#FFFFFF;
      border-radius:6px;
      border:1px solid {CLR['gris_claro']};
      padding:0;
      overflow:hidden;
      margin:.6rem 0;
      box-shadow:0 1px 3px rgba(0,0,0,.05);
  }}
  .ficha-top {{
      background:{CLR['fondo']};
      border-bottom:2px solid {CLR['rojo']};
      padding:.6rem 1rem;
      display:flex;
      align-items:center;
      gap:.5rem;
  }}
  .ficha-top-icon {{display:none;}}  /* Eliminar iconos decorativos */
  .ficha-top-title {{
      color:{CLR['gris_oscuro']};
      font-weight:600;
      font-size:.88rem;
      font-family:'Montserrat',Arial,sans-serif;
  }}
  .ficha-top-tag {{
      margin-left:auto;
      font-size:.65rem;
      padding:2px 8px;
      border-radius:2px;
      background:{CLR['rojo']};
      color:#FFFFFF;
      font-weight:600;
      letter-spacing:.04em;
      text-transform:uppercase;
      white-space:nowrap;
  }}
  .ficha-body {{padding:.8rem 1rem;}}
  .ficha-simple {{
      background:{CLR['fondo']};
      border-left:2px solid {CLR['gris_oscuro']};
      border-radius:0 4px 4px 0;
      padding:.6rem .9rem;
      margin-bottom:.65rem;
      font-size:.8rem;
      color:{CLR['gris_oscuro']};
      line-height:1.65;
  }}
  .ficha-simple .etiqueta {{
      font-size:.65rem;
      font-weight:600;
      color:{CLR['gris_oscuro']};
      text-transform:uppercase;
      letter-spacing:.05em;
      margin-bottom:.2rem;
      font-family:'Montserrat',Arial,sans-serif;
  }}
  .ficha-tecnico {{
      background:#FFFFFF;
      border-left:2px solid {CLR['gris_claro']};
      border-radius:0 4px 4px 0;
      padding:.6rem .9rem;
      margin-bottom:.65rem;
      font-size:.78rem;
      color:{CLR['gris_oscuro']};
      line-height:1.65;
  }}
  .ficha-tecnico .etiqueta {{
      font-size:.65rem;
      font-weight:600;
      color:#555555;
      text-transform:uppercase;
      letter-spacing:.05em;
      margin-bottom:.2rem;
      font-family:'Montserrat',Arial,sans-serif;
  }}
  .ficha-formula {{
      font-family:'Courier New',monospace;
      font-size:.78rem;
      background:{CLR['fondo']};
      color:{CLR['gris_oscuro']};
      border:1px solid {CLR['gris_claro']};
      padding:.4rem .8rem;
      border-radius:3px;
      margin:.4rem 0 .65rem;
      display:inline-block;
      width:100%;
  }}
  .ficha-semaforo {{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.2rem;}}
  .nivel {{
      padding:.22rem .7rem;
      border-radius:2px;
      font-size:.7rem;
      font-weight:600;
      white-space:nowrap;
      font-family:'Montserrat',Arial,sans-serif;
  }}
  .nivel-verde   {{background:rgba(61,122,61,.08);color:{CLR['verde']};
                   border:1px solid {CLR['verde']};}}
  .nivel-amarillo{{background:rgba(200,88,26,.08);color:{CLR['amarillo']};
                   border:1px solid {CLR['amarillo']};}}
  .nivel-rojo    {{background:rgba(194,14,26,.08);color:{CLR['rojo']};
                   border:1px solid {CLR['rojo']};}}
  .nivel-gris    {{background:{CLR['fondo']};color:{CLR['gris_medio']};
                   border:1px solid {CLR['gris_claro']};}}

  /* ── Glosario ───────────────────────────────────────────────────────── */
  .glos-grid {{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin:.6rem 0;}}

  /* ── Expander institucional ─────────────────────────────────────────── */
  .streamlit-expanderHeader {{
      background:#FFFFFF !important;
      border-radius:4px !important;
      color:{CLR['gris_oscuro']} !important;
      font-size:.8rem !important;
      font-family:'Montserrat',Arial,sans-serif !important;
      font-weight:600 !important;
      border:1px solid {CLR['gris_claro']} !important;
  }}
  .streamlit-expanderContent {{
      background:#FFFFFF;
      border-radius:0 0 4px 4px;
      border:1px solid {CLR['gris_claro']};
      border-top:0;
      padding:.7rem .9rem;
  }}

  /* ── Botón principal ────────────────────────────────────────────────── */
  .stButton > button[kind="primary"] {{
      background-color:{CLR['rojo']} !important;
      color:#FFFFFF !important;
      border:none !important;
      border-radius:3px !important;
      font-family:'Montserrat',Arial,sans-serif !important;
      font-weight:600 !important;
      font-size:.82rem !important;
      padding:.5rem 1.2rem !important;
  }}
  .stButton > button[kind="secondary"],
  .stButton > button:not([kind]) {{
      background-color:#FFFFFF !important;
      color:{CLR['gris_oscuro']} !important;
      border:1px solid {CLR['gris_claro']} !important;
      border-radius:3px !important;
      font-family:'Montserrat',Arial,sans-serif !important;
      font-size:.82rem !important;
  }}

  /* ── Dataframe / table ──────────────────────────────────────────────── */
  .stDataFrame {{border-radius:4px !important;}}
  [data-testid="stDataFrame"] th {{
      background-color:{CLR['fondo']} !important;
      color:{CLR['gris_oscuro']} !important;
      font-family:'Montserrat',Arial,sans-serif !important;
      font-weight:600 !important;
      font-size:.78rem !important;
      border-bottom:2px solid {CLR['rojo']} !important;
  }}
  [data-testid="stDataFrame"] td {{
      font-size:.78rem !important;
      color:{CLR['gris_oscuro']} !important;
  }}

  /* ── Number input ───────────────────────────────────────────────────── */
  .stNumberInput > div > div > input {{
      border:1px solid {CLR['gris_claro']} !important;
      border-radius:3px !important;
      font-size:.82rem !important;
      background:#FFFFFF !important;
      color:{CLR['gris_oscuro']} !important;
  }}
  .stTextInput > div > div > input {{
      border:1px solid {CLR['gris_claro']} !important;
      border-radius:3px !important;
      font-size:.82rem !important;
      color:{CLR['gris_oscuro']} !important;
  }}
  .stSelectbox > div > div {{
      border:1px solid {CLR['gris_claro']} !important;
      border-radius:3px !important;
  }}

  /* ── FIX texto visible en selectbox sidebar (Año de análisis) ───────── */
  /* Streamlit hereda colores del tema en el valor seleccionado del select  */
  [data-baseweb="select"] [data-baseweb="value"],
  [data-baseweb="select"] [data-baseweb="value"] span,
  [data-baseweb="select"] [data-baseweb="input"] input,
  [data-baseweb="select"] > div > div > div,
  [data-baseweb="select"] > div > div span {{
      color:{CLR['gris_oscuro']} !important;
      font-size:.82rem !important;
  }}
  /* Placeholder text */
  [data-baseweb="select"] [data-baseweb="input"] input::placeholder {{
      color:#999999 !important;
  }}
  /* Dropdown options list */
  [data-baseweb="popover"] [data-baseweb="menu"] li {{
      color:{CLR['gris_oscuro']} !important;
      font-size:.82rem !important;
      background:#FFFFFF !important;
  }}
  [data-baseweb="popover"] [data-baseweb="menu"] li:hover,
  [data-baseweb="popover"] [data-baseweb="menu"] [aria-selected="true"] {{
      background:{CLR['fondo']} !important;
      color:{CLR['rojo']} !important;
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS Y MODELOS (con caché)
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

SECTOR_CORTO = {
    "A – Agricultura y ganadería":                       "A – Agricultura",
    "B – Explotación de minas y canteras":               "B – Minería",
    "C – Industria manufacturera":                       "C – Manufactura",
    "D – Electricidad, gas, vapor":                      "D – Electricidad",
    "F – Construcción":                                  "F – Construcción",
    "G – Comercio al por mayor y menor":                 "G – Comercio",
    "H – Transporte y almacenamiento":                   "H – Transporte",
    "I – Alojamiento y gastronomía":                     "I – Alojamiento",
    "J – Información y comunicaciones":                  "J – Información",
    "K – Actividades financieras y de seguros":          "K – Financiero",
    "L – Actividades inmobiliarias":                     "L – Inmobiliario",
    "M – Actividades profesionales y científicas":       "M – Profesionales",
    "N – Actividades de servicios administrativos":      "N – Administrativos",
    "P – Educación":                                     "P – Educación",
    "Q – Salud y servicios sociales":                    "Q – Salud",
    "R – Arte, entretenimiento y recreación":            "R – Arte/Recreación",
    "S – Otras actividades de servicios":                "S – Otros servicios",
    "Sin clasificar":                                    "Sin clasificar",
}

@st.cache_data(show_spinner="⏳ Cargando dataset financiero…")
def load_data():
    df = pd.read_csv(os.path.join(BASE, "mi_dataset_final.csv"), low_memory=False)

    # ── NORMALIZACIÓN DE NOMBRES DE COLUMNA ──────────────────────────────────
    # El cuaderno reconstruido exporta el CSV con nombres limpios (sin tildes,
    # sin espacios). El dashboard necesita los nombres estandarizados.
    # Este bloque detecta qué versión del CSV está cargada y adapta en consecuencia.

    RENOMBRE_INVERSO = {
        # Columnas de identificación
        "Compania":           "Compañía",
        "Seccion_CIIU":       "Sección CIIU",
        "Clasificacion_Tamaño": "Clasificación_957",
        # Liquidez
        "Razon_Liquidez":     "Razón De Liquidez (x)",
        "Prueba_Acida":       "Prueba Ácida (x)",
        "Razon_Efectivo":     "Razón De Efectivo (x)",
        # Endeudamiento
        "Deuda_Activos_pct":  "Deuda / Activos (%)",
        "Deuda_Capital_pct":  "Relación Deuda/Capital (%)",
        "Deuda_Neta_EBITDA":  "Deuda Neta / EBITDA (x)",
        "ZScore_Altman":      "Modelo Z-Score de Altman",
        # Rentabilidad
        "ROA_pct":            "Rendimiento Sobre Los Activos (ROA) (%)",
        "ROE_pct":            "Rendimiento Sobre El Patrimonio (ROE) (%)",
        "Margen_Operacional_pct": "Margen Operacional (%)",
        "Margen_Neto_pct":    "Margen Neto (%)",
        "Margen_EBITDA_pct":  "Margen Ebitda (%)",
        # Financieras base
        "Activos_Totales":    "Activos Totales",
        "Pasivos_Totales":    "Pasivos Totales",
        "Patrimonio_Total":   "Total de patrimonio",
        "Ingresos_Operativos":"Total Ingreso Operativo",
        "Utilidad_Neta":      "Ganancia (Pérdida) Neta",
        "EBIT":               "Ganancia operativa (EBIT)",
        "Capital_Trabajo":    "Capital De Trabajo",
    }

    # Renombrar solo las columnas que existen en el CSV con el nombre nuevo
    cols_a_renombrar = {k: v for k, v in RENOMBRE_INVERSO.items() if k in df.columns}
    if cols_a_renombrar:
        df = df.rename(columns=cols_a_renombrar)

    # Para métricas financieras con año embebido (ej. Razon_Liquidez 2024
    # → Razón De Liquidez (x) 2024), el rename por año se hace genérico:
    # El CSV del cuaderno reconstruido guarda los nombres limpios + año separado
    # porque los genera desde el panel de datos. No requiere renombre adicional
    # ya que las columnas métricas llevan el nombre base sin año como prefijo.

    # ── COLUMNA Sección CIIU ─────────────────────────────────────────────────
    # Determinar qué columna de sector existe después del posible renombre
    col_sector = None
    for candidato in ["Sección CIIU", "Seccion_CIIU", "Sector_CIIU"]:
        if candidato in df.columns:
            col_sector = candidato
            break

    if col_sector is None:
        # Último recurso: buscar cualquier columna que contenga "CIIU" o "sector"
        candidatos = [c for c in df.columns if "CIIU" in c or "ector" in c.lower()]
        if candidatos:
            col_sector = candidatos[0]

    if col_sector:
        # Estandarizar al nombre que usa el resto del dashboard
        if col_sector != "Sección CIIU":
            df = df.rename(columns={col_sector: "Sección CIIU"})
        df["Sector_Corto"] = df["Sección CIIU"].map(SECTOR_CORTO).fillna(df["Sección CIIU"])
    else:
        df["Sección CIIU"] = "Sin clasificar"
        df["Sector_Corto"] = "Sin clasificar"

    # ── COLUMNA Clasificación_957 ────────────────────────────────────────────
    for candidato in ["Clasificación_957", "Clasificacion_Tamaño",
                      "Clasificacion_Tamano", "Tamaño"]:
        if candidato in df.columns and candidato != "Clasificación_957":
            df = df.rename(columns={candidato: "Clasificación_957"})
            break

    # ── COLUMNA Compañía ─────────────────────────────────────────────────────
    for candidato in ["Compañía", "Compania", "Empresa", "Company"]:
        if candidato in df.columns and candidato != "Compañía":
            df = df.rename(columns={candidato: "Compañía"})
            break

    # ── RECONSTRUIR columnas de métricas con año si vienen en formato panel ──
    # El CSV del cuaderno reconstruido puede traer la columna "Año" y las
    # métricas sin año. En ese caso hay que pivotear a formato wide.
    if "Año" in df.columns and "Compañía" in df.columns:
        años_csv = df["Año"].dropna().unique()
        # Solo pivotar si las columnas métricas NO tienen año embebido
        cols_sin_año = [c for c in df.columns
                        if not any(str(a) in c for a in [2021, 2022, 2023, 2024])
                        and c not in ["Compañía","Sección CIIU","Clasificación_957",
                                      "Año","Sector_Corto","Zona_ZScore",
                                      "Cluster_Label","Movimiento_ZScore"]]
        if cols_sin_año and len(años_csv) > 1:
            # El CSV ya está en formato panel (long) — pivotar a wide
            id_cols  = ["Compañía","Sección CIIU","Clasificación_957"]
            id_cols  = [c for c in id_cols if c in df.columns]
            val_cols = [c for c in cols_sin_año if c in df.columns]
            try:
                df_wide  = df[id_cols + ["Año"] + val_cols].copy()
                df_pivot = df_wide.pivot_table(
                    index=id_cols, columns="Año", values=val_cols, aggfunc="first"
                )
                df_pivot.columns = [f"{col} {int(año)}" for col, año in df_pivot.columns]
                df_pivot = df_pivot.reset_index()
                # Recuperar Sector_Corto
                df_pivot["Sector_Corto"] = (
                    df_pivot["Sección CIIU"].map(SECTOR_CORTO).fillna(df_pivot["Sección CIIU"])
                    if "Sección CIIU" in df_pivot.columns else "Sin clasificar"
                )
                df = df_pivot
            except Exception:
                pass  # Si el pivot falla, usar el DataFrame tal como está

    # ── CLASIFICACIÓN Z-SCORE POR AÑO ────────────────────────────────────────
    for año in [2021, 2022, 2023, 2024]:
        col_z  = f"Modelo Z-Score de Altman {año}"
        col_zc = f"ZScore_Altman {año}"  # nombre alternativo del CSV reconstruido
        col_cl = f"Zona_ZScore_{año}"

        # Usar cualquier columna de Z-Score disponible para ese año
        col_zscore_real = None
        for candidato in [col_z, col_zc, f"ZScore_Calculado_{año}"]:
            if candidato in df.columns:
                col_zscore_real = candidato
                break

        if col_zscore_real and col_cl not in df.columns:
            df[col_cl] = pd.cut(
                pd.to_numeric(df[col_zscore_real], errors="coerce"),
                bins=[-np.inf, 1.10, 2.60, np.inf],
                labels=["🔴 Zona Quiebra", "🟡 Zona Alerta", "🟢 Zona Segura"],
            ).astype(str)

    # ── MATRIZ DE TRANSICIÓN 2021 → 2024 ─────────────────────────────────────
    if "Zona_ZScore_2021" in df.columns and "Zona_ZScore_2024" in df.columns:
        def calcular_movimiento(r):
            z21 = str(r.get("Zona_ZScore_2021", ""))
            z24 = str(r.get("Zona_ZScore_2024", ""))
            if z21 in ("nan", "<NA>", "") or z24 in ("nan", "<NA>", ""):
                return "Sin dato"
            if (z21 == "🔴 Zona Quiebra" and z24 in ["🟡 Zona Alerta", "🟢 Zona Segura"]) or \
               (z21 == "🟡 Zona Alerta"  and z24 == "🟢 Zona Segura"):
                return "Mejoró"
            if (z21 == "🟢 Zona Segura"  and z24 in ["🟡 Zona Alerta", "🔴 Zona Quiebra"]) or \
               (z21 == "🟡 Zona Alerta"  and z24 == "🔴 Zona Quiebra"):
                return "Empeoró"
            return "Permaneció"

        df["Movimiento_ZScore"] = df.apply(calcular_movimiento, axis=1)

    return df

@st.cache_resource(show_spinner="⏳ Cargando modelos ML…")
def load_models():
    m = {}
    for name in ["random_forest_zscore", "arbol_decision_zscore",
                 "scaler_financiero", "pca_2d", "kmeans_k3",
                 "label_encoder_zscore", "features_modelo"]:
        path = os.path.join(BASE, f"{name}.pkl")
        if os.path.exists(path):
            m[name] = joblib.load(path)
    return m

df_raw  = load_data()
modelos = load_models()

AÑOS = [2021, 2022, 2023, 2024]

# ─────────────────────────────────────────────────────────────────────────────
# 2. SIDEBAR — FILTROS GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Encabezado institucional del sidebar
    st.markdown(f"""
    <div style='background:{CLR["rojo"]};height:4px;margin:-1rem -1rem 1rem;'></div>
    <div style='padding:.2rem 0 .8rem;'>
      <div style='font-family:Montserrat,Arial,sans-serif;font-weight:700;
                  font-size:.95rem;color:{CLR["gris_oscuro"]};'>
        Filtros de análisis
      </div>
      <div style='font-size:.72rem;color:#555555;margin-top:.15rem;'>
        Seleccione los parámetros para explorar los datos
      </div>
    </div>
    <hr style='border:none;border-top:1px solid {CLR["gris_claro"]};margin:.2rem 0 .8rem;'>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-family:Montserrat,Arial,sans-serif;font-size:.75rem;"
                f"font-weight:600;color:{CLR['gris_oscuro']};margin-bottom:.3rem;'>"
                f"Año de análisis</div>", unsafe_allow_html=True)
    año_sel = st.selectbox("", AÑOS, index=3, label_visibility="collapsed")

    st.markdown(f"<div style='font-family:Montserrat,Arial,sans-serif;font-size:.75rem;"
                f"font-weight:600;color:{CLR['gris_oscuro']};margin-bottom:.3rem;margin-top:.7rem;'>"
                f"Sector económico (CIIU)</div>", unsafe_allow_html=True)
    sectores_disp = ["Todos"] + sorted(df_raw["Sector_Corto"].unique().tolist())
    sector_sel    = st.multiselect("", sectores_disp, default=["Todos"],
                                   label_visibility="collapsed")

    st.markdown(f"<div style='font-family:Montserrat,Arial,sans-serif;font-size:.75rem;"
                f"font-weight:600;color:{CLR['gris_oscuro']};margin-bottom:.3rem;margin-top:.7rem;'>"
                f"Tamaño empresarial</div>", unsafe_allow_html=True)
    tamaños_disp = df_raw["Clasificación_957"].dropna().unique().tolist()
    tamaño_sel   = st.multiselect("", tamaños_disp, default=tamaños_disp,
                                  label_visibility="collapsed")

    st.markdown(f"<div style='font-family:Montserrat,Arial,sans-serif;font-size:.75rem;"
                f"font-weight:600;color:{CLR['gris_oscuro']};margin-bottom:.3rem;margin-top:.7rem;'>"
                f"Clasificación Z-Score de Altman</div>", unsafe_allow_html=True)
    zonas_disp = ["🟢 Zona Segura", "🟡 Zona Alerta", "🔴 Zona Quiebra"]
    zona_sel   = st.multiselect("", zonas_disp, default=zonas_disp,
                                label_visibility="collapsed")

    st.markdown(f"""
    <hr style='border:none;border-top:1px solid {CLR["gris_claro"]};margin:1rem 0 .8rem;'>
    <div style='font-size:.7rem;color:#444444;line-height:1.75;'>
      <div style='font-family:Montserrat,Arial,sans-serif;font-weight:700;
                  color:{CLR["rojo"]};margin-bottom:.3rem;font-size:.72rem;'>
        Universidad del Valle
      </div>
      Maestría en Analítica e Inteligencia de Negocios<br>
      Trabajo de grado — Analítica descriptiva del desempeño financiero<br>
      del sector real en Colombia (2021–2024)<br><br>
      <b>Fuente de datos:</b> EMIS Platform<br>
      <b>Universo:</b> 1.739 empresas | 18 sectores CIIU<br>
      <b>Período:</b> 2021–2024<br>
      <b>Modelo de riesgo:</b> Z-Score Altman Z&#8242;&#8242; (1995)
    </div>
    """, unsafe_allow_html=True)

# Filtrar dataframe según sidebar
def filtrar(df, año):
    d = df.copy()
    col_z = f"Zona_ZScore_{año}"
    if "Todos" not in sector_sel:
        d = d[d["Sector_Corto"].isin(sector_sel)]
    if tamaño_sel:
        d = d[d["Clasificación_957"].isin(tamaño_sel)]
    if zona_sel and col_z in d.columns:
        d = d[d[col_z].isin(zona_sel)]
    return d

df = filtrar(df_raw, año_sel)

# ─────────────────────────────────────────────────────────────────────────────
# 3. CABECERA DEL DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='dash-header'>
  <div class='dash-header-stripe'></div>
  <div class='dash-header-body'>
    <div class='dash-header-logo'>
      <svg width='42' height='48' viewBox='0 0 42 48' fill='none' xmlns='http://www.w3.org/2000/svg'>
        <rect width='42' height='42' fill='#C20E1A'/>
        <rect x='21' y='0' width='21' height='21' fill='#FFFFFF'/>
        <polygon points='7,18 35,18 21,36' fill='#FFFFFF'/>
      </svg>
    </div>
    <div class='dash-header-texts'>
      <div class='dash-header-pretitle'>Universidad del Valle &nbsp;·&nbsp; MAIN — Maestría en Analítica e Inteligencia de Negocios</div>
      <h1>Analítica descriptiva del desempeño financiero del <span class='titulo-rojo'>sector real colombiano</span></h1>
      <p>Prototipo de herramienta de visualización · Principales empresas por ingresos operativos · Período 2021–2024</p>
      <div class='dash-header-meta'>
        <div class='dash-header-meta-item'>Empresas analizadas: <span>{len(df):,}</span></div>
        <div class='dash-header-meta-item'>Año seleccionado: <span>{año_sel}</span></div>
        <div class='dash-header-meta-item'>Fuente: <span>EMIS Platform</span></div>
        <div class='dash-header-meta-item'>Modelo de riesgo: <span>Z-Score Altman Z&#8242;&#8242;</span></div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: figuras con tema oscuro
# ─────────────────────────────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font_color=CLR["gris_oscuro"],
    font_family="Arial, 'Helvetica Neue', sans-serif",
    margin=dict(t=36, b=36, l=10, r=10),
    legend=dict(
        bgcolor="#FFFFFF",
        bordercolor=CLR["gris_claro"],
        borderwidth=1,
        font_size=11,
        font_color=CLR["gris_oscuro"],
    ),
    xaxis=dict(
        gridcolor="#F0F0F0",
        zerolinecolor=CLR["gris_claro"],
        linecolor=CLR["gris_claro"],
        tickfont=dict(color=CLR["gris_medio"], size=11),
    ),
    yaxis=dict(
        gridcolor="#F0F0F0",
        zerolinecolor=CLR["gris_claro"],
        linecolor=CLR["gris_claro"],
        tickfont=dict(color=CLR["gris_medio"], size=11),
    ),
)

# Paleta de datos institucional: baja saturación, legible sobre fondo blanco
COLOR_SEQ = [
    CLR["rojo"],       # Rojo Univalle — primer dato
    CLR["gris_oscuro"],# Gris oscuro — segundo dato
    CLR["azul_claro"], # Azul grisáceo — tercer dato
    CLR["verde"],      # Verde institucional — cuarto dato
    CLR["granate"],    # Granate — quinto dato
    "#5C7A8A",         # Azul pizarra
    CLR["amarillo"],   # Naranja
    "#7A6A5A",         # Café institucional
    "#4A7A6A",         # Verde oscuro
    "#5A4A7A",         # Violeta apagado
    "#7A4A4A",         # Granate suave
    "#4A6A4A",         # Verde bosque
]

def apply_layout(fig, title="", height=360):
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=title,
            font_size=12,
            font_color=CLR["gris_oscuro"],
            font_family="Montserrat, Arial, sans-serif",
            x=0,
        ),
        height=height,
    )
    return fig


def kpi_html(titulo, valor, sub="", clase="kpi-info"):
    # Limpiar emojis del título para estilo institucional
    titulo_limpio = titulo
    for em in ["📊","📈","📐","💼","💧","🏢","🟢","🟡","🔴","⚖️","⚡","🧪","💵",
               "🔗","🏦","🔄","🤖","📋","📦","📑","🏷️","💹","💰","⚠️","🚨"]:
        titulo_limpio = titulo_limpio.replace(em, "").strip()
    return f"""
    <div class='kpi-card {clase}'>
      <div class='kpi-title'>{titulo_limpio}</div>
      <div class='kpi-value'>{valor}</div>
      <div class='kpi-sub'>{sub}</div>
    </div>"""


def zona_color(zona_str):
    if "Segura"  in str(zona_str): return CLR["verde"]
    if "Alerta"  in str(zona_str): return CLR["amarillo"]
    if "Quiebra" in str(zona_str): return CLR["rojo"]
    return CLR["gris_medio"]


def zona_bg(zona_str):
    if "Segura"  in str(zona_str): return "rgba(61,122,61,.10)"
    if "Alerta"  in str(zona_str): return "rgba(200,88,26,.10)"
    if "Quiebra" in str(zona_str): return "rgba(194,14,26,.10)"
    return CLR["fondo"]


def interp(texto):
    # Limpiar emojis de métricas del texto de interpretación (conservar símbolos de zona)
    st.markdown(f"<div class='interp-box'>{texto}</div>", unsafe_allow_html=True)


def ficha_as_html(icono, nombre, tag, simple, analogia,
                  formula, niveles, tecnico="", **kwargs):
    """
    Retorna el HTML de una ficha como STRING puro — sin llamar a st.markdown().
    Usar en el glosario para agrupar fichas en una sola llamada st.markdown()
    y evitar el error React 'removeChild' que ocurre con múltiples
    st.markdown() dentro de st.columns() en un loop con re-renders.
    """
    c_rojo    = CLR["rojo"]
    c_gris_os = CLR["gris_oscuro"]
    c_gris_md = CLR["gris_medio"]
    c_gris_cl = CLR["gris_claro"]
    c_fondo   = CLR["fondo"]
    c_verde   = CLR["verde"]
    c_amari   = CLR["amarillo"]

    def nivel_badge(c, t):
        paleta = {
            "verde":    (c_verde,  "rgba(61,122,61,.09)"),
            "amarillo": (c_amari,  "rgba(200,88,26,.09)"),
            "rojo":     (c_rojo,   "rgba(194,14,26,.09)"),
        }
        text_c, bg = paleta.get(c, (c_gris_md, c_fondo))
        s = (f"display:inline-block;padding:.22rem .75rem;border-radius:2px;"
             f"font-size:.7rem;font-weight:600;white-space:nowrap;"
             f"font-family:Montserrat,Arial,sans-serif;"
             f"background:{bg};color:{text_c};border:1px solid {text_c};"
             f"margin:.1rem .1rem .1rem 0;")
        return f"<span style='{s}'>{t}</span>"

    niveles_html = "".join(nivel_badge(c, t) for c, t in niveles)

    tecnico_bloque = ""
    if tecnico:
        tecnico_bloque = (
            f"<div style='border-left:2px solid {c_gris_cl};padding:.55rem .85rem;"
            f"margin-bottom:.6rem;font-size:.78rem;color:{c_gris_os};line-height:1.65;'>"
            f"<div style='font-size:.63rem;font-weight:700;color:{c_gris_md};"
            f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:.22rem;"
            f"font-family:Montserrat,Arial,sans-serif;'>Definición técnica</div>"
            f"{tecnico}</div>"
        )

    return (
        f"<div style='background:#FFFFFF;border-radius:6px;"
        f"border:1px solid {c_gris_cl};overflow:hidden;margin:.55rem 0;"
        f"box-shadow:0 1px 3px rgba(0,0,0,.05);'>"
        f"<div style='background:{c_fondo};border-bottom:2px solid {c_rojo};"
        f"padding:.55rem .95rem;display:flex;align-items:center;gap:.5rem;'>"
        f"<span style='color:{c_gris_os};font-weight:700;font-size:.86rem;"
        f"font-family:Montserrat,Arial,sans-serif;flex:1;'>{nombre}</span>"
        f"<span style='font-size:.63rem;padding:.18rem .65rem;border-radius:2px;"
        f"background:{c_rojo};color:#FFF;font-weight:700;"
        f"letter-spacing:.05em;text-transform:uppercase;"
        f"font-family:Montserrat,Arial,sans-serif;white-space:nowrap;'>{tag}</span>"
        f"</div>"
        f"<div style='padding:.75rem .95rem;'>"
        f"<div style='background:{c_fondo};border-left:2px solid {c_gris_os};"
        f"padding:.55rem .85rem;margin-bottom:.6rem;"
        f"font-size:.79rem;color:{c_gris_os};line-height:1.65;'>"
        f"<div style='font-size:.63rem;font-weight:700;color:{c_gris_os};"
        f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:.2rem;"
        f"font-family:Montserrat,Arial,sans-serif;'>Descripción accesible</div>"
        f"{simple}"
        f"<div style='margin-top:.4rem;font-style:italic;color:{c_gris_md};"
        f"font-size:.78rem;'>{analogia}</div>"
        f"</div>"
        + tecnico_bloque
        + f"<div style='font-size:.63rem;font-weight:700;color:{c_gris_os};"
        f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:.24rem;"
        f"font-family:Montserrat,Arial,sans-serif;'>Fórmula</div>"
        f"<div style='font-family:Courier New,monospace;font-size:.78rem;"
        f"background:{c_fondo};color:{c_gris_os};"
        f"border:1px solid {c_gris_cl};padding:.38rem .75rem;"
        f"border-radius:3px;margin:.1rem 0 .6rem;'>"
        f"{formula}</div>"
        f"<div style='font-size:.63rem;font-weight:700;color:{c_gris_os};"
        f"text-transform:uppercase;letter-spacing:.06em;margin:.35rem 0 .25rem;"
        f"font-family:Montserrat,Arial,sans-serif;'>Interpretación de resultados</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:.15rem;margin-top:.1rem;'>"
        f"{niveles_html}"
        f"</div>"
        f"</div></div>"
    )


def ficha_indicador(icono, nombre, tag, simple, analogia,
                    formula, niveles, tecnico=""):
    """
    Renderiza una ficha metodológica con st.markdown().
    Solo para uso en expanders (tabs 1–4) donde no hay loop+columns.
    En el glosario (Tab 6), usar ficha_as_html() + st.markdown() único por categoría.
    """
    st.markdown(
        ficha_as_html(icono, nombre, tag, simple, analogia, formula, niveles, tecnico),
        unsafe_allow_html=True
    )


# Diccionario centralizado de todas las fichas
FICHAS = {
    "razon_corriente": dict(
        icono="💧", nombre="Razón Corriente (Liquidez)", tag="Liquidez",
        simple=(
            "Mide si la empresa tiene suficiente dinero disponible para pagar "
            "todas sus deudas del próximo año. Es como revisar si hay saldo "
            "en la cuenta antes de pagar las facturas del mes."
        ),
        analogia=(
            "Si debes $1.000.000 este mes y tienes $1.420.000 en el banco, "
            "tu razón corriente es 1.42x: puedes pagar todo y te sobran $420.000."
        ),
        formula="Activos Corrientes  ÷  Pasivos Corrientes",
        tecnico=(
            "Compara los activos que se convertirán en efectivo en menos de 12 meses "
            "(caja, cuentas por cobrar, inventarios) frente a las obligaciones con vencimiento "
            "menor a un año. Indicador estándar de solvencia a corto plazo (Van Horne, 2010)."
        ),
        niveles=[
            ("verde",    "✅ ≥ 1.5x — Holgura financiera"),
            ("amarillo", "⚠️ 1.0 – 1.5x — Zona de atención"),
            ("rojo",     "🚨 < 1.0x — Riesgo de liquidez"),
        ],
    ),
    "prueba_acida": dict(
        icono="🧪", nombre="Prueba Ácida", tag="Liquidez",
        simple=(
            "Es como la razón corriente pero más estricta: excluye los inventarios "
            "porque venderlos puede tardar tiempo. Mide si la empresa puede pagar sus "
            "deudas inmediatas sin necesidad de vender lo que tiene en bodega."
        ),
        analogia=(
            "Si tienes $1.420.000 pero $400.000 están en productos que debes vender "
            "primero, tu liquidez real inmediata es $1.020.000. Eso es la prueba ácida."
        ),
        formula="(Activos Corrientes − Inventarios)  ÷  Pasivos Corrientes",
        tecnico=(
            "Eliminación de los inventarios del numerador porque no son activos de alta "
            "liquidez inmediata. Relevante en sectores con ciclos de inventario largos "
            "(manufactura, construcción). La brecha entre razón corriente y prueba ácida "
            "revela la dependencia de las ventas para honrar obligaciones."
        ),
        niveles=[
            ("verde",    "✅ ≥ 1.0x — Solvencia inmediata"),
            ("amarillo", "⚠️ 0.8 – 1.0x — Vigilar"),
            ("rojo",     "🚨 < 0.8x — Dependencia de ventas"),
        ],
    ),
    "razon_efectivo": dict(
        icono="💵", nombre="Razón de Efectivo", tag="Liquidez",
        simple=(
            "La prueba más estricta de liquidez: ¿cuánto dinero en caja o en el banco "
            "tiene la empresa para pagar sus deudas de hoy? No cuenta inventarios ni "
            "lo que le deben los clientes, solo el dinero disponible ahora mismo."
        ),
        analogia=(
            "Es como el billete en el bolsillo: no cuenta lo que te deben ni lo que "
            "tienes guardado en mercancía. Solo el efectivo del momento."
        ),
        formula="(Efectivo + Equivalentes de Efectivo)  ÷  Pasivos Corrientes",
        tecnico=(
            "Subconjunto ultra-conservador de la razón corriente. Útil en análisis de "
            "crisis de liquidez o stress testing. Valores muy altos pueden indicar "
            "subutilización de recursos (exceso de caja sin invertir)."
        ),
        niveles=[
            ("verde",    "✅ ≥ 0.5x — Liquidez inmediata sólida"),
            ("amarillo", "⚠️ 0.2 – 0.5x — Aceptable"),
            ("rojo",     "🚨 < 0.2x — Baja disponibilidad inmediata"),
        ],
    ),
    "roa": dict(
        icono="📈", nombre="ROA — Retorno sobre Activos", tag="Rentabilidad",
        simple=(
            "Por cada $100 que tiene la empresa (máquinas, edificios, inventarios, "
            "dinero en caja...), ¿cuánta utilidad genera? Un ROA del 5% significa "
            "$5 de ganancia por cada $100 invertidos en activos."
        ),
        analogia=(
            "Como poner $100 en una cuenta de ahorros y recibir $5 de intereses al año. "
            "Cuanto más alto el ROA, más eficiente es la empresa usando lo que tiene."
        ),
        formula="Utilidad Neta  ÷  Activos Totales  × 100",
        tecnico=(
            "Mide la eficiencia global en el uso de todos los recursos de la empresa "
            "independientemente de su estructura de financiamiento. ROA bajo puede indicar "
            "sobredimensionamiento de activos o márgenes deprimidos. Componente clave del "
            "análisis DuPont (Brigham & Houston, 2014)."
        ),
        niveles=[
            ("verde",    "✅ > 10% — Alta eficiencia"),
            ("verde",    "✅ 5 – 10% — Eficiencia aceptable"),
            ("amarillo", "⚠️ 2 – 5% — Baja eficiencia"),
            ("rojo",     "🚨 < 2% — Muy baja / pérdida"),
        ],
    ),
    "roe": dict(
        icono="💼", nombre="ROE — Retorno sobre Patrimonio", tag="Rentabilidad",
        simple=(
            "Mide cuánto ganan los dueños (socios/accionistas) de la empresa por cada "
            "peso que pusieron como capital. Es la rentabilidad desde el punto de vista "
            "del inversionista."
        ),
        analogia=(
            "Si pusiste $100 millones en un negocio y al año tienes $115 millones "
            "de patrimonio, tu ROE es 15%: recuperaste el 15% de tu inversión en un año."
        ),
        formula="Utilidad Neta  ÷  Patrimonio Total  × 100",
        tecnico=(
            "Indicador central para los accionistas. Un ROE alto puede originarse en "
            "alta rentabilidad operativa (deseable) o en alto apalancamiento financiero "
            "(riesgoso). La descomposición DuPont desagrega el ROE en sus tres motores: "
            "ROE = Margen Neto × Rotación de Activos × Multiplicador de Patrimonio."
        ),
        niveles=[
            ("verde",    "✅ > 15% — Rentabilidad alta para el accionista"),
            ("verde",    "✅ 8 – 15% — Rentabilidad aceptable"),
            ("amarillo", "⚠️ 3 – 8% — Baja rentabilidad"),
            ("rojo",     "🚨 < 3% — Muy baja / destrucción de valor"),
        ],
    ),
    "margen_neto": dict(
        icono="📊", nombre="Margen Neto", tag="Rentabilidad",
        simple=(
            "De cada $100 que factura la empresa en ventas, ¿cuántos pesos quedan "
            "como ganancia final después de pagar todo: empleados, materiales, "
            "impuestos, intereses de deuda?"
        ),
        analogia=(
            "Vendes arepas a $1.000 la unidad. Pagas $966 entre ingredientes, gas, "
            "arriendo y sueldo. Te quedan $34. Tu margen neto es 3.4%."
        ),
        formula="Utilidad Neta  ÷  Ventas Netas  × 100",
        tecnico=(
            "Resultado final de toda la cadena de valor: refleja la eficiencia en "
            "costos de producción, gastos de operación, carga financiera e impuestos. "
            "Varía significativamente por sector: comercio suele tener márgenes bajos "
            "(1–4%) mientras servicios especializados pueden superar el 15%."
        ),
        niveles=[
            ("verde",    "✅ > 10% — Margen sólido"),
            ("verde",    "✅ 4 – 10% — Margen aceptable"),
            ("amarillo", "⚠️ 1 – 4% — Margen estrecho"),
            ("rojo",     "🚨 < 1% — Márgenes en riesgo"),
        ],
    ),
    "margen_ebitda": dict(
        icono="⚡", nombre="Margen EBITDA", tag="Rentabilidad",
        simple=(
            "Mide cuánto dinero genera la empresa con sus operaciones principales "
            "ANTES de pagar deudas, impuestos y gastos contables como depreciación. "
            "Es una aproximación a la 'caja operativa' que genera el negocio."
        ),
        analogia=(
            "EBITDA es lo que sobra después de pagar solo los costos del negocio "
            "del día a día, sin contar el pago de créditos bancarios ni los impuestos. "
            "Una empresa con EBITDA positivo y negativo al final puede estarlo 'pagando' en deudas."
        ),
        formula="EBITDA  ÷  Ventas Netas  × 100",
        tecnico=(
            "Earnings Before Interest, Taxes, Depreciation & Amortization. "
            "Ampliamente usado para comparar empresas de distintos países o con "
            "diferentes estructuras de capital. No es un indicador de flujo de caja "
            "libre pero es proxy aceptado de generación operativa (Damodaran, 2012)."
        ),
        niveles=[
            ("verde",    "✅ > 15% — Alta generación operativa"),
            ("verde",    "✅ 8 – 15% — Generación aceptable"),
            ("amarillo", "⚠️ 3 – 8% — Generación ajustada"),
            ("rojo",     "🚨 < 3% — Generación insuficiente"),
        ],
    ),
    "zscore": dict(
        icono="⚖️", nombre="Z-Score de Altman (Z'')", tag="Riesgo financiero",
        simple=(
            "Un puntaje de salud financiera creado por el Prof. Edward Altman (1968). "
            "Combina 4 indicadores y produce un número: mayor = más sana. "
            "Funciona como el resultado de un examen médico: indica si la empresa "
            "está bien, en vigilancia, o en peligro."
        ),
        analogia=(
            "Igual que un médico mira presión arterial, azúcar y colesterol para evaluar "
            "tu salud, el Z-Score mira liquidez, endeudamiento y rentabilidad juntos. "
            "Una sola señal de alarma puede ignorarse; todas al mismo tiempo es riesgo real."
        ),
        formula="Z'' = 6.56·(CT/AT) + 3.26·(UR/AT) + 6.72·(EBIT/AT) + 1.05·(Patrim./Pasivos)",
        tecnico=(
            "Versión Z'' de Altman, Hartzell & Peck (1995) para empresas no cotizadas "
            "en mercados emergentes. X₁=Capital de Trabajo/Activos, X₂=Utilidades Retenidas/Activos, "
            "X₃=EBIT/Activos, X₄=Patrimonio/Pasivos (valor en libros, no bursátil)."
        ),
        niveles=[
            ("verde",    "✅ Z'' > 2.60 — Zona Segura"),
            ("amarillo", "⚠️ 1.10 – 2.60 — Zona de Alerta"),
            ("rojo",     "🚨 Z'' < 1.10 — Zona de Quiebra"),
        ],
    ),
    "deuda_activos": dict(
        icono="🏦", nombre="Deuda / Activos", tag="Endeudamiento",
        simple=(
            "¿Qué porcentaje de todo lo que tiene la empresa fue financiado con deuda? "
            "Si es 52%, por cada $100 de activos, $52 son prestados. "
            "A mayor porcentaje, mayor riesgo financiero."
        ),
        analogia=(
            "Compraste un apartamento de $200 millones. Pediste crédito de $120 millones. "
            "Tu endeudamiento es 60%: el banco es dueño de más de la mitad del apartamento."
        ),
        formula="Pasivos Totales  ÷  Activos Totales  × 100",
        tecnico=(
            "Ratio de estructura financiera que mide el grado de apalancamiento. "
            "También llamado 'razón de endeudamiento' o Debt Ratio. "
            "Un ratio alto implica mayor riesgo de insolvencia pero también mayor "
            "potencial de apalancamiento del ROE (efecto palanca financiero)."
        ),
        niveles=[
            ("verde",    "✅ < 40% — Endeudamiento bajo"),
            ("amarillo", "⚠️ 40 – 60% — Endeudamiento moderado"),
            ("rojo",     "🚨 > 60% — Endeudamiento alto"),
        ],
    ),
    "deuda_capital": dict(
        icono="🔗", nombre="Relación Deuda / Capital (Patrimonio)", tag="Endeudamiento",
        simple=(
            "Compara cuánta deuda tiene la empresa respecto a lo que los dueños "
            "pusieron de su propio dinero. Un 100% significa que la deuda es igual "
            "al capital de los socios."
        ),
        analogia=(
            "Si los dueños pusieron $100 millones y la empresa debe $150 millones al banco, "
            "la relación deuda/capital es 150%: los dueños ponen $1 por cada $1.50 de deuda."
        ),
        formula="Pasivos Totales  ÷  Patrimonio Total  × 100",
        tecnico=(
            "Debt-to-Equity Ratio (D/E). Mide el apalancamiento financiero desde la "
            "perspectiva del accionista. Alto D/E amplifica el ROE en tiempos buenos pero "
            "también amplifica las pérdidas y el riesgo de quiebra. Relacionado con el "
            "multiplicador de apalancamiento en el modelo DuPont."
        ),
        niveles=[
            ("verde",    "✅ < 80% — Saludable"),
            ("amarillo", "⚠️ 80 – 150% — Moderado"),
            ("rojo",     "🚨 > 150% — Alto riesgo"),
        ],
    ),
    "dupont": dict(
        icono="🔄", nombre="Descomposición DuPont del ROE", tag="Análisis avanzado",
        simple=(
            "El análisis DuPont descompone el ROE (lo que ganan los dueños) en tres partes: "
            "¿gana bien por ventas? ¿usa bien sus bienes? ¿depende de deuda para ser rentable? "
            "Permite entender por qué un sector es rentable o no."
        ),
        analogia=(
            "Como diagnosticar por qué un equipo de fútbol gana partidos: ¿por el delantero "
            "(margen)? ¿por el centrocampista que crea jugadas (rotación)? "
            "¿o porque el árbitro es amigo (apalancamiento = deuda)?"
        ),
        formula="ROE  =  Margen Neto  ×  Rotación de Activos  ×  Multiplicador de Apalancamiento",
        tecnico=(
            "Metodología desarrollada por DuPont de Nemours (1914). "
            "Margen Neto = UN/Ventas (eficiencia en costos); "
            "Rotación = Ventas/Activos (eficiencia operativa); "
            "Multiplicador = Activos/Patrimonio (efecto palanca financiera). "
            "Permite comparar empresas de distintos sectores aislando cada motor del ROE."
        ),
        niveles=[
            ("verde",   "✅ Impulsado por margen — modelo de negocio robusto"),
            ("verde",   "✅ Impulsado por rotación — alta eficiencia operativa"),
            ("amarillo","⚠️ Impulsado por apalancamiento — depende de deuda"),
            ("rojo",    "🚨 ROE negativo — destruye valor para los socios"),
        ],
    ),
    "kmeans": dict(
        icono="🤖", nombre="Segmentación K-Means (Machine Learning)", tag="Modelo ML",
        simple=(
            "El algoritmo K-Means agrupa automáticamente las 1.739 empresas en 3 grupos "
            "según su perfil financiero, sin que nadie le diga cómo agruparlas. "
            "Es como si el computador revisara todos los indicadores y dijera: "
            "'estas empresas se parecen entre sí'."
        ),
        analogia=(
            "Como separar frutas en una mesa sin etiquetas: el algoritmo identifica "
            "las manzanas, naranjas y bananos por su forma, color y tamaño. "
            "Aquí separa empresas por su liquidez, rentabilidad y endeudamiento."
        ),
        formula="Minimiza: Σ ||xᵢ - μₖ||²  (distancia al centroide del cluster)",
        tecnico=(
            "Algoritmo de clustering no supervisado con K=3 clusters (justificado por "
            "las tres zonas del Z-Score de Altman). Preprocesamiento: StandardScaler "
            "para normalización de escala. Visualización mediante PCA (2 componentes), "
            "que explica el 32.9% de la varianza total del conjunto de 48 features."
        ),
        niveles=[
            ("azul",    "Cluster A — Perfil moderado (mayoría del tejido empresarial)"),
            ("verde",   "Cluster B — Alto desempeño (márgenes EBITDA >15%)"),
            ("amarillo","Cluster C — Atípico (valores extremos, validar individualmente)"),
        ],
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑAS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Resumen macroeconómico",
    "Liquidez por sector",
    "Riesgo Z-Score y endeudamiento",
    "Rentabilidad y clusters",
    "Simulador financiero",
    "Glosario de indicadores",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESUMEN MACROECONÓMICO
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── KPIs ──────────────────────────────────────────────────────────────────
    col_z = f"Zona_ZScore_{año_sel}"
    n_total  = len(df)
    n_segura = (df[col_z] == "🟢 Zona Segura").sum()  if col_z in df.columns else 0
    n_alerta = (df[col_z] == "🟡 Zona Alerta").sum()  if col_z in df.columns else 0
    n_quiebra= (df[col_z] == "🔴 Zona Quiebra").sum() if col_z in df.columns else 0
    roa_med  = df[f"Rendimiento Sobre Los Activos (ROA) (%) {año_sel}"].median()
    roe_med  = df[f"Rendimiento Sobre El Patrimonio (ROE) (%) {año_sel}"].median()
    liq_med  = df[f"Razón De Liquidez (x) {año_sel}"].median()
    mn_med   = df[f"Margen Neto (%) {año_sel}"].median()
    z_med    = df[f"Modelo Z-Score de Altman {año_sel}"].median()

    # Fila 1 KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi_html("🏢 Empresas analizadas",  f"{n_total:,}",
                         f"{df['Sección CIIU'].nunique()} sectores CIIU", "kpi-info"),  True)
    k2.markdown(kpi_html("🟢 Zona Segura",
                         f"{n_segura:,}", f"{n_segura/n_total:.1%} del total", "kpi-safe"), True)
    k3.markdown(kpi_html("🟡 Zona Alerta",
                         f"{n_alerta:,}", f"{n_alerta/n_total:.1%} requieren monitoreo", "kpi-warn"), True)
    k4.markdown(kpi_html("🔴 Zona Quiebra",
                         f"{n_quiebra:,}", f"{n_quiebra/n_total:.1%} riesgo crítico", "kpi-danger"), True)

    st.markdown("<br>", True)

    # Fila 2 KPIs
    k5, k6, k7, k8 = st.columns(4)
    k5.markdown(kpi_html("📐 Z-Score mediano",  f"{z_med:.2f}",
                         "Umbral seguro: >2.60",    "kpi-info"), True)
    k6.markdown(kpi_html("📈 ROA mediano",      f"{roa_med:.1f}%",
                         "Retorno sobre activos",   "kpi-info"), True)
    k7.markdown(kpi_html("💼 ROE mediano",      f"{roe_med:.1f}%",
                         "Retorno sobre patrimonio","kpi-info"), True)
    k8.markdown(kpi_html("💧 Liquidez corriente",f"{liq_med:.2f}x",
                         "Mediana; ≥1.0x = solvente","kpi-info"), True)

    # ── Expanders explicativos de los KPIs ────────────────────────────────────
    with st.expander("Definiciones de indicadores del resumen ejecutivo"):
        ec1, ec2 = st.columns(2)
        with ec1:
            ficha_indicador(**FICHAS["zscore"])
            ficha_indicador(**FICHAS["roa"])
        with ec2:
            ficha_indicador(**FICHAS["roe"])
            ficha_indicador(**FICHAS["razon_corriente"])

    st.markdown("<br>", True)

    # ── Gráficas ──────────────────────────────────────────────────────────────
    col_l, col_r = st.columns([1.2, 0.8])

    with col_l:
        st.markdown("<p class='section-header'>Evolución temporal de indicadores clave (medianas)</p>", True)

        serie = {}
        for yr in AÑOS:
            mask_all = df_raw.index  # sin filtro de año en la evolución
            serie[yr] = {
                "ROA":     df_raw[f"Rendimiento Sobre Los Activos (ROA) (%) {yr}"].median(),
                "ROE":     df_raw[f"Rendimiento Sobre El Patrimonio (ROE) (%) {yr}"].median(),
                "Margen Neto": df_raw[f"Margen Neto (%) {yr}"].median(),
                "Liquidez": df_raw[f"Razón De Liquidez (x) {yr}"].median(),
                "Z-Score":  df_raw[f"Modelo Z-Score de Altman {yr}"].median(),
            }

        df_evol = pd.DataFrame(serie).T.reset_index().rename(columns={"index": "Año"})

        fig_evol = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Rentabilidad (% — mediana)", "Liquidez y Z-Score (mediana)"),
            vertical_spacing=0.14,
        )
        for met, color in [("ROA", CLR["azul_claro"]),
                           ("ROE", CLR["verde_claro"]),
                           ("Margen Neto", CLR["amarillo"])]:
            fig_evol.add_trace(go.Scatter(
                x=df_evol["Año"], y=df_evol[met].round(2),
                name=met, line=dict(width=2.5, color=color),
                mode="lines+markers+text",
                text=[f"{v:.1f}%" for v in df_evol[met]],
                textposition="top center", textfont_size=9,
            ), row=1, col=1)

        fig_evol.add_trace(go.Scatter(
            x=df_evol["Año"], y=df_evol["Liquidez"].round(3),
            name="Liquidez (x)", line=dict(width=2.5, color=CLR["rojo"], dash="dot"),
            mode="lines+markers",
        ), row=2, col=1)
        fig_evol.add_trace(go.Scatter(
            x=df_evol["Año"], y=df_evol["Z-Score"].round(2),
            name="Z-Score", line=dict(width=2.5, color=CLR["verde"]),
            mode="lines+markers",
        ), row=2, col=1)
        # Línea umbral Z segura
        fig_evol.add_hline(y=2.60, row=2, col=1, line_dash="dash",
                           line_color=CLR["verde"], opacity=0.4,
                           annotation_text=" Umbral seguro (2.60)",
                           annotation_font_color=CLR["verde"], annotation_font_size=10)

        fig_evol.update_layout(
            **LAYOUT_BASE, height=420,
            title=dict(text="Desempeño financiero 2021–2024 — universo completo (1.739 empresas)",
                       font_size=12, font_color="#C20E1A"),
        )
        st.plotly_chart(fig_evol, use_container_width=True)
        interp(
            "📌 <b>Recuperación (2021–2022):</b> mejora generalizada en rentabilidad post-pandemia. "
            "<b>Consolidación (2022–2023):</b> estabilización con leve descenso en márgenes. "
            "<b>Moderación (2023–2024):</b> contracción del ROA y Margen Neto; el Z-Score se mantiene "
            "en zona segura gracias a la solvencia patrimonial."
        )

    with col_r:
        st.markdown("<p class='section-header'>Distribución Z-Score — " + str(año_sel) + "</p>", True)

        # Donut zonas
        if col_z in df.columns:
            dist_z = df[col_z].value_counts().reset_index()
            dist_z.columns = ["Zona", "n"]
            dist_z["pct"] = dist_z["n"] / dist_z["n"].sum()

            colors_z = {
                "🟢 Zona Segura":  CLR["verde"],
                "🟡 Zona Alerta":  CLR["amarillo"],
                "🔴 Zona Quiebra": CLR["rojo"],
            }
            fig_donut = go.Figure(go.Pie(
                labels=dist_z["Zona"],
                values=dist_z["n"],
                hole=0.6,
                textinfo="percent+label",
                textfont_size=11,
                marker=dict(
                    colors=[colors_z.get(z, CLR["gris_medio"]) for z in dist_z["Zona"]],
                    line=dict(color="#FFFFFF", width=2),
                ),
                hovertemplate="%{label}<br><b>%{value:,} empresas</b><br>%{percent}<extra></extra>",
            ))
            fig_donut.add_annotation(
                text=f"<b>{n_total:,}</b><br><span style='font-size:10px'>Empresas</span>",
                x=0.5, y=0.5, showarrow=False, font_size=14, font_color="#FFFFFF",
            )
            apply_layout(fig_donut, height=260)
            st.plotly_chart(fig_donut, use_container_width=True)

        # Distribución sectorial
        st.markdown("<p class='section-header'>Empresas por sector</p>", True)
        dist_sec = df.groupby("Sector_Corto").size().reset_index(name="n")
        dist_sec = dist_sec.sort_values("n", ascending=True).tail(12)

        fig_sec = go.Figure(go.Bar(
            x=dist_sec["n"], y=dist_sec["Sector_Corto"],
            orientation="h",
            marker_color=CLR["azul_claro"],
            text=dist_sec["n"],
            textposition="outside",
            textfont_size=10,
        ))
        apply_layout(fig_sec, height=320)
        st.plotly_chart(fig_sec, use_container_width=True)

    # ── Fases temporales ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<p class='section-header'>Fases del ciclo financiero 2021–2024</p>", True)
    f1, f2, f3 = st.columns(3)
    with f1:
        interp("🟢 <b>Fase 1 — Recuperación post-pandemia (2021–2022)</b><br>"
               "Mejora generalizada en márgenes operativos y EBITDA. "
               "El ROE aumentó impulsado principalmente por el multiplicador de "
               "apalancamiento en sectores con alta intensidad de activos.")
    with f2:
        interp("🟡 <b>Fase 2 — Consolidación (2022–2023)</b><br>"
               "Estabilización de indicadores con leve deterioro en liquidez "
               "en sectores de alta exposición a tasas de interés (Construcción, "
               "Inmobiliario). La rotación de activos se mantiene estable.")
    with f3:
        interp("🟠 <b>Fase 3 — Moderación (2023–2024)</b><br>"
               "Contracción de márgenes hacia medianas históricas. El análisis "
               "DuPont confirma que las variaciones del ROE responden principalmente "
               "a cambios en márgenes, no en eficiencia operativa.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIQUIDEZ
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    liq_col  = f"Razón De Liquidez (x) {año_sel}"
    acida_col= f"Prueba Ácida (x) {año_sel}"
    efec_col = f"Razón De Efectivo (x) {año_sel}"

    # KPIs
    ka, kb, kc = st.columns(3)
    liq_all = df[liq_col].median()
    aci_all = df[acida_col].median()
    efe_all = df[efec_col].median()

    def kpi_liq_clase(v, u1=1.5, u2=1.0):
        if v >= u1: return "kpi-safe"
        if v >= u2: return "kpi-warn"
        return "kpi-danger"

    ka.markdown(kpi_html("💧 Razón Corriente (mediana)",
                         f"{liq_all:.2f}x",
                         "✅ ≥1.5x holgura | ⚠️ 1.0–1.5x | 🚨 <1.0x",
                         kpi_liq_clase(liq_all)), True)
    kb.markdown(kpi_html("🧪 Prueba Ácida (mediana)",
                         f"{aci_all:.2f}x",
                         "Sin inventarios; mide liquidez inmediata",
                         kpi_liq_clase(aci_all, 1.0, 0.8)), True)
    kc.markdown(kpi_html("💵 Razón de Efectivo (mediana)",
                         f"{efe_all:.2f}x",
                         "Solo caja y equivalentes",
                         kpi_liq_clase(efe_all, 0.5, 0.2)), True)

    # ── Expanders liquidez ────────────────────────────────────────────────────
    with st.expander("Definición y lectura de los indicadores de liquidez"):
        el1, el2, el3 = st.columns(3)
        with el1:
            ficha_indicador(**FICHAS["razon_corriente"])
        with el2:
            ficha_indicador(**FICHAS["prueba_acida"])
        with el3:
            ficha_indicador(**FICHAS["razon_efectivo"])

    st.markdown("<br>", True)

    # ── Barras por sector ─────────────────────────────────────────────────────
    col_l2, col_r2 = st.columns([1.2, 0.8])

    with col_l2:
        st.markdown("<p class='section-header'>Razón corriente y prueba ácida por sector</p>", True)

        liq_sec = (df.groupby("Sector_Corto")[[liq_col, acida_col]]
                   .median().reset_index()
                   .rename(columns={liq_col: "Razón Corriente", acida_col: "Prueba Ácida"})
                   .sort_values("Razón Corriente", ascending=False))

        fig_liq = go.Figure()
        fig_liq.add_trace(go.Bar(
            name="Razón Corriente",
            x=liq_sec["Sector_Corto"], y=liq_sec["Razón Corriente"],
            marker_color=CLR["azul_claro"],
            text=liq_sec["Razón Corriente"].round(2),
            textposition="outside", textfont_size=9,
        ))
        fig_liq.add_trace(go.Bar(
            name="Prueba Ácida",
            x=liq_sec["Sector_Corto"], y=liq_sec["Prueba Ácida"],
            marker_color=CLR["verde_claro"],
            text=liq_sec["Prueba Ácida"].round(2),
            textposition="outside", textfont_size=9,
        ))
        fig_liq.add_hline(y=1.0, line_dash="dash", line_color=CLR["rojo"],
                          opacity=0.6, annotation_text=" Mínimo solvencia (1.0x)",
                          annotation_font_color=CLR["rojo"], annotation_font_size=9)
        fig_liq.add_hline(y=1.5, line_dash="dash", line_color=CLR["verde"],
                          opacity=0.4, annotation_text=" Holgura (1.5x)",
                          annotation_font_color=CLR["verde"], annotation_font_size=9)
        fig_liq.update_layout(**LAYOUT_BASE, barmode="group", height=380,
                              xaxis_tickangle=-35)
        st.plotly_chart(fig_liq, use_container_width=True)

        interp("📌 <b>¿Qué es la razón corriente?</b> Mide cuánto dinero tiene disponible "
               "la empresa por cada peso que debe pagar este año. Una razón de 1.5x significa "
               "que por cada $100 de deuda tiene $150 disponibles. La <b>prueba ácida</b> "
               "excluye inventarios, revelando la liquidez sin depender de las ventas.")

    with col_r2:
        st.markdown("<p class='section-header'>Dispersión: corriente vs. prueba ácida</p>", True)

        fig_scat = px.scatter(
            df.dropna(subset=[liq_col, acida_col]),
            x=liq_col, y=acida_col,
            color="Sector_Corto",
            size_max=8,
            opacity=0.65,
            labels={liq_col: "Razón Corriente (x)",
                    acida_col: "Prueba Ácida (x)",
                    "Sector_Corto": "Sector"},
            color_discrete_sequence=COLOR_SEQ,
        )
        fig_scat.update_traces(marker_size=5)
        fig_scat.add_vline(x=1.0, line_dash="dot", line_color=CLR["rojo"], opacity=0.5)
        fig_scat.add_hline(y=1.0, line_dash="dot", line_color=CLR["rojo"], opacity=0.5)
        apply_layout(fig_scat, height=340)
        st.plotly_chart(fig_scat, use_container_width=True)

        interp("Empresas en el cuadrante inferior-izquierdo (ambos ratios &lt;1.0x) "
               "presentan las mayores presiones de liquidez y suelen coincidir con "
               "zonas de alerta o quiebra en el Z-Score de Altman.")

    # ── Evolución temporal de liquidez ────────────────────────────────────────
    st.markdown("<p class='section-header'>Evolución de liquidez sectorial 2021–2024</p>", True)

    top_sectores = (df_raw.groupby("Sector_Corto")
                    .size().nlargest(8).index.tolist())

    liq_evol = []
    for yr in AÑOS:
        col = f"Razón De Liquidez (x) {yr}"
        if col in df_raw.columns:
            tmp = (df_raw[df_raw["Sector_Corto"].isin(top_sectores)]
                   .groupby("Sector_Corto")[col].median().reset_index())
            tmp.columns = ["Sector", "Liquidez"]
            tmp["Año"] = yr
            liq_evol.append(tmp)

    df_liq_evol = pd.concat(liq_evol)
    fig_lev = px.line(
        df_liq_evol, x="Año", y="Liquidez", color="Sector",
        markers=True, color_discrete_sequence=COLOR_SEQ,
        labels={"Liquidez": "Razón Corriente (mediana)"},
    )
    fig_lev.add_hline(y=1.0, line_dash="dash", line_color=CLR["rojo"],
                      opacity=0.5, annotation_text=" Umbral mínimo")
    apply_layout(fig_lev, height=340)
    st.plotly_chart(fig_lev, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RIESGO Z-SCORE & ENDEUDAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    z_col   = f"Modelo Z-Score de Altman {año_sel}"
    deuda_col = f"Deuda / Activos (%) {año_sel}"
    deudat_col= f"Relación Deuda/Capital (%) {año_sel}"
    col_zona  = f"Zona_ZScore_{año_sel}"

    # KPIs
    z_med_filt  = df[z_col].median()
    deuda_med   = df[deuda_col].median()
    deudat_med  = df[deudat_col].median()
    n_seg  = (df[col_zona] == "🟢 Zona Segura").sum()  if col_zona in df.columns else 0
    n_ale  = (df[col_zona] == "🟡 Zona Alerta").sum()  if col_zona in df.columns else 0
    n_qui  = (df[col_zona] == "🔴 Zona Quiebra").sum() if col_zona in df.columns else 0

    kz1, kz2, kz3, kz4 = st.columns(4)
    kz1.markdown(kpi_html("⚖️ Z-Score mediano",   f"{z_med_filt:.2f}",
                          "Zona segura: >2.60",
                          "kpi-safe" if z_med_filt > 2.60
                          else ("kpi-warn" if z_med_filt >= 1.10 else "kpi-danger")), True)
    kz2.markdown(kpi_html("🟢 Zona Segura",  f"{n_seg:,}", f"{n_seg/(len(df) or 1):.1%}", "kpi-safe"), True)
    kz3.markdown(kpi_html("🟡 Zona Alerta",  f"{n_ale:,}", f"{n_ale/(len(df) or 1):.1%}", "kpi-warn"), True)
    kz4.markdown(kpi_html("🔴 Zona Quiebra", f"{n_qui:,}", f"{n_qui/(len(df) or 1):.1%}", "kpi-danger"), True)

    # ── Expanders riesgo ──────────────────────────────────────────────────────
    with st.expander("Metodología del Z-Score de Altman e indicadores de endeudamiento"):
        ez1, ez2, ez3 = st.columns(3)
        with ez1:
            ficha_indicador(**FICHAS["zscore"])
        with ez2:
            ficha_indicador(**FICHAS["deuda_activos"])
        with ez3:
            ficha_indicador(**FICHAS["deuda_capital"])

    st.markdown("<br>", True)

    col_zl, col_zr = st.columns([1.1, 0.9])

    with col_zl:
        # Distribución Z-Score por sector
        st.markdown("<p class='section-header'>Z-Score mediano y deuda/activos por sector</p>", True)

        if z_col in df.columns and deuda_col in df.columns:
            zsec = (df.groupby("Sector_Corto")
                    .agg(ZScore=(z_col, "median"),
                         Deuda_Activos=(deuda_col, "median"))
                    .reset_index()
                    .sort_values("ZScore", ascending=False))

            fig_z = make_subplots(specs=[[{"secondary_y": True}]])
            fig_z.add_trace(go.Bar(
                x=zsec["Sector_Corto"], y=zsec["ZScore"],
                name="Z-Score (mediana)",
                marker_color=[
                    CLR["verde"] if v > 2.60
                    else (CLR["amarillo"] if v >= 1.10 else CLR["rojo"])
                    for v in zsec["ZScore"]
                ],
                text=zsec["ZScore"].round(2), textposition="outside", textfont_size=9,
            ), secondary_y=False)
            fig_z.add_trace(go.Scatter(
                x=zsec["Sector_Corto"], y=zsec["Deuda_Activos"],
                name="Deuda / Activos % (eje der.)",
                mode="lines+markers",
                line=dict(color=CLR["amarillo"], width=2, dash="dot"),
                marker_size=7,
            ), secondary_y=True)
            fig_z.add_hline(y=2.60, line_dash="dash", line_color=CLR["verde"],
                            opacity=0.5, annotation_text=" Segura (2.60)",
                            annotation_font_size=9, annotation_font_color=CLR["verde"])
            fig_z.add_hline(y=1.10, line_dash="dash", line_color=CLR["rojo"],
                            opacity=0.5, annotation_text=" Quiebra (1.10)",
                            annotation_font_size=9, annotation_font_color=CLR["rojo"])
            fig_z.update_layout(**LAYOUT_BASE, height=380, xaxis_tickangle=-35)
            fig_z.update_yaxes(title_text="Z-Score de Altman", secondary_y=False,
                               title_font_color="#C20E1A")
            fig_z.update_yaxes(title_text="Deuda / Activos (%)", secondary_y=True,
                               title_font_color="#854F0B")
            st.plotly_chart(fig_z, use_container_width=True)

        interp("📌 <b>Z-Score de Altman (Z''):</b> modelo multivariado que combina "
               "4 ratios financieros ponderados para predecir riesgo de insolvencia. "
               "Fue adaptado por Altman, Hartzell & Peck (1995) para mercados emergentes. "
               "<b>Zona Segura Z>2.60 | Alerta 1.10≤Z≤2.60 | Quiebra Z&lt;1.10</b>")

    with col_zr:
        # Semáforo de riesgo por tamaño y sector
        st.markdown("<p class='section-header'>Riesgo por tamaño empresarial</p>", True)

        if col_zona in df.columns:
            risk_tam = (df.groupby(["Clasificación_957", col_zona])
                        .size().reset_index(name="n"))
            risk_tam["total"] = risk_tam.groupby("Clasificación_957")["n"].transform("sum")
            risk_tam["pct"] = risk_tam["n"] / risk_tam["total"]

            fig_rk = px.bar(
                risk_tam,
                x="Clasificación_957", y="pct", color=col_zona,
                barmode="stack",
                color_discrete_map={
                    "🟢 Zona Segura":  CLR["verde"],
                    "🟡 Zona Alerta":  CLR["amarillo"],
                    "🔴 Zona Quiebra": CLR["rojo"],
                },
                labels={"pct": "Proporción", "Clasificación_957": "Tamaño", col_zona: "Zona"},
            )
            fig_rk.update_layout(**LAYOUT_BASE, height=270,
                                 yaxis_tickformat=".0%")
            st.plotly_chart(fig_rk, use_container_width=True)

        # Top 15 empresas en zona quiebra
        st.markdown("<p class='section-header'>Empresas en mayor riesgo — Z-Score más bajo</p>", True)
        if col_zona in df.columns:
            df_qui = (df[df[col_zona] == "🔴 Zona Quiebra"]
                      [["Compañía", "Sector_Corto", z_col]]
                      .sort_values(z_col).head(12)
                      .rename(columns={z_col: "Z-Score"}))
            df_qui["Z-Score"] = df_qui["Z-Score"].round(3)

            fig_qui = go.Figure(go.Bar(
                x=df_qui["Z-Score"],
                y=df_qui["Compañía"],
                orientation="h",
                marker_color=CLR["rojo"],
                text=df_qui["Z-Score"],
                textposition="outside",
                textfont_size=9,
            ))
            apply_layout(fig_qui, height=330)
            st.plotly_chart(fig_qui, use_container_width=True)

    # ── Heatmap Riesgo sectorial ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<p class='section-header'>Mapa de calor: % empresas en Zona Quiebra por sector y año</p>", True)

    heat_data = {}
    for yr in AÑOS:
        cz = f"Zona_ZScore_{yr}"
        if cz in df_raw.columns:
            tmp = (df_raw.groupby("Sector_Corto")
                   .apply(lambda g: (g[cz] == "🔴 Zona Quiebra").sum() / len(g) * 100)
                   .round(1))
            heat_data[str(yr)] = tmp

    df_heat = pd.DataFrame(heat_data).dropna(how="all")
    df_heat = df_heat.sort_values(str(año_sel), ascending=False)

    fig_heat = go.Figure(go.Heatmap(
        z=df_heat.values,
        x=df_heat.columns.tolist(),
        y=df_heat.index.tolist(),
        colorscale=[[0, CLR["verde"]], [0.4, CLR["amarillo"]], [1, CLR["rojo"]]],
        text=np.round(df_heat.values, 1),
        texttemplate="%{text}%",
        textfont_size=10,
        hovertemplate="%{y} — %{x}<br><b>%{z:.1f}% en Zona Quiebra</b><extra></extra>",
        colorbar=dict(title=dict(text="% Quiebra", side="right")),
    ))
    fig_heat.update_layout(**LAYOUT_BASE, height=420)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── NUEVA SECCIÓN: Decantación de componentes X1-X4 ──────────────────
    st.markdown("---")
    st.markdown(
        f"<p style='font-family:Montserrat,Arial,sans-serif;font-size:.85rem;"
        f"font-weight:600;color:#222222;border-bottom:2px solid {CLR['rojo']};"
        f"padding-bottom:.35rem;margin-bottom:.9rem;'>Decantación de componentes Z-Score por sector — {año_sel}</p>",
        unsafe_allow_html=True
    )
    interp(
        "El Z-Score combina cuatro componentes con diferentes pesos. "
        "Esta visualización muestra la <b>contribución mediana ponderada de cada componente</b> "
        "por sector en el año seleccionado. El componente con menor contribución es el que "
        "<b>arrastra</b> a las empresas de cada sector hacia zonas de mayor riesgo. "
        "X1 = Liquidez (CT/AT) · X2 = Rent. acumulada (UR/AT) · X3 = Efic. operativa (EBIT/AT) · X4 = Solvencia (Pat/Pas)"
    )

    col_contrib_cols = {
        "X1 — Liquidez (CT/AT)":           f"Contrib_X1_{año_sel}",
        "X2 — Rent. acumulada (UR/AT)":    f"Contrib_X2_{año_sel}",
        "X3 — Efic. operativa (EBIT/AT)":  f"Contrib_X3_{año_sel}",
        "X4 — Solvencia (Pat/Pas)":        f"Contrib_X4_{año_sel}",
    }
    contrib_ok = {k: v for k, v in col_contrib_cols.items() if v in df.columns}

    if contrib_ok:
        df_dec = df[["Sector_Corto"] + list(contrib_ok.values())].dropna()
        df_dec = df_dec.rename(columns={v: k for k, v in contrib_ok.items()})
        dec_sec = df_dec.groupby("Sector_Corto")[list(contrib_ok.keys())].median().round(4)
        dec_sec = dec_sec.sort_values(list(contrib_ok.keys())[0], ascending=False)

        fig_dec = go.Figure()
        colores_x = [CLR["azul_claro"], CLR["gris_oscuro"], CLR["rojo"], CLR["verde"]]
        for i, comp in enumerate(contrib_ok.keys()):
            fig_dec.add_trace(go.Bar(
                name=comp,
                x=dec_sec.index,
                y=dec_sec[comp],
                marker_color=colores_x[i % len(colores_x)],
            ))
        fig_dec.update_layout(
            **LAYOUT_BASE, barmode="group", height=380,
            xaxis_tickangle=-35,
            title=dict(
                text=f"Contribución mediana ponderada de X1–X4 al Z-Score por sector — {año_sel}",
                font_size=11, font_color=CLR["gris_oscuro"], x=0
            ),
            yaxis_title="Contribución al Z-Score (ponderada)",
        )
        st.plotly_chart(fig_dec, use_container_width=True)

        # Componente más débil por sector
        dec_sec["Componente débil"] = dec_sec[list(contrib_ok.keys())].idxmin(axis=1)
        st.markdown(
            f"<p style='font-family:Montserrat,Arial,sans-serif;font-size:.85rem;"
            f"font-weight:600;color:#222222;border-bottom:2px solid {CLR['rojo']};"
            f"padding-bottom:.35rem;margin-bottom:.9rem;'>Componente que más arrastra el Z-Score por sector</p>",
            unsafe_allow_html=True
        )
        st.dataframe(
            dec_sec[["Componente débil"]].rename(
                columns={"Componente débil": f"Factor limitante del Z-Score ({año_sel})"}
            ),
            use_container_width=True
        )
    else:
        interp("Los datos de componentes X1–X4 no están disponibles en el dataset cargado. "
               "Ejecuta el cuaderno de Jupyter actualizado para generarlos.")

    # ── NUEVA SECCIÓN: Matriz de transición de zonas 2021→2024 ───────────
    st.markdown("---")
    st.markdown(
        f"<p style='font-family:Montserrat,Arial,sans-serif;font-size:.85rem;"
        f"font-weight:600;color:#222222;border-bottom:2px solid {CLR['rojo']};"
        f"padding-bottom:.35rem;margin-bottom:.9rem;'>Matriz de transición de zonas Z-Score — 2021 → 2024</p>",
        unsafe_allow_html=True
    )
    interp(
        "La matriz de transición describe <b>cómo se movieron las empresas entre zonas de riesgo</b> "
        "durante el período 2021–2024. Cada celda muestra el porcentaje de empresas que, estando "
        "en una zona en 2021, terminaron en otra zona en 2024. La diagonal principal indica permanencia: "
        "empresas que no cambiaron de zona. Valores altos fuera de la diagonal indican movilidad financiera. "
        "<b>Validación empírica de Altman:</b> alta permanencia en zona de quiebra confirma la premisa "
        "de que el Z-Score captura señales persistentes 2–3 años antes de dificultades financieras."
    )

    z21_col = "Zona_ZScore_2021"
    z24_col = "Zona_ZScore_2024"
    mov_col = "Movimiento_ZScore"

    if z21_col in df_raw.columns and z24_col in df_raw.columns:
        df_tr = df_raw[[z21_col, z24_col, mov_col]].copy()
        df_tr = df_tr[
            (~df_tr[z21_col].isin(["nan","<NA>"])) &
            (~df_tr[z24_col].isin(["nan","<NA>"]))
        ].dropna()

        if len(df_tr) > 0:
            col_tr1, col_tr2 = st.columns([1.2, 0.8])

            with col_tr1:
                # Heatmap de la matriz de transición
                ORDEN_Z = ["🔴 Zona Quiebra", "🟡 Zona Alerta", "🟢 Zona Segura"]
                mtz = pd.crosstab(df_tr[z21_col], df_tr[z24_col], normalize="index") * 100
                mtz = mtz.reindex(index=[z for z in ORDEN_Z if z in mtz.index],
                                   columns=[z for z in ORDEN_Z if z in mtz.columns],
                                   fill_value=0).round(1)

                fig_mtz = go.Figure(go.Heatmap(
                    z=mtz.values,
                    x=mtz.columns.tolist(),
                    y=mtz.index.tolist(),
                    colorscale=[[0, "#FFFFFF"], [0.5, CLR["amarillo"]], [1, CLR["verde"]]],
                    text=[[f"{v:.1f}%" for v in row] for row in mtz.values],
                    texttemplate="%{text}",
                    textfont=dict(size=13, color="#222222"),
                    hovertemplate="Zona 2021: %{y}<br>Zona 2024: %{x}<br>%{text} de empresas<extra></extra>",
                    colorbar=dict(title=dict(text="% empresas")),
                    zmin=0, zmax=100,
                ))
                fig_mtz.update_layout(
                    **LAYOUT_BASE, height=320,
                    xaxis_title="Zona en 2024",
                    yaxis_title="Zona en 2021",
                    title=dict(text="Matriz de transición — % por fila (zona de origen)",
                               font_size=11, font_color=CLR["gris_oscuro"], x=0),
                )
                st.plotly_chart(fig_mtz, use_container_width=True)

            with col_tr2:
                # Barras de movilidad
                if mov_col in df_tr.columns:
                    mov_counts = df_tr[mov_col].value_counts()
                    colores_mov = {"Mejoró": CLR["verde"], "Permaneció": CLR["gris_oscuro"],
                                   "Empeoró": CLR["rojo"], "Sin dato": "#CFCFCF"}
                    fig_mov = go.Figure(go.Bar(
                        x=mov_counts.index.tolist(),
                        y=mov_counts.values.tolist(),
                        marker_color=[colores_mov.get(m, CLR["azul_claro"])
                                      for m in mov_counts.index],
                        text=[f"{v:,}<br>({v/len(df_tr)*100:.1f}%)" for v in mov_counts.values],
                        textposition="outside",
                        textfont_size=11,
                    ))
                    fig_mov.update_layout(
                        **LAYOUT_BASE, height=320,
                        yaxis_title="Número de empresas",
                        title=dict(text="Movilidad entre zonas 2021→2024",
                                   font_size=11, font_color=CLR["gris_oscuro"], x=0),
                    )
                    st.plotly_chart(fig_mov, use_container_width=True)
    else:
        interp("La matriz de transición requiere datos de zona Z-Score para 2021 y 2024. "
               "Ejecuta el cuaderno de Jupyter actualizado para generarlos y exportar el CSV.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RENTABILIDAD & CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    roa_col  = f"Rendimiento Sobre Los Activos (ROA) (%) {año_sel}"
    roe_col  = f"Rendimiento Sobre El Patrimonio (ROE) (%) {año_sel}"
    mn_col   = f"Margen Neto (%) {año_sel}"
    ebitda_col= f"Margen Ebitda (%) {año_sel}"
    rot_col  = f"Rotación De Activos (x) {año_sel}"

    # KPIs
    kr1, kr2, kr3, kr4 = st.columns(4)
    kr1.markdown(kpi_html("📈 ROA mediano",  f"{df[roa_col].median():.1f}%",
                          "Retorno sobre activos",    "kpi-info"), True)
    kr2.markdown(kpi_html("💼 ROE mediano",  f"{df[roe_col].median():.1f}%",
                          "Retorno sobre patrimonio", "kpi-info"), True)
    kr3.markdown(kpi_html("📊 Margen Neto",  f"{df[mn_col].median():.1f}%",
                          "De cada $100 vendidos",    "kpi-info"), True)
    kr4.markdown(kpi_html("⚡ Margen EBITDA", f"{df[ebitda_col].median():.1f}%",
                          "Generación de caja operativa", "kpi-info"), True)

    # ── Expanders rentabilidad ────────────────────────────────────────────────
    with st.expander("Definición y lectura de los indicadores de rentabilidad"):
        er1, er2 = st.columns(2)
        with er1:
            ficha_indicador(**FICHAS["roa"])
            ficha_indicador(**FICHAS["margen_neto"])
        with er2:
            ficha_indicador(**FICHAS["roe"])
            ficha_indicador(**FICHAS["margen_ebitda"])

    st.markdown("<br>", True)
    col_r1, col_r2 = st.columns([1.1, 0.9])

    with col_r1:
        # ROA por sector
        st.markdown("<p class='section-header'>ROA y Margen Neto por sector — " + str(año_sel) + "</p>", True)
        rent_sec = (df.groupby("Sector_Corto")
                    .agg(ROA=(roa_col, "median"), Margen_Neto=(mn_col, "median"))
                    .reset_index().sort_values("ROA", ascending=False))

        fig_rent = go.Figure()
        fig_rent.add_trace(go.Bar(
            name="ROA (%)", x=rent_sec["Sector_Corto"], y=rent_sec["ROA"],
            marker_color=CLR["azul_claro"],
            text=rent_sec["ROA"].round(1), textposition="outside", textfont_size=9,
        ))
        fig_rent.add_trace(go.Scatter(
            name="Margen Neto (%)", x=rent_sec["Sector_Corto"], y=rent_sec["Margen_Neto"],
            mode="lines+markers", line=dict(color=CLR["amarillo"], width=2.5),
            marker_size=8,
        ))
        apply_layout(fig_rent, height=370)
        fig_rent.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_rent, use_container_width=True)

        interp("📌 <b>ROA</b> (Return on Assets): mide la eficiencia en el uso de activos. "
               "Un ROA alto en servicios profesionales (M) refleja la intensidad de capital "
               "baja del sector. La caída en educación (P) sugiere altos activos fijos con "
               "rentabilidad presionada por costos operativos crecientes.")

    with col_r2:
        # Top 20 empresas por ROA
        st.markdown("<p class='section-header'>Top 15 empresas más rentables (ROA)</p>", True)
        top_roa = (df.nlargest(15, roa_col)
                   [["Compañía", "Sector_Corto", roa_col, roe_col, mn_col]]
                   .rename(columns={roa_col: "ROA%", roe_col: "ROE%", mn_col: "MN%"}))
        top_roa[["ROA%","ROE%","MN%"]] = top_roa[["ROA%","ROE%","MN%"]].round(1)

        fig_top = go.Figure(go.Bar(
            x=top_roa["ROA%"],
            y=top_roa["Compañía"].str[:30],
            orientation="h",
            marker=dict(
                color=top_roa["ROA%"],
                colorscale=[[0, CLR["azul_claro"]], [0.5, CLR["verde_claro"]], [1, CLR["verde"]]],
                showscale=False,
            ),
            text=top_roa["ROA%"].astype(str) + "%",
            textposition="outside", textfont_size=9,
            customdata=top_roa[["Sector_Corto", "ROE%", "MN%"]],
            hovertemplate=(
                "<b>%{y}</b><br>Sector: %{customdata[0]}<br>"
                "ROA: %{x:.1f}%<br>ROE: %{customdata[1]:.1f}%<br>"
                "Margen Neto: %{customdata[2]:.1f}%<extra></extra>"
            ),
        ))
        apply_layout(fig_top, height=400)
        st.plotly_chart(fig_top, use_container_width=True)

    # ── DuPont decomposición ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<p class='section-header'>Descomposición DuPont del ROE por sector — " + str(año_sel) + "</p>", True)

    # Calcular componentes DuPont en el dataset filtrado
    at_col  = f"Activos Totales {año_sel}"
    pt_col  = f"Total de patrimonio {año_sel}"
    ve_col  = f"Total Ingreso Operativo {año_sel}"
    un_col  = f"Ganancia (Pérdida) Neta {año_sel}"

    dp_df = df[[at_col, pt_col, ve_col, un_col, "Sector_Corto"]].dropna()

    dp_df = dp_df.copy()
    dp_df["Margen_Neto_dp"]     = dp_df[un_col] / dp_df[ve_col]
    dp_df["Rotacion_dp"]        = dp_df[ve_col] / dp_df[at_col]
    dp_df["Apalancamiento_dp"]  = dp_df[at_col] / dp_df[pt_col]

    # Eliminar outliers extremos para el plot
    for col in ["Margen_Neto_dp", "Rotacion_dp", "Apalancamiento_dp"]:
        q_lo, q_hi = dp_df[col].quantile([0.05, 0.95])
        dp_df = dp_df[(dp_df[col] > q_lo) & (dp_df[col] < q_hi)]

    dp_sec = (dp_df.groupby("Sector_Corto")
              .agg(Margen=("Margen_Neto_dp", "median"),
                   Rotacion=("Rotacion_dp", "median"),
                   Apalancamiento=("Apalancamiento_dp", "median"))
              .reset_index()
              .sort_values("Margen", ascending=False))

    fig_dp = go.Figure()
    for comp, color, label in [
        ("Margen",       CLR["azul_claro"], "Margen Neto"),
        ("Rotacion",     CLR["verde_claro"],"Rotación Activos"),
        ("Apalancamiento", CLR["amarillo"], "Apalancamiento"),
    ]:
        fig_dp.add_trace(go.Bar(
            name=label,
            x=dp_sec["Sector_Corto"],
            y=dp_sec[comp],
            marker_color=color,
        ))

    fig_dp.update_layout(
        **LAYOUT_BASE, barmode="group", height=360, xaxis_tickangle=-35,
        title=dict(text="ROE = Margen Neto × Rotación de Activos × Multiplicador de Apalancamiento",
                   font_size=11, font_color="#C20E1A"),
    )
    st.plotly_chart(fig_dp, use_container_width=True)
    interp("📌 <b>Descomposición DuPont:</b> desglosa el ROE en tres motores. Sectores con "
           "alto apalancamiento (barras amarillas altas) generan rentabilidad mediante deuda, "
           "no eficiencia. Sectores como servicios profesionales (M) logran ROE alto vía "
           "<b>margen neto elevado</b> con bajo uso de activos — modelo de negocio superior.")

    with st.expander("Metodología de la descomposición DuPont del ROE"):
        ficha_indicador(**FICHAS["dupont"])

    # ── Clusters K-Means ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<p class='section-header'>Segmentación K-Means — perfil financiero de clusters</p>", True)

    if "kmeans_k3" in modelos and "pca_2d" in modelos and "scaler_financiero" in modelos:
        feats = modelos["features_modelo"]
        X_raw = df_raw[feats].fillna(0).values
        X_sc  = modelos["scaler_financiero"].transform(X_raw)
        X_pca = modelos["pca_2d"].transform(X_sc)

        cluster_labels_map = {0: "Cluster A — Moderado", 1: "Cluster B — Alto desempeño", 2: "Cluster C — Atípico"}
        clusters = modelos["kmeans_k3"].predict(X_sc)

        df_pca_plot = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": [cluster_labels_map.get(c, f"C{c}") for c in clusters],
            "Empresa": df_raw["Compañía"],
            "Sector":  df_raw["Sector_Corto"],
            "Z-Score": df_raw[f"Modelo Z-Score de Altman {año_sel}"].round(2),
        })

        fig_pca = px.scatter(
            df_pca_plot, x="PC1", y="PC2", color="Cluster",
            hover_data=["Empresa", "Sector", "Z-Score"],
            opacity=0.65,
            color_discrete_map={
                "Cluster A — Moderado":       CLR["azul_claro"],
                "Cluster B — Alto desempeño": CLR["verde_claro"],
                "Cluster C — Atípico":        CLR["amarillo"],
            },
        )
        fig_pca.update_traces(marker_size=4)
        apply_layout(fig_pca,
                     title="Proyección PCA 2D — agrupación K-Means (K=3) por perfil financiero",
                     height=420)
        st.plotly_chart(fig_pca, use_container_width=True)
        interp("📌 <b>Cluster B (Alto desempeño):</b> empresas con márgenes EBITDA >15% y bajo "
               "endeudamiento. <b>Cluster A (Moderado):</b> el segmento más amplio — perfil "
               "financiero estable pero conservador. <b>Cluster C (Atípico):</b> valores extremos "
               "que requieren validación individual antes de incluir en benchmarks sectoriales.")

        with st.expander("Metodología de segmentación K-Means y reducción PCA"):
            ficha_indicador(**FICHAS["kmeans"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SIMULADOR FINANCIERO (DuPont + Z-Score)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"""
    <div style='background:#FFFFFF;border:1px solid {CLR["gris_claro"]};
                border-radius:6px;padding:1rem 1.4rem;margin-bottom:1rem;
                border-left:4px solid {CLR["rojo"]};'>
      <div style='font-family:Montserrat,Arial,sans-serif;font-weight:700;
                  font-size:1.05rem;color:{CLR["gris_oscuro"]};margin-bottom:.25rem;'>
        Simulador financiero interactivo
      </div>
      <div style='font-size:.8rem;color:#555555;line-height:1.55;'>
        Ingrese los datos financieros de una empresa para calcular su Z-Score de Altman,
        la descomposición DuPont del ROE y obtener un diagnóstico automático de salud financiera.
        El sistema utiliza los modelos de machine learning entrenados sobre el universo de empresas colombianas.
      </div>
    </div>
    """, True)

    # ── Formulario de entrada ─────────────────────────────────────────────────
    st.markdown("<p class='section-header'>Datos financieros de la empresa</p>", True)

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        st.markdown("**📦 Balance General (millones COP)**")
        activos_totales  = st.number_input("Activos Totales",   min_value=0.0, value=5000.0, step=100.0, format="%.1f")
        pasivos_totales  = st.number_input("Pasivos Totales",   min_value=0.0, value=2500.0, step=100.0, format="%.1f")
        patrimonio       = st.number_input("Patrimonio Total",  min_value=0.0, value=2500.0, step=100.0, format="%.1f")
        capital_trabajo  = st.number_input("Capital de Trabajo",value=800.0,  step=50.0,   format="%.1f")
        ut_retenidas     = st.number_input("Utilidades Retenidas", value=500.0, step=50.0,  format="%.1f")

    with col_f2:
        st.markdown("**📈 Estado de Resultados (millones COP)**")
        ventas      = st.number_input("Ventas / Ingresos Operativos", min_value=0.0, value=8000.0, step=100.0, format="%.1f")
        utilidad_neta = st.number_input("Utilidad Neta",   value=400.0, step=50.0, format="%.1f")
        ebit          = st.number_input("EBIT (Ut. Operativa)", value=600.0, step=50.0, format="%.1f")
        ebitda        = st.number_input("EBITDA",          value=900.0, step=50.0, format="%.1f")

    with col_f3:
        st.markdown("**📑 Activos corrientes y pasivos CP**")
        activos_corrientes = st.number_input("Activos Corrientes",   value=2000.0, step=100.0, format="%.1f")
        pasivos_corrientes = st.number_input("Pasivos Corrientes",   value=1200.0, step=100.0, format="%.1f")
        inventarios        = st.number_input("Inventarios",          value=400.0,  step=50.0,  format="%.1f")
        efectivo           = st.number_input("Efectivo y equivalentes", value=300.0, step=50.0, format="%.1f")

        st.markdown("**🏷️ Contexto**")
        sector_sim  = st.selectbox("Sector CIIU", list(SECTOR_CORTO.values()))
        nombre_sim  = st.text_input("Nombre empresa (opcional)", value="Mi Empresa S.A.")

    calcular = st.button("⚡ Calcular diagnóstico financiero", type="primary",
                         use_container_width=True)

    st.markdown("---")

    if calcular:
        # ── Cálculos ─────────────────────────────────────────────────────────

        # Ratios básicos
        razon_corriente = activos_corrientes / pasivos_corrientes   if pasivos_corrientes else 0
        prueba_acida    = (activos_corrientes - inventarios) / pasivos_corrientes if pasivos_corrientes else 0
        razon_efectivo  = efectivo / pasivos_corrientes              if pasivos_corrientes else 0
        deuda_activos   = pasivos_totales  / activos_totales * 100   if activos_totales else 0
        deuda_capital   = pasivos_totales  / patrimonio * 100        if patrimonio else 0

        # ── Z-Score de Altman Z'' (mercados emergentes, no cotizadas) ─────────
        # X1 = Capital de Trabajo / Activos Totales
        X1 = capital_trabajo / activos_totales          if activos_totales else 0
        # X2 = Utilidades Retenidas / Activos Totales
        X2 = ut_retenidas   / activos_totales           if activos_totales else 0
        # X3 = EBIT / Activos Totales
        X3 = ebit            / activos_totales           if activos_totales else 0
        # X4 = Patrimonio (valor en libros) / Pasivos Totales
        X4 = patrimonio      / pasivos_totales           if pasivos_totales else 0

        zscore = 6.56 * X1 + 3.26 * X2 + 6.72 * X3 + 1.05 * X4

        # Zona
        if zscore > 2.60:
            zona_txt   = "🟢 Zona Segura"
            zona_color_val = CLR["verde"]
            zona_class = "kpi-safe"
            zona_bg_val= "rgba(30,132,73,.2)"
        elif zscore >= 1.10:
            zona_txt   = "🟡 Zona de Alerta"
            zona_color_val = CLR["amarillo"]
            zona_class = "kpi-warn"
            zona_bg_val= "rgba(212,172,13,.2)"
        else:
            zona_txt   = "🔴 Zona de Quiebra"
            zona_color_val = CLR["rojo"]
            zona_class = "kpi-danger"
            zona_bg_val= "rgba(192,57,43,.2)"

        # ── DuPont ────────────────────────────────────────────────────────────
        margen_neto_dp  = utilidad_neta / ventas         if ventas else 0
        rotacion_dp     = ventas        / activos_totales if activos_totales else 0
        apalancamiento_dp = activos_totales / patrimonio  if patrimonio else 0
        roe_dupont      = margen_neto_dp * rotacion_dp * apalancamiento_dp

        roa_calc  = utilidad_neta / activos_totales * 100  if activos_totales else 0
        roe_calc  = utilidad_neta / patrimonio * 100        if patrimonio else 0
        mn_calc   = utilidad_neta / ventas * 100            if ventas else 0
        ebitda_mg = ebitda        / ventas * 100            if ventas else 0

        # ── Predicción con modelo ML ──────────────────────────────────────────
        ml_pred = None
        ml_prob = None
        if "random_forest_zscore" in modelos:
            feats_ml  = modelos["features_modelo"]
            # Construir vector con el año más reciente de cada variable
            vals_sim = {}
            for f in feats_ml:
                base = f.rsplit(" ", 1)[0]  # quitar el año
                if "ROA"              in f: vals_sim[f] = roa_calc
                elif "ROE"            in f: vals_sim[f] = roe_calc
                elif "Ebitda"         in f: vals_sim[f] = ebitda_mg
                elif "Operacional"    in f: vals_sim[f] = ebit/ventas*100 if ventas else 0
                elif "Margen Neto"    in f: vals_sim[f] = mn_calc
                elif "Razón De Liquidez" in f: vals_sim[f] = razon_corriente
                elif "Prueba Ácida"   in f: vals_sim[f] = prueba_acida
                elif "Razón De Efectivo" in f: vals_sim[f] = razon_efectivo
                elif "Z-Score"        in f: vals_sim[f] = zscore
                elif "Deuda / Activos" in f: vals_sim[f] = deuda_activos
                elif "Deuda/Capital"  in f: vals_sim[f] = deuda_capital
                elif "Deuda Neta / EBITDA" in f: vals_sim[f] = 0
                else: vals_sim[f] = 0
            X_sim    = np.array([[vals_sim[f] for f in feats_ml]])
            X_sim_sc = modelos["scaler_financiero"].transform(X_sim)
            pred_enc = modelos["random_forest_zscore"].predict(X_sim_sc)[0]
            ml_prob  = modelos["random_forest_zscore"].predict_proba(X_sim_sc)[0]
            le       = modelos["label_encoder_zscore"]
            ml_pred  = le.inverse_transform([pred_enc])[0]

        # ══════════════ RESULTADOS ════════════════════════════════════════════
        st.markdown(f"""
        <div style='background:#FFFFFF;border:1.5px solid {zona_color_val};border-left:4px solid {zona_color_val};
                    border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;
                    text-align:center;'>
          <div style='font-size:.85rem;color:#333333;margin-bottom:.4rem;'>
            Diagnóstico financiero — <b>{nombre_sim}</b> | Sector: {sector_sim}
          </div>
          <div style='font-size:2rem;font-weight:700;color:{zona_color_val};'>
            {zona_txt}
          </div>
          <div style='font-size:1.4rem;font-weight:600;color:#333333;margin-top:.3rem;'>
            Z-Score Altman Z'' = {zscore:.3f}
          </div>
          <div style='font-size:.8rem;color:#333333;margin-top:.4rem;'>
            Fórmula: Z'' = 6.56·X₁ + 3.26·X₂ + 6.72·X₃ + 1.05·X₄
          </div>
        </div>
        """, True)

        # ── Grid resultados ───────────────────────────────────────────────────
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            st.markdown("<p class='section-header'>Componentes Z-Score</p>", True)
            comp_df = pd.DataFrame({
                "Variable": ["X₁ = CT / Activos", "X₂ = Ut.Ret / Activos",
                             "X₃ = EBIT / Activos", "X₄ = Patrim. / Pasivos"],
                "Valor":    [round(X1,4), round(X2,4), round(X3,4), round(X4,4)],
                "Pond.":    [6.56, 3.26, 6.72, 1.05],
            })
            comp_df["Contribución"] = (comp_df["Valor"] * comp_df["Pond."]).round(4)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Gauge Z-Score
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(zscore, 3),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Z-Score de Altman", "font": {"color": "#C20E1A", "size": 13}},
                number={"font": {"color": "#1A1A1A", "size": 28}},
                gauge={
                    "axis": {"range": [-2, 6], "tickcolor": "#888888"},
                    "bar":  {"color": zona_color_val, "thickness": 0.25},
                    "bgcolor": CLR["tarjeta"],
                    "bordercolor": CLR["azul_medio"],
                    "steps": [
                        {"range": [-2, 1.10], "color": "rgba(192,57,43,.3)"},
                        {"range": [1.10, 2.60], "color": "rgba(212,172,13,.3)"},
                        {"range": [2.60, 6],  "color": "rgba(30,132,73,.3)"},
                    ],
                    "threshold": {
                        "line": {"color": "#333333", "width": 2},
                        "thickness": 0.75,
                        "value": zscore,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                font_color="#333333", height=230,
                margin=dict(t=30, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_res2:
            st.markdown("<p class='section-header'>Indicadores calculados</p>", True)

            indicadores = [
                ("💧 Razón Corriente",  f"{razon_corriente:.2f}x",
                 "✅ OK" if razon_corriente >= 1.5 else ("⚠️" if razon_corriente >= 1.0 else "🚨 Riesgo")),
                ("🧪 Prueba Ácida",     f"{prueba_acida:.2f}x",
                 "✅ OK" if prueba_acida >= 1.0 else "⚠️ Vigilar"),
                ("📈 ROA",              f"{roa_calc:.2f}%", ""),
                ("💼 ROE",              f"{roe_calc:.2f}%", ""),
                ("📊 Margen Neto",      f"{mn_calc:.2f}%",  ""),
                ("⚡ Margen EBITDA",    f"{ebitda_mg:.2f}%",""),
                ("⚖️ Deuda/Activos",    f"{deuda_activos:.1f}%",
                 "✅ Bajo" if deuda_activos < 40 else ("⚠️ Moderado" if deuda_activos < 60 else "🚨 Alto")),
                ("🔗 Deuda/Capital",    f"{deuda_capital:.1f}%", ""),
            ]

            for nombre, valor, estado in indicadores:
                col_a, col_b, col_c = st.columns([2.2, 1.1, 0.9])
                col_a.markdown(f"<span style='color:#333333;font-size:.82rem;'>{nombre}</span>", True)
                col_b.markdown(f"<span style='color:#1A1A1A;font-weight:600;font-size:.9rem;'>{valor}</span>", True)
                col_c.markdown(f"<span style='font-size:.75rem;'>{estado}</span>", True)

        with col_res3:
            st.markdown("<p class='section-header'>Descomposición DuPont</p>", True)

            # Velocímetro DuPont
            dp_val = {
                "Margen Neto": margen_neto_dp * 100,
                "Rotación Activos": rotacion_dp,
                "Apalancamiento": apalancamiento_dp,
                "ROE (DuPont)": roe_dupont * 100,
            }

            fig_dp_bar = go.Figure()
            colors_dp = [CLR["azul_claro"], CLR["verde_claro"], CLR["amarillo"], CLR["rojo"]]
            for i, (k, v) in enumerate(dp_val.items()):
                fig_dp_bar.add_trace(go.Bar(
                    name=k, x=[k], y=[round(v, 3)],
                    marker_color=colors_dp[i],
                    text=[f"{v:.2f}" + ("%" if "ROE" in k or "Margen" in k else "x")],
                    textposition="outside", textfont_size=11,
                ))
            apply_layout(fig_dp_bar, height=220)
            fig_dp_bar.update_layout(showlegend=False, barmode="group")
            st.plotly_chart(fig_dp_bar, use_container_width=True)

            # Fórmula DuPont
            st.markdown(f"""
            <div class='interp-box' style='text-align:center;font-size:.83rem;margin-top:.5rem;'>
              <b>ROE = Margen × Rotación × Apalancamiento</b><br>
              <code style='color:{CLR["verde_claro"]};'>{mn_calc:.2f}%</code> ×
              <code style='color:{CLR["azul_claro"]};'>{rotacion_dp:.3f}x</code> ×
              <code style='color:{CLR["amarillo"]};'>{apalancamiento_dp:.3f}x</code> =
              <code style='color:#1A1A1A;font-size:1rem;font-weight:700;'>{roe_dupont*100:.2f}%</code>
            </div>
            """, True)

            # Motor DuPont
            if apalancamiento_dp >= rotacion_dp and apalancamiento_dp >= abs(margen_neto_dp):
                motor = "⚠️ El ROE se sustenta principalmente en <b>deuda (apalancamiento)</b>, no en eficiencia operativa. Revisar estructura financiera."
            elif rotacion_dp >= abs(margen_neto_dp):
                motor = "✅ El ROE se impulsa por <b>eficiencia en el uso de activos</b> (rotación alta). Modelo operativo saludable."
            else:
                motor = "✅ El ROE se explica por <b>márgenes netos sólidos</b>. Alta rentabilidad sobre ventas."
            interp(motor)

        # ── Predicción ML ─────────────────────────────────────────────────────
        if ml_pred and ml_prob is not None:
            st.markdown("---")
            st.markdown("<p class='section-header'>Clasificación con Random Forest (modelo entrenado)</p>", True)

            le_classes = modelos["label_encoder_zscore"].classes_
            prob_dict  = dict(zip(le_classes, ml_prob))

            col_ml1, col_ml2 = st.columns([1, 1.5])
            with col_ml1:
                st.markdown(f"""
                <div style='background:#FFFFFF;border:1px solid {CLR["gris_claro"]};
                            border-radius:10px;padding:1rem;text-align:center;'>
                  <div style='color:#C20E1A;font-size:.8rem;'>Clasificación RF (Accuracy 99.3%)</div>
                  <div style='color:#1A1A1A;font-size:1.4rem;font-weight:700;margin:.4rem 0;'>
                    {ml_pred}
                  </div>
                  <div style='color:#333333;font-size:.75rem;'>
                    Basado en 48 indicadores financieros × 4 años
                  </div>
                </div>
                """, True)

            with col_ml2:
                fig_prob = go.Figure(go.Bar(
                    x=list(prob_dict.values()),
                    y=list(prob_dict.keys()),
                    orientation="h",
                    marker_color=[zona_color(z) for z in prob_dict.keys()],
                    text=[f"{p:.1%}" for p in prob_dict.values()],
                    textposition="outside", textfont_size=10,
                ))
                fig_prob.update_layout(**LAYOUT_BASE, height=160,
                                       xaxis_tickformat=".0%",
                                       title=dict(text="Probabilidades por clase (Random Forest)",
                                                  font_size=11, font_color="#C20E1A"))
                st.plotly_chart(fig_prob, use_container_width=True)

        # ── Recomendaciones ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("<p class='section-header'>Diagnóstico y recomendaciones automáticas</p>", True)

        recs = []
        if zscore > 2.60:
            recs.append("✅ <b>Salud financiera sólida.</b> La empresa mantiene un equilibrio "
                        "adecuado entre liquidez, solvencia y rentabilidad.")
        elif zscore >= 1.10:
            recs.append("⚠️ <b>Zona de alerta.</b> La empresa no está en crisis inmediata pero "
                        "presenta señales de advertencia que requieren monitoreo trimestral.")
        else:
            recs.append("🚨 <b>Zona de quiebra.</b> Se detectan señales severas de dificultad "
                        "financiera. Se recomienda intervención urgente en la estructura de capital.")

        if razon_corriente < 1.0:
            recs.append("🔴 <b>Liquidez crítica:</b> la razón corriente es inferior a 1.0x. "
                        "La empresa no puede cubrir sus obligaciones de corto plazo con activos "
                        "disponibles. Prioridad: mejorar el capital de trabajo.")
        elif razon_corriente < 1.5:
            recs.append("🟡 <b>Liquidez ajustada:</b> razón corriente entre 1.0–1.5x. "
                        "Hay margen, pero limitado. Revisar ciclo de caja y políticas de crédito.")

        if deuda_activos > 60:
            recs.append("🔴 <b>Endeudamiento alto (>60%):</b> más de la mitad de los activos "
                        "están financiados con deuda. Riesgo de refinanciamiento y costo "
                        "financiero elevado. Considerar capitalización o reducción de pasivos.")
        elif deuda_activos > 40:
            recs.append("🟡 <b>Endeudamiento moderado (40–60%):</b> nivel de apalancamiento "
                        "manejable pero que requiere control. Monitorear la cobertura de intereses.")

        if roa_calc < 3:
            recs.append("🟡 <b>ROA bajo (<3%):</b> la empresa genera poca utilidad por peso "
                        "invertido en activos. Revisar eficiencia operativa y posible sobredimensionamiento.")

        if apalancamiento_dp > 3:
            recs.append("⚠️ <b>ROE impulsado por apalancamiento:</b> el multiplicador DuPont "
                        "indica que la rentabilidad sobre el patrimonio depende más de la deuda "
                        "que de la eficiencia. Riesgo en escenarios de alza de tasas.")

        for rec in recs:
            interp(rec)

        # Benchmarks vs. muestra
        st.markdown("<p class='section-header'>Comparación vs. benchmarks del sector ({año_sel})</p>", True)
        bench_sec = df_raw[df_raw["Sector_Corto"] == sector_sim]
        bench_all = df_raw

        bench_cols = {
            "Razón Corriente (x)": f"Razón De Liquidez (x) {año_sel}",
            "ROA (%)":             f"Rendimiento Sobre Los Activos (ROA) (%) {año_sel}",
            "Margen Neto (%)":     f"Margen Neto (%) {año_sel}",
            "Z-Score":             f"Modelo Z-Score de Altman {año_sel}",
            "Deuda/Activos (%)":   f"Deuda / Activos (%) {año_sel}",
        }
        vals_empresa = {
            "Razón Corriente (x)": razon_corriente,
            "ROA (%)":             roa_calc,
            "Margen Neto (%)":     mn_calc,
            "Z-Score":             zscore,
            "Deuda/Activos (%)":   deuda_activos,
        }

        bench_rows = []
        for etiqueta, col in bench_cols.items():
            if col in df_raw.columns:
                bench_rows.append({
                    "Indicador":        etiqueta,
                    "Tu empresa":       round(vals_empresa[etiqueta], 2),
                    "Mediana sector":   round(bench_sec[col].median(), 2)  if len(bench_sec) else None,
                    "Mediana Colombia": round(bench_all[col].median(), 2),
                })

        df_bench = pd.DataFrame(bench_rows)
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Bar(
            name="Tu empresa",
            x=df_bench["Indicador"], y=df_bench["Tu empresa"],
            marker_color=CLR["azul_claro"],
        ))
        fig_bench.add_trace(go.Scatter(
            name="Mediana del sector",
            x=df_bench["Indicador"], y=df_bench["Mediana sector"],
            mode="lines+markers", line=dict(color=CLR["verde_claro"], width=2.5, dash="dot"),
            marker_size=9,
        ))
        fig_bench.add_trace(go.Scatter(
            name="Mediana Colombia",
            x=df_bench["Indicador"], y=df_bench["Mediana Colombia"],
            mode="lines+markers", line=dict(color=CLR["amarillo"], width=2, dash="dash"),
            marker_size=7,
        ))
        apply_layout(fig_bench, height=340)
        st.plotly_chart(fig_bench, use_container_width=True)
        interp("Las líneas punteadas representan los <b>benchmarks de referencia</b>: "
               "el mediano del sector seleccionado y el mediano del universo de 1.739 empresas "
               "colombianas. Las barras muestran el valor calculado para tu empresa.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — GLOSARIO COMPLETO DE INDICADORES
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown(f"""
    <div style='background:#FFFFFF;border:1px solid {CLR["gris_claro"]};
                border-radius:6px;padding:1rem 1.4rem;margin-bottom:1.2rem;
                border-left:4px solid {CLR["rojo"]};'>
      <div style='font-family:Montserrat,Arial,sans-serif;font-weight:700;
                  font-size:1.05rem;color:{CLR["gris_oscuro"]};margin-bottom:.25rem;'>
        Glosario de indicadores financieros
      </div>
      <div style='font-size:.8rem;color:#555555;line-height:1.55;'>
        Fichas metodológicas con definiciones, fórmulas e interpretación de referencia para todos
        los indicadores del dashboard. Diseñado para audiencias especializadas y no especializadas.
      </div>
    </div>
    """, True)

    # ── Buscador ──────────────────────────────────────────────────────────────
    buscar = st.text_input(
        "🔍 Buscar indicador",
        placeholder="Escribe el nombre de un indicador (ej: ROA, liquidez, Z-Score…)",
    )

    CATEGORIAS = {
        "Indicadores de Liquidez": ["razon_corriente", "prueba_acida", "razon_efectivo"],
        "Indicadores de Rentabilidad": ["roa", "roe", "margen_neto", "margen_ebitda"],
        "Indicadores de Endeudamiento": ["deuda_activos", "deuda_capital"],
        "Modelo de Riesgo Financiero": ["zscore"],
        "Análisis Avanzado": ["dupont", "kmeans"],
    }

    # Filtrar por búsqueda
    termino = buscar.lower().strip()
    for cat_titulo, claves in CATEGORIAS.items():
        claves_filtradas = [
            k for k in claves
            if not termino
            or termino in FICHAS[k]["nombre"].lower()
            or termino in FICHAS[k]["tag"].lower()
            or termino in FICHAS[k]["simple"].lower()
        ]
        if not claves_filtradas:
            continue

        # Encabezado de categoría
        c_r = CLR["rojo"]
        c_g = CLR["gris_oscuro"]
        st.markdown(
            f"<p style='font-family:Montserrat,Arial,sans-serif;font-size:.85rem;"
            f"font-weight:600;color:{c_g};border-bottom:2px solid "
            f"{c_r};padding-bottom:.35rem;margin-bottom:.9rem;"
            f"letter-spacing:.01em;'>{cat_titulo}</p>",
            unsafe_allow_html=True
        )

        # ── SOLUCIÓN React removeChild: renderizar TODAS las fichas de la
        # categoría en UNA SOLA llamada st.markdown() con CSS Grid nativo.
        # Esto evita que múltiples st.markdown() dentro de st.columns()
        # provoquen conflictos de reconciliación en el DOM de React.
        fichas_html = "".join(ficha_as_html(**FICHAS[k]) for k in claves_filtradas)
        grid_html = (
            f"<div style='display:grid;"
            f"grid-template-columns:repeat(auto-fill,minmax(340px,1fr));"
            f"gap:.8rem;margin-bottom:1.2rem;'>"
            + fichas_html
            + "</div>"
        )
        st.markdown(grid_html, unsafe_allow_html=True)

    # ── Tabla resumen de umbrales ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<p class='section-header'>Tabla resumen de umbrales de referencia</p>", True)
    interp(
        "Esta tabla consolida los rangos de interpretación de todos los indicadores. "
        "Los umbrales son referencias orientadoras; el contexto sectorial y el tamaño "
        "de la empresa siempre deben considerarse para una interpretación correcta."
    )

    tabla_umbrales = pd.DataFrame([
        {"Indicador": "Razón Corriente",      "🟢 Bueno":  "≥ 1.5x",   "🟡 Atención": "1.0 – 1.5x",  "🔴 Riesgo":  "< 1.0x"},
        {"Indicador": "Prueba Ácida",         "🟢 Bueno":  "≥ 1.0x",   "🟡 Atención": "0.8 – 1.0x",  "🔴 Riesgo":  "< 0.8x"},
        {"Indicador": "Razón de Efectivo",    "🟢 Bueno":  "≥ 0.5x",   "🟡 Atención": "0.2 – 0.5x",  "🔴 Riesgo":  "< 0.2x"},
        {"Indicador": "ROA (%)",              "🟢 Bueno":  "> 10%",     "🟡 Atención": "2 – 10%",      "🔴 Riesgo":  "< 2%"},
        {"Indicador": "ROE (%)",              "🟢 Bueno":  "> 15%",     "🟡 Atención": "3 – 15%",      "🔴 Riesgo":  "< 3%"},
        {"Indicador": "Margen Neto (%)",      "🟢 Bueno":  "> 10%",     "🟡 Atención": "1 – 10%",      "🔴 Riesgo":  "< 1%"},
        {"Indicador": "Margen EBITDA (%)",    "🟢 Bueno":  "> 15%",     "🟡 Atención": "3 – 15%",      "🔴 Riesgo":  "< 3%"},
        {"Indicador": "Deuda / Activos (%)",  "🟢 Bueno":  "< 40%",     "🟡 Atención": "40 – 60%",     "🔴 Riesgo":  "> 60%"},
        {"Indicador": "Deuda / Capital (%)",  "🟢 Bueno":  "< 80%",     "🟡 Atención": "80 – 150%",    "🔴 Riesgo":  "> 150%"},
        {"Indicador": "Z-Score de Altman",    "🟢 Bueno":  "> 2.60",    "🟡 Atención": "1.10 – 2.60",  "🔴 Riesgo":  "< 1.10"},
    ])
    st.dataframe(
        tabla_umbrales,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Indicador":    st.column_config.TextColumn("Indicador",        width="medium"),
            "🟢 Bueno":    st.column_config.TextColumn("🟢 Zona sana",     width="small"),
            "🟡 Atención": st.column_config.TextColumn("🟡 Zona atención", width="small"),
            "🔴 Riesgo":   st.column_config.TextColumn("🔴 Zona riesgo",   width="small"),
        },
    )

    # ── Glosario de términos financieros generales ────────────────────────────
    st.markdown("---")
    st.markdown("<p class='section-header'>Glosario de términos financieros clave</p>", True)

    TERMINOS = [
        ("Activos Totales",
         "Todo lo que tiene la empresa: dinero en banco, inventarios, máquinas, "
         "edificios, derechos sobre clientes, etc. Es el 'patrimonio total' de la "
         "empresa en términos contables."),
        ("Pasivos Totales",
         "Todo lo que debe la empresa: créditos bancarios, facturas por pagar a "
         "proveedores, deudas con empleados, impuestos pendientes, etc."),
        ("Patrimonio",
         "Lo que le queda a los dueños si la empresa pagara todas sus deudas. "
         "Patrimonio = Activos − Pasivos. También llamado capital contable o "
         "valor en libros del negocio."),
        ("Capital de Trabajo",
         "La 'caja operativa' del negocio: Activos Corrientes − Pasivos Corrientes. "
         "Un capital de trabajo positivo significa que la empresa puede operar día "
         "a día sin necesitar financiamiento extra."),
        ("EBIT",
         "Ganancias Antes de Intereses e Impuestos (Earnings Before Interest and "
         "Taxes). Mide el resultado de las operaciones principales sin contar cómo "
         "está financiada la empresa ni los impuestos."),
        ("EBITDA",
         "EBIT más depreciación y amortización. Aproxima el flujo de caja generado "
         "por las operaciones. Ampliamente usado para valorar empresas y comparar "
         "negocios de distintos países o sectores."),
        ("Utilidades Retenidas",
         "Ganancias acumuladas que la empresa no repartió como dividendos. Son "
         "una fuente de financiamiento propia; empresas con altas utilidades "
         "retenidas suelen ser más robustas financieramente."),
        ("Mediana vs. Media",
         "En este dashboard se usa la mediana (valor central) en lugar de la media "
         "(promedio) porque los datos financieros tienen distribuciones asimétricas. "
         "Una empresa con ROA del 2.000% elevaría el promedio pero no la mediana, "
         "dando una imagen más realista del sector."),
        ("Sector CIIU",
         "Clasificación Industrial Internacional Uniforme. Sistema de la ONU para "
         "clasificar las actividades económicas. La letra indica la macrosección "
         "(C=Manufactura, G=Comercio, F=Construcción, etc.)."),
        ("Clasificación Decreto 957",
         "Norma colombiana que clasifica las empresas por tamaño según sus ingresos "
         "anuales: Microempresa, Pequeña, Mediana y Grande. Los umbrales varían "
         "por sector (manufactura, servicios y comercio)."),
    ]

    tg1, tg2 = st.columns(2)
    for i, (termino, defn) in enumerate(TERMINOS):
        col_target = tg1 if i % 2 == 0 else tg2
        with col_target:
            st.markdown(f"""
            <div style='background:{CLR["tarjeta"]};border-radius:8px;padding:.8rem 1rem;
                        border-left:3px solid #C20E1A;margin-bottom:.6rem;'>
              <div style='color:#C20E1A;font-weight:600;font-size:.85rem;margin-bottom:.35rem;'>
                {termino}
              </div>
              <div style='color:#333333;font-size:.8rem;line-height:1.65;'>{defn}</div>
            </div>
            """, True)

    # ── Referencias metodológicas ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<p class='section-header'>Referencias metodológicas</p>", True)
    refs = [
        ("Altman, E. (1968)", "Financial ratios, discriminant analysis and the prediction of "
         "corporate bankruptcy. <i>Journal of Finance</i>, 23(4), 589–609."),
        ("Altman, Hartzell & Peck (1995)", "Emerging markets corporate bonds: a scoring system. "
         "<i>Salomon Brothers</i>. Versión Z'' para mercados emergentes y empresas no cotizadas."),
        ("Brigham & Houston (2014)", "<i>Fundamentals of Financial Management</i>. "
         "Cengage Learning. Referencia estándar para ROA, ROE y análisis DuPont."),
        ("Chapman et al. (2000)", "CRISP-DM 1.0: Step-by-step data mining guide. "
         "Metodología adoptada para el desarrollo del presente trabajo de grado."),
        ("Damodaran, A. (2012)", "<i>Investment Valuation</i>. Wiley. "
         "Referencia para EBITDA como proxy de generación de caja operativa."),
        ("Few, S. (2012)", "<i>Show Me the Numbers</i>. Analytics Press. "
         "Principios de diseño de dashboards y uso de mediana vs. media en datos financieros."),
        ("Tufte, E. (2001)", "<i>The Visual Display of Quantitative Information</i>. "
         "Graphics Press. Principios de integridad visual aplicados al diseño del dashboard."),
        ("Van Horne & Wachowicz (2010)", "<i>Fundamentals of Financial Management</i>. "
         "Pearson. Referencia para indicadores de liquidez y solvencia."),
    ]
    for autor, desc in refs:
        st.markdown(f"""
        <div style='margin-bottom:.5rem;font-size:.8rem;color:#333333;line-height:1.6;'>
          <b style='color:#C20E1A;'>{autor}.</b> {desc}
        </div>
        """, True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='border-top:1px solid {CLR["gris_claro"]};margin:2rem 0 .8rem;
            padding-top:.9rem;display:flex;align-items:center;
            justify-content:space-between;flex-wrap:wrap;gap:.5rem;'>
  <div style='font-size:.72rem;color:#555555;line-height:1.75;'>
    <span style='color:{CLR["rojo"]};font-family:Montserrat,Arial,sans-serif;
                 font-weight:700;'>Universidad del Valle</span>
    &nbsp;·&nbsp; Maestría en Analítica e Inteligencia de Negocios<br>
    Trabajo de grado — Analítica descriptiva del desempeño financiero del sector real colombiano (2021–2024)
  </div>
  <div style='font-size:.7rem;color:#555555;text-align:right;line-height:1.7;'>
    Modelo Z-Score: Altman Z&#8242;&#8242; (1995) · K-Means K=3 · Random Forest Acc. 99.3% · Metodología CRISP-DM<br>
    Fuente: EMIS Platform · 1.739 empresas · 18 sectores CIIU · COP millones
  </div>
</div>
""", unsafe_allow_html=True)
