"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Analítica Descriptiva — Sector Real Colombiano 2021–2024 (V12)            ║
║  Trabajo de Grado | Maestría en Analítica e Inteligencia de Negocios       ║
║  Stack: Streamlit · Plotly · scikit-learn · pandas                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import base64, io, os, re, warnings
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

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

# ─────────────────────────────────────────────────────────────────────────────
# PALETA — Manual de Identidad Visual Corporativa, Universidad del Valle (2024)
#   · Color principal: Rojo Univalle #C20E1A (presente en todo el diseño).
#   · Colores funcionales: #841F1C y neutrales de la paleta web #636363 / #CFCFCF.
#   · Colores expresivos (#599876, #DC911B, #3D5AA2, #42A8BF, #523888): solo como
#     acentos en los datos (semáforo Z'', series), nunca en la interfaz; el manual
#     limita su uso a ≈ 10 % del espacio.
#   · Tramas (100/75/50/25 %): el manual las admite en gráficos estadísticos.
# ─────────────────────────────────────────────────────────────────────────────
def trama(hex_color, pct):
    """Trama de un color del manual al pct % (mezcla con blanco), como en el capítulo 'Tramas o tonos de color'."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    f = pct / 100
    return "#{:02X}{:02X}{:02X}".format(*(round(255 - (255 - c) * f) for c in (r, g, b)))


CLR = {
    # Principal y funcionales (interfaz y datos)
    "rojo":         "#C20E1A",  # Rojo Univalle — color principal de marca
    "granate":      "#841F1C",  # Color funcional
    "gris_oscuro":  "#636363",  # Neutral de la paleta web — texto secundario y series neutras
    "gris_medio":   "#636363",  # Alias del neutral oscuro
    "gris_claro":   "#CFCFCF",  # Neutral de la paleta web — bordes de 1 px y separadores
    "fondo":        trama("#CFCFCF", 25),  # Trama 25 % del neutral claro (#F3F3F3) — fondo general
    "tarjeta":      "#FFFFFF",  # Blanco — superficie de tarjetas (espacio en blanco)
    "azul_oscuro":  "#1A1A1A",  # Negro tipográfico — títulos y texto principal
    # Expresivos del manual — solo para codificar datos
    "verde":        "#599876",  # Semáforo: Zona Segura
    "amarillo":     "#DC911B",  # Semáforo: Zona Alerta
    "azul_claro":   "#3D5AA2",  # Serie de datos principal (azul expresivo)
    "verde_claro":  trama("#599876", 75),  # Trama 75 % del verde para series secundarias
    "cian":         "#42A8BF",
    "morado":       "#523888",
    "azul_medio":   "#CFCFCF",
}

# Tipografía — el manual define Helvetica Neue (Medium) como fuente institucional y,
# para medios digitales, fuentes sin serifa seguras como Arial o Verdana.
FUENTE_TITULOS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
FUENTE_TEXTO = "'Helvetica Neue', Helvetica, Arial, sans-serif"

st.markdown(f"""
<style>
  /* ═══ SISTEMA GRÁFICO — UNIVERSIDAD DEL VALLE · diseño plano, sin sombras ═══ */

  :root {{
      --rojo:{CLR['rojo']}; --granate:{CLR['granate']}; --gris:{CLR['gris_oscuro']};
      --gris-claro:{CLR['gris_claro']}; --fondo:{CLR['fondo']}; --negro:{CLR['azul_oscuro']};
  }}

  /* Sin sombras en ningún contenedor */
  .stApp *, .stApp *::before, .stApp *::after {{ box-shadow:none !important; text-shadow:none !important; }}

  /* Base y tipografía */
  .stApp {{ background-color:var(--fondo); font-family:{FUENTE_TEXTO}; color:var(--negro); }}
  .stApp p, .stApp li, .stApp label {{ font-family:{FUENTE_TEXTO}; }}
  .block-container {{ padding-top:3.2rem; padding-bottom:3rem; max-width:1400px; }}
  h1, h2, h3, h4 {{ font-family:{FUENTE_TITULOS} !important; color:var(--negro); }}

  /* ── Sidebar ─────────────────────────────────────────────────────────── */
  section[data-testid="stSidebar"] {{ background-color:#FFFFFF; border-right:1px solid var(--gris-claro); }}
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] span:not([data-baseweb="tag"] span),
  section[data-testid="stSidebar"] div.stMarkdown {{ color:var(--gris) !important; }}

  /* ── Controles de formulario: borde de 1 px, esquinas mínimas ───────── */
  [data-baseweb="select"] > div,
  .stNumberInput input, .stTextInput input {{
      background:#FFFFFF !important; border:1px solid var(--gris-claro) !important; border-radius:2px !important;
      color:var(--negro) !important; font-size:.82rem !important;
  }}
  [data-baseweb="select"] > div:focus-within,
  .stNumberInput input:focus, .stTextInput input:focus {{ border-color:var(--rojo) !important; }}
  [data-baseweb="select"] [data-baseweb="value"], [data-baseweb="select"] span,
  [data-baseweb="select"] input {{ color:var(--negro) !important; font-size:.82rem !important; }}
  [data-baseweb="select"] input::placeholder {{ color:var(--gris) !important; }}
  [data-baseweb="tag"] {{ background:#FFFFFF !important; border:1px solid var(--rojo) !important; border-radius:2px !important; }}
  [data-baseweb="tag"] span {{ color:var(--rojo) !important; font-size:.76rem !important; font-weight:600; }}
  [data-baseweb="tag"] svg {{ fill:var(--rojo) !important; color:var(--rojo) !important; }}
  [data-baseweb="popover"], [data-baseweb="menu"] {{ background:#FFFFFF !important; border:1px solid var(--gris-claro) !important; border-radius:2px !important; }}
  [data-baseweb="menu"] li, [data-baseweb="option"] {{ color:var(--negro) !important; font-size:.82rem !important; background:#FFFFFF !important; }}
  [data-baseweb="menu"] li:hover, [data-baseweb="menu"] [aria-selected="true"] {{ background:var(--fondo) !important; color:var(--rojo) !important; }}
  /* Streamlit ≥ 1.50 (react-aria): mismo estilo para selectores y etiquetas */
  .stSelectbox .react-aria-ComboBox > [role="group"],
  .stMultiSelect .react-aria-ComboBox > [role="group"] {{
      background:#FFFFFF !important; border:1px solid var(--gris-claro) !important; border-radius:2px !important; }}
  .stSelectbox .react-aria-ComboBox > [role="group"]:focus-within,
  .stMultiSelect .react-aria-ComboBox > [role="group"]:focus-within {{ border-color:var(--rojo) !important; }}
  .stSelectbox input, .stMultiSelect input {{ color:var(--negro) !important; font-size:.82rem !important; }}
  [data-tag] {{ background:#FFFFFF !important; border:1px solid var(--rojo) !important; border-radius:2px !important;
                color:var(--rojo) !important; }}
  [data-tag] span {{ color:var(--rojo) !important; font-size:.76rem !important; font-weight:600; }}
  [data-tag] button, [data-tag] svg {{ color:var(--rojo) !important; }}
  [role="listbox"] {{ background:#FFFFFF !important; border:1px solid var(--gris-claro) !important; border-radius:2px !important; }}
  [role="listbox"] [role="option"] {{ color:var(--negro) !important; font-size:.82rem !important; }}
  [role="listbox"] [role="option"]:hover, [role="listbox"] [role="option"][data-focused] {{ background:var(--fondo) !important; color:var(--rojo) !important; }}
  .stCheckbox [data-testid="stCheckbox"] span, [data-baseweb="checkbox"] > div {{ border-radius:2px !important; }}

  /* ── Encabezado ──────────────────────────────────────────────────────── */
  .dash-header {{ background:#FFFFFF; border:1px solid var(--gris-claro); border-top:6px solid var(--rojo); margin-bottom:1.6rem; }}
  .dash-header-stripe {{ display:none; }}
  .dash-header-body {{ padding:1.4rem 1.8rem 1.2rem; display:flex; align-items:flex-start; gap:1.4rem; }}
  .dash-header-logo {{ flex-shrink:0; }}
  .dash-header-texts {{ flex:1; }}
  .dash-header-pretitle {{ font-family:{FUENTE_TITULOS}; font-size:.7rem; font-weight:600; letter-spacing:.08em;
                           text-transform:uppercase; color:var(--rojo); margin-bottom:.35rem; }}
  .dash-header h1 {{ font-family:{FUENTE_TITULOS}; font-size:1.5rem; font-weight:700; color:var(--negro);
                     margin:0 0 .3rem; padding:0; line-height:1.3; }}
  .dash-header h1 span.titulo-rojo {{ color:var(--rojo); }}
  .dash-header p {{ font-size:.85rem; color:var(--gris); margin:0; line-height:1.5; }}
  .dash-header-meta {{ display:flex; flex-wrap:wrap; gap:2rem; margin-top:.8rem; padding-top:.7rem; border-top:1px solid var(--gris-claro); }}
  .dash-header-meta-item {{ font-size:.74rem; color:var(--gris); }}
  .dash-header-meta-item span {{ color:var(--negro); font-weight:700; }}

  /* ── Tarjetas KPI: superficie blanca, borde 1 px, acento superior ───── */
  .kpi-card {{ background:#FFFFFF; border:1px solid var(--gris-claro); border-top:3px solid var(--gris);
               border-radius:0; padding:1.1rem 1.2rem 1rem; height:100%; }}
  .kpi-title {{ font-family:{FUENTE_TITULOS}; font-size:.68rem; font-weight:600; letter-spacing:.06em;
                text-transform:uppercase; color:var(--gris); margin-bottom:.55rem; }}
  .kpi-value {{ font-family:{FUENTE_TITULOS}; font-size:1.85rem; font-weight:700; color:var(--negro); line-height:1.05; }}
  .kpi-sub {{ font-size:.74rem; color:var(--gris); margin-top:.45rem; line-height:1.45; }}
  .kpi-safe   {{ border-top-color:{CLR['verde']} !important; }}
  .kpi-warn   {{ border-top-color:{CLR['amarillo']} !important; }}
  .kpi-danger {{ border-top-color:var(--rojo) !important; }}
  .kpi-info   {{ border-top-color:var(--gris) !important; }}

  /* ── Títulos de sección ──────────────────────────────────────────────── */
  .section-header {{ font-family:{FUENTE_TITULOS}; font-size:.95rem; font-weight:700; color:var(--negro);
                     border-bottom:1px solid var(--gris-claro); padding-bottom:.45rem; margin:1.6rem 0 .9rem;
                     position:relative; }}
  .section-header::after {{ content:''; position:absolute; left:0; bottom:-1px; width:48px; height:2px; background:var(--rojo); }}

  /* ── Lectura del gráfico: bloque plano ──────────────────────────────── */
  .interp-box {{ background:#FFFFFF; border:1px solid var(--gris-claro); border-left:3px solid var(--granate);
                 border-radius:0; padding:.85rem 1.1rem; margin:.4rem 0 1.2rem; font-size:.84rem;
                 color:var(--negro); line-height:1.65; }}
  .interp-box b {{ color:var(--granate); font-family:{FUENTE_TITULOS}; font-weight:600; }}

  /* ── Pestañas ────────────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {{ background:#FFFFFF; border:1px solid var(--gris-claro);
                                       border-bottom:2px solid var(--rojo); gap:0; padding:0 .4rem; flex-wrap:wrap; }}
  .stTabs [data-baseweb="tab"], .stTabs [role="tab"] {{ font-family:{FUENTE_TITULOS}; font-size:.76rem; font-weight:500; color:var(--gris) !important;
                                  padding:.7rem .5rem; background:transparent !important; border-radius:0; white-space:nowrap; }}
  .stTabs [data-baseweb="tab"]:hover, .stTabs [role="tab"]:hover {{ color:var(--rojo) !important; }}
  .stTabs [aria-selected="true"] {{ color:var(--rojo) !important; font-weight:700 !important; }}
  .stTabs [data-baseweb="tab-highlight"] {{ background-color:var(--rojo) !important; height:3px; }}
  .stTabs [data-baseweb="tab-border"] {{ display:none; }}
  .stTabs [data-baseweb="tab-panel"] {{ padding-top:1.2rem; }}

  /* ── Contenedores de gráficos y tablas: superficie blanca, 1 px ─────── */
  [data-testid="stPlotlyChart"] {{ background:#FFFFFF; border:1px solid var(--gris-claro); padding:.6rem .6rem .2rem; }}
  [data-testid="stDataFrame"] {{ border:1px solid var(--gris-claro); }}
  [data-testid="stExpander"] details {{ background:#FFFFFF; border:1px solid var(--gris-claro) !important; border-radius:0 !important; }}
  [data-testid="stExpander"] summary {{ font-family:{FUENTE_TITULOS}; font-size:.82rem; font-weight:600; color:var(--negro); }}
  [data-testid="stExpander"] summary:hover {{ color:var(--rojo); }}
  [data-testid="stFileUploader"] section {{ background:#FFFFFF; border:1px dashed var(--gris-claro); border-radius:0; }}
  hr {{ border:none !important; border-top:1px solid var(--gris-claro) !important; margin:1.8rem 0 !important; }}

  /* ── Botones ─────────────────────────────────────────────────────────── */
  .stButton > button[kind="primary"], .stDownloadButton > button {{
      background:var(--rojo) !important; color:#FFFFFF !important; border:1px solid var(--rojo) !important; border-radius:2px !important;
      font-family:{FUENTE_TITULOS} !important; font-weight:600 !important; font-size:.82rem !important; }}
  .stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {{ background:var(--granate) !important; border-color:var(--granate) !important; }}
  .stButton > button[kind="secondary"] {{ background:#FFFFFF !important; color:var(--negro) !important;
      border:1px solid var(--gris-claro) !important; border-radius:2px !important; }}

  /* ── Radio horizontal ───────────────────────────────────────────────── */
  .stRadio [role="radiogroup"] label {{ font-size:.82rem; }}

  /* Scrollbar discreta */
  ::-webkit-scrollbar {{ width:6px; height:6px; }}
  ::-webkit-scrollbar-thumb {{ background:var(--gris-claro); }}
</style>
""", unsafe_allow_html=True)

# Evita que el traductor automático del navegador reescriba la página: si Chrome traduce,
# React no puede actualizar los textos y estos quedan "congelados" al cambiar los filtros.
st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)
components.html("""<script>
  const d = window.parent.document;
  d.documentElement.setAttribute('lang', 'es');
  d.documentElement.setAttribute('translate', 'no');
  d.documentElement.classList.add('notranslate');
  d.body.classList.add('notranslate');
</script>""", height=0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS Y MODELOS (con caché)
#    Fuente: panel exportado por el cuaderno V12 (mi_dataset_final.csv) y
#    artefactos de modelos_exportados/. El dashboard NO recalcula el Z'':
#    usa el Z'' CALCULADO en el cuaderno (decisión metodológica del proyecto).
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
AÑOS = [2021, 2022, 2023, 2024]
ZONAS = ["🔴 Zona Crítica", "🟡 Zona Alerta", "🟢 Zona Segura"]
MAPA_ZONA = {"Zona Crítica": "🔴 Zona Crítica", "Zona Alerta": "🟡 Zona Alerta", "Zona Segura": "🟢 Zona Segura"}
SIN_Z = "⚪ Sin Z calculado"
N_MIN_SECTOR = 10          # mínimo de empresas para interpretar una mediana sectorial
ORDEN_TAM = ["Microempresa", "Pequeña", "Mediana", "Grande", "Sin dato"]

SECTOR_CORTO = {
    "A – Agricultura y ganadería":                  "A – Agricultura",
    "B – Explotación de minas y canteras":          "B – Minería",
    "C – Industria manufacturera":                  "C – Manufactura",
    "D – Electricidad, gas, vapor":                 "D – Electricidad",
    "E – Agua y saneamiento":                       "E – Agua y saneamiento",
    "F – Construcción":                             "F – Construcción",
    "G – Comercio al por mayor y menor":            "G – Comercio",
    "H – Transporte y almacenamiento":              "H – Transporte",
    "I – Alojamiento y gastronomía":                "I – Alojamiento",
    "J – Información y comunicaciones":             "J – Información",
    "K – Actividades financieras y de seguros":     "K – Financiero",
    "L – Actividades inmobiliarias":                "L – Inmobiliario",
    "M – Actividades profesionales y científicas":  "M – Profesionales",
    "N – Actividades de servicios administrativos": "N – Administrativos",
    "P – Educación":                                "P – Educación",
    "Q – Salud y servicios sociales":               "Q – Salud",
    "R – Arte, entretenimiento y recreación":       "R – Arte/Recreación",
    "S – Otras actividades de servicios":           "S – Otros servicios",
}

# Nombres del CSV → nombres de visualización (coinciden con las columnas del cuaderno)
RENOMBRE = {
    "Compania": "Compañía", "Seccion_CIIU": "Sección CIIU", "Clasificacion_Tamaño": "Tamaño 2024",
    "Clasificacion_957_Año": "Tamaño (UVT del año)",
    "Razon_Liquidez": "Razón De Liquidez (x)", "Prueba_Acida": "Prueba Ácida (x)",
    "Razon_Efectivo": "Razón De Efectivo (x)", "Deuda_Activos_pct": "Deuda / Activos (%)",
    "Deuda_Capital_pct": "Relación Deuda/Capital (%)", "Deuda_Neta_EBITDA": "Deuda Neta / EBITDA (x)",
    "ROA_pct": "Rendimiento Sobre Los Activos (ROA) (%)", "ROE_pct": "Rendimiento Sobre El Patrimonio (ROE) (%)",
    "Margen_Operacional_pct": "Margen Operacional (%)", "Margen_Neto_pct": "Margen Neto (%)",
    "Margen_EBITDA_pct": "Margen Ebitda (%)", "Activos_Totales": "Activos Totales",
    "Pasivos_Totales": "Pasivos Totales", "Patrimonio_Total": "Total de patrimonio",
    "Ingresos_Operativos": "Total Ingreso Operativo", "Utilidad_Neta": "Ganancia (Pérdida) Neta",
    "EBIT": "Ganancia operativa (EBIT)", "Capital_Trabajo": "Capital De Trabajo",
}
# Atajos de columnas (formato largo)
LIQ, ACI, EFE = "Razón De Liquidez (x)", "Prueba Ácida (x)", "Razón De Efectivo (x)"
DFA, DCAP, DNE = "Deuda / Activos (%)", "Relación Deuda/Capital (%)", "Deuda Neta / EBITDA (x)"
PAS_ACT = "Pasivos/Activos (%)"
ROA, ROE = "Rendimiento Sobre Los Activos (ROA) (%)", "Rendimiento Sobre El Patrimonio (ROE) (%)"
MOP, MNE, MEB = "Margen Operacional (%)", "Margen Neto (%)", "Margen Ebitda (%)"
ZC = "ZScore_Calculado"


# Formato numérico colombiano: punto para miles, coma para decimales
def fmt(v, dec=2, suf=""):
    return "—" if v is None or pd.isna(v) else f"{v:,.{dec}f}{suf}".replace(",", "X").replace(".", ",").replace("X", ".")


CARPETAS_BUSQUEDA = [BASE, os.getcwd(), os.path.join(BASE, "final2"), os.path.join(BASE, "data"),
                     os.path.join(BASE, "datos"), os.path.dirname(BASE)]


def _ruta(nombre, carpeta=None):
    """Busca un archivo junto al dashboard, en la carpeta desde donde se ejecuta y en subcarpetas habituales."""
    for raiz in dict.fromkeys(CARPETAS_BUSQUEDA):
        for r in ([os.path.join(raiz, carpeta, nombre)] if carpeta else []) + [os.path.join(raiz, nombre)]:
            if os.path.exists(r):
                return r
    return None


@st.cache_data(show_spinner="⏳ Cargando dataset financiero…")
def load_data(fuente):
    """fuente: ruta del CSV o bytes de un archivo cargado desde la pantalla."""
    if fuente is None:
        return None, None, "no_encontrado"
    try:
        panel = pd.read_csv(io.BytesIO(fuente) if isinstance(fuente, bytes) else fuente, low_memory=False)
    except Exception as e:
        return None, None, f"No fue posible leer el archivo CSV: {e}"
    faltan = [c for c in ["ZScore_Calculado", "Rol_Modelo_Multivariado", "Zona_ZScore", "Clasificacion_957_Año"] if c not in panel.columns]
    if faltan:
        return None, None, ("El archivo 'mi_dataset_final.csv' corresponde a una versión anterior del cuaderno (le faltan las columnas "
                            + ", ".join(faltan) + "). Use el CSV que exporta V12_Trabajo_Integrador_Auditado.ipynb.")
    panel = panel.rename(columns=RENOMBRE)
    panel["Año"] = panel["Año"].astype(int)
    panel["Sector_Corto"] = panel["Sección CIIU"].map(SECTOR_CORTO).fillna(panel["Sección CIIU"])
    panel["Zona"] = panel["Zona_ZScore"].map(MAPA_ZONA).fillna(SIN_Z)
    panel[PAS_ACT] = panel["Pasivos Totales"] / panel["Activos Totales"].replace(0, np.nan) * 100
    for c in ["Alerta_Calidad_Dato", "Patrimonio_Negativo"]:
        panel[c] = panel[c].astype(str).str.lower().isin(["true", "1"])
    # Formato ancho (una fila por empresa) para el módulo multivariado y la matriz de transición
    valores = [c for c in panel.columns if c not in ("Compañía", "Año") and not re.search(r"_20\d{2}$", c)]
    wide = panel.set_index(["Compañía", "Año"])[valores].unstack("Año")
    wide.columns = [f"{c} {a}" for c, a in wide.columns]
    estaticas = panel.groupby("Compañía")[["Sección CIIU", "Sector_Corto", "Cluster_Label",
                                           "Cluster_KMeans", "Rol_Modelo_Multivariado"]].first()
    wide = estaticas.join(wide).reset_index()
    return panel, wide, None


@st.cache_resource(show_spinner="⏳ Cargando modelos…")
def load_models():
    m = {}
    for nombre in ["preproceso_clustering", "pca_2d", "kmeans_k3", "etiquetas_cluster",
                   "arbol_decision_zscore", "features_arbol", "features_modelo"]:
        r = _ruta(f"{nombre}.pkl", "modelos_exportados")
        if r:
            try:
                m[nombre] = joblib.load(r)
            except Exception as e:                       # versión de scikit-learn incompatible, etc.
                m.setdefault("_errores", []).append(f"{nombre}: {e}")
    return m


def _mensaje(titulo, cuerpo):
    st.markdown(f"<div style='background:#FFFFFF;border:1px solid {CLR['rojo']};border-left:6px solid {CLR['rojo']};"
                f"border-radius:0;padding:1rem 1.3rem;margin:1rem 0;color:#1A1A1A;font-size:.9rem;line-height:1.6;'>"
                f"<b style='color:{CLR['rojo']};font-size:1rem;'>{titulo}</b><br>{cuerpo}</div>", unsafe_allow_html=True)


panel_raw, wide_raw, error_carga = load_data(_ruta("mi_dataset_final.csv"))
if error_carga:
    if error_carga == "no_encontrado":
        _mensaje("No se encontró el archivo de datos 'mi_dataset_final.csv'",
                 "El dashboard lo buscó en estas carpetas:<br>" + "<br>".join(f"• <code>{c}</code>" for c in dict.fromkeys(CARPETAS_BUSQUEDA))
                 + "<br><br>Copie en la misma carpeta de <code>dashboard_app.py</code> el archivo <code>mi_dataset_final.csv</code> y la "
                   "carpeta <code>modelos_exportados</code> que genera el cuaderno V12, y recargue la página. "
                   "También puede cargar el CSV aquí abajo.")
    else:
        _mensaje("El archivo de datos no es compatible", error_carga + "<br><br>Puede cargar el CSV correcto aquí abajo.")
    subido = st.file_uploader("Cargar mi_dataset_final.csv (exportado por el cuaderno V12)", type="csv")
    if subido is None:
        st.stop()
    panel_raw, wide_raw, error_carga = load_data(subido.getvalue())
    if error_carga:
        _mensaje("El archivo cargado no es compatible", error_carga)
        st.stop()
modelos = load_models()
N_EMPRESAS_TOTAL = panel_raw["Compañía"].nunique()
N_SECTORES_TOTAL = panel_raw["Sección CIIU"].nunique()


# ─────────────────────────────────────────────────────────────────────────────
# 2. SIDEBAR — FILTROS GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
def _etiqueta(txt, top=".7rem"):
    st.markdown(f"<div style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;font-size:.75rem;font-weight:600;"
                f"color:{CLR['gris_oscuro']};margin-bottom:.3rem;margin-top:{top};'>{txt}</div>",
                unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div style='background:{CLR["rojo"]};height:4px;margin:-1rem -1rem 1rem;'></div>
    <div style='padding:.2rem 0 .8rem;'>
      <div style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;font-weight:700;font-size:.95rem;color:{CLR["gris_oscuro"]};'>
        Filtros de análisis</div>
      <div style='font-size:.72rem;color:#636363;margin-top:.15rem;'>
        Los filtros se aplican a todas las pestañas. Las series 2021–2024 respetan el sector y la exclusión de alertas.</div>
    </div>
    <hr style='border:none;border-top:1px solid {CLR["gris_claro"]};margin:.2rem 0 .8rem;'>
    """, unsafe_allow_html=True)

    _etiqueta("Año de análisis", ".2rem")
    año_sel = st.selectbox("Año de análisis", AÑOS, index=3, label_visibility="collapsed")

    _etiqueta("Sector económico (CIIU)")
    sectores_disp = sorted(panel_raw["Sector_Corto"].unique().tolist())
    sector_sel = st.multiselect("Sector", sectores_disp, default=[], placeholder="Todos los sectores",
                                label_visibility="collapsed")

    _etiqueta(f"Tamaño — Decreto 957 (UVT de {año_sel})")
    tam_disp = [t for t in ORDEN_TAM if t in panel_raw.loc[panel_raw["Año"] == año_sel, "Tamaño (UVT del año)"].unique()]
    tam_sel = st.multiselect("Tamaño", tam_disp, default=tam_disp, label_visibility="collapsed",
                             help="El tamaño se calcula en cada año con los ingresos y el UVT de ese año. "
                                  "Como la muestra son las 3.000 mayores empresas por ingresos de 2024, casi todas son 'Grandes' en 2024.")

    _etiqueta("Zona Z'' de Altman (calculado)")
    zona_opts = ZONAS[::-1] + [SIN_Z]
    zona_sel = st.multiselect("Zona", zona_opts, default=zona_opts, label_visibility="collapsed")

    _etiqueta("Calidad de datos")
    excluir_alertas = st.checkbox("Excluir empresas con alerta de calidad de dato", value=False,
                                  help="Marca empresas-año con valores implausibles en EMIS (ROA > 100 %, ingresos > 50 veces "
                                       "los activos, margen neto EMIS inconsistente o razón corriente > 100). Los rankings "
                                       "de empresas siempre las excluyen.")
    ocultar_peq = st.checkbox(f"Ocultar sectores con menos de {N_MIN_SECTOR} empresas", value=True,
                              help="Las medianas de grupos muy pequeños dependen de pocas empresas y no se interpretan.")

def aplicar_filtros(p, año=None, filtros_fila=True):
    d = p.copy()
    if sector_sel:
        d = d[d["Sector_Corto"].isin(sector_sel)]
    if excluir_alertas:
        d = d[~d["Alerta_Calidad_Dato"]]
    if filtros_fila:
        if tam_sel:
            d = d[d["Tamaño (UVT del año)"].isin(tam_sel)]
        if zona_sel:
            d = d[d["Zona"].isin(zona_sel)]
    if año is not None:
        d = d[d["Año"] == año]
    return d

df_año = aplicar_filtros(panel_raw, año_sel)                      # empresas del año seleccionado
df_serie = aplicar_filtros(panel_raw, filtros_fila=False)         # series 2021–2024 (sector + alertas)
if df_año.empty:
    st.warning("Los filtros seleccionados no dejan empresas. Amplíe la selección en la barra lateral.")
    st.stop()

with st.sidebar:
    st.markdown(f"""
    <hr style='border:none;border-top:1px solid {CLR["gris_claro"]};margin:1rem 0 .8rem;'>
    <div style='font-size:.7rem;color:#636363;line-height:1.75;'>
      <div style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;font-weight:700;color:{CLR["rojo"]};margin-bottom:.3rem;font-size:.72rem;'>
        Universidad del Valle</div>
      Maestría en Analítica e Inteligencia de Negocios<br>
      Trabajo de grado — caso de estudio de analítica descriptiva (2021–2024)<br><br>
      <b>Fuente:</b> EMIS Platform (COP millones)<br>
      <b>Universo:</b> {fmt(N_EMPRESAS_TOTAL, 0)} empresas · {N_SECTORES_TOTAL} secciones CIIU<br>
      <b>Selección actual ({año_sel}):</b> {fmt(df_año['Compañía'].nunique(), 0)} empresas<br>
      <b>Riesgo:</b> Z'' de Altman calculado (4 variables, valores en libros)
    </div>""", unsafe_allow_html=True)
    st.download_button("⬇️ Descargar datos filtrados (CSV)", df_año.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"empresas_filtradas_{año_sel}.csv", mime="text/csv", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CABECERA
# El manual exige la firma horizontal oficial en la esquina superior izquierda, sin alterarla
# (tamaño preferido 424 × 51 px). No se dibuja una versión propia: si en la carpeta del dashboard
# existe el archivo oficial (logo_univalle.png / .svg), se muestra; si no, la cabecera queda solo con texto.
# ─────────────────────────────────────────────────────────────────────────────
def _logo_html():
    for nombre, mime in [("logo_univalle.svg", "image/svg+xml"), ("logo_univalle.png", "image/png")]:
        r = _ruta(nombre)
        if r:
            with open(r, "rb") as f:
                datos = base64.b64encode(f.read()).decode()
            return (f"<div class='dash-header-logo'><img src='data:{mime};base64,{datos}' alt='Universidad del Valle' "
                    f"style='height:51px;max-width:424px;'></div>")
    return ""
LOGO_HTML = _logo_html()

st.markdown(f"""
<div class='dash-header'>
  <div class='dash-header-stripe'></div>
  <div class='dash-header-body'>{LOGO_HTML}
    <div class='dash-header-texts'>
      <div class='dash-header-pretitle'>Universidad del Valle &nbsp;·&nbsp; MAIN — Maestría en Analítica e Inteligencia de Negocios</div>
      <h1>Desempeño financiero de las principales empresas del <span class='titulo-rojo'>sector real colombiano</span></h1>
      <p>Prototipo de analítica descriptiva · Liquidez, endeudamiento y rentabilidad · 2021–2024</p>
      <div class='dash-header-meta'>
        <div class='dash-header-meta-item'>Empresas en la selección: <span>{fmt(df_año['Compañía'].nunique(), 0)}</span></div>
        <div class='dash-header-meta-item'>Año: <span>{año_sel}</span></div>
        <div class='dash-header-meta-item'>Fuente: <span>EMIS Platform</span></div>
        <div class='dash-header-meta-item'>Riesgo: <span>Z'' de Altman calculado</span></div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE VISUALIZACIÓN E INTERPRETACIÓN
# ─────────────────────────────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", separators=",.",
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=CLR["gris_claro"], font_color=CLR["azul_oscuro"], font_family="Helvetica Neue, Helvetica, Arial, sans-serif"), font_color=CLR["gris_oscuro"],
    font_family="Helvetica Neue, Helvetica, Arial, sans-serif", margin=dict(t=40, b=40, l=10, r=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font_size=11, font_color=CLR["azul_oscuro"],
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(gridcolor=CLR["fondo"], zerolinecolor=CLR["gris_claro"], linecolor=CLR["gris_claro"], tickfont=dict(color=CLR["gris_medio"], size=11)),
    yaxis=dict(gridcolor=CLR["fondo"], zerolinecolor=CLR["gris_claro"], linecolor=CLR["gris_claro"], tickfont=dict(color=CLR["gris_medio"], size=11)),
)
# Secuencia para series categóricas: solo colores de la paleta (CLR)
COLOR_SEQ = [CLR["rojo"], CLR["gris_oscuro"], CLR["azul_claro"], CLR["verde"], CLR["granate"], CLR["amarillo"],
             CLR["cian"], CLR["morado"]]
COLOR_ZONA = {"🟢 Zona Segura": CLR["verde"], "🟡 Zona Alerta": CLR["amarillo"], "🔴 Zona Crítica": CLR["rojo"], SIN_Z: CLR["gris_claro"]}
COLOR_CLUSTER = {"Alta rentabilidad y liquidez": CLR["verde"], "Perfil intermedio de baja deuda financiera": CLR["azul_claro"],
                 "Mayor deuda financiera y menor rentabilidad": CLR["granate"]}


def apply_layout(fig, title="", height=360):
    fig.update_layout(**LAYOUT_BASE, height=height,
                      title=dict(text=title, font_size=13, font_color=CLR["azul_oscuro"],
                                 font_family="Helvetica Neue, Helvetica, Arial, sans-serif", x=0))
    return fig


def eje_años(fig):
    """Muestra solo años enteros en el eje X (evita marcas como 2021.5)."""
    fig.update_xaxes(tickmode="array", tickvals=AÑOS, ticktext=[str(a) for a in AÑOS], range=[2020.7, 2024.3])
    return fig


def kpi_html(titulo, valor, sub="", clase="kpi-info"):
    return (f"<div class='kpi-card {clase}'><div class='kpi-title'>{titulo}</div>"
            f"<div class='kpi-value'>{valor}</div><div class='kpi-sub'>{sub}</div></div>")


def interp(texto, titulo="Lectura del gráfico"):
    st.markdown(f"<div class='interp-box notranslate' translate='no'><b>{titulo}:</b> {texto}</div>", unsafe_allow_html=True)


def seccion(txt):
    st.markdown(f"<p class='section-header'>{txt}</p>", unsafe_allow_html=True)


def pregunta(n, texto):
    st.markdown(f"<div style='background:#FFFFFF;border:1px solid {CLR['gris_claro']};border-left:3px solid {CLR['rojo']};border-radius:0;"
                f"padding:.55rem .9rem;margin:.2rem 0 .9rem;font-size:.85rem;color:{CLR['gris_oscuro']};'>"
                f"<b style='color:{CLR['rojo']};'>Pregunta orientadora {n}.</b> {texto}</div>", unsafe_allow_html=True)




def resumen_por_grupo(d, grupo, col):
    """Mediana, cuartiles y n por grupo; aplica el umbral de tamaño mínimo si corresponde."""
    g = (d.dropna(subset=[col]).groupby(grupo)[col]
         .agg(Mediana="median", P25=lambda s: s.quantile(.25), P75=lambda s: s.quantile(.75), n="size").reset_index())
    if ocultar_peq and grupo == "Sector_Corto":
        g = g[g["n"] >= N_MIN_SECTOR]
    return g


def nota_n(g):
    excl = []
    if ocultar_peq:
        conteo = df_año.groupby("Sector_Corto").size()
        excl = conteo[conteo < N_MIN_SECTOR].index.tolist()
    return (f" Sectores ocultos por tener menos de {N_MIN_SECTOR} empresas: {', '.join(excl)}." if excl else "")


def serie_medianas(d, col):
    g = d.dropna(subset=[col]).groupby("Año")[col].agg(Mediana="median", P25=lambda s: s.quantile(.25),
                                                        P75=lambda s: s.quantile(.75), n="size").reset_index()
    return g


def zona_pct(d):
    con_z = d[d["Zona"] != SIN_Z]
    n = len(con_z)
    out = {z: ((con_z["Zona"] == z).sum(), (con_z["Zona"] == z).mean() * 100 if n else np.nan) for z in ZONAS}
    return out, n, (d["Zona"] == SIN_Z).sum()


HOVER_MED = "<b>%{x}</b><br>Mediana: %{y:.2f}<br>P25–P75: %{customdata[0]:.2f} – %{customdata[1]:.2f}<br>n = %{customdata[2]}<extra></extra>"


# ─────────────────────────────────────────────────────────────────────────────
# FICHAS METODOLÓGICAS (glosario)
# ─────────────────────────────────────────────────────────────────────────────
def ficha_as_html(nombre, tag, simple, analogia, formula, niveles, tecnico="", nota_niveles=""):
    c_rojo, c_os, c_md, c_cl, c_fondo = CLR["rojo"], CLR["gris_oscuro"], CLR["gris_medio"], CLR["gris_claro"], CLR["fondo"]
    pal = {"verde": (CLR["verde"], trama(CLR["verde"], 10)), "amarillo": (CLR["amarillo"], trama(CLR["amarillo"], 10)),
           "rojo": (c_rojo, trama(c_rojo, 10))}

    def badge(c, t):
        tc, bg = pal.get(c, (c_md, c_fondo))
        return (f"<span style='display:inline-block;padding:.22rem .75rem;border-radius:0;font-size:.7rem;font-weight:600;"
                f"background:{bg};color:{tc};border:1px solid {tc};margin:.1rem .1rem .1rem 0;'>{t}</span>")
    tec = (f"<div style='border-left:2px solid {c_cl};padding:.55rem .85rem;margin-bottom:.6rem;font-size:.78rem;"
           f"color:{c_os};line-height:1.65;'><div style='font-size:.63rem;font-weight:700;color:{c_md};text-transform:uppercase;"
           f"margin-bottom:.22rem;'>Definición técnica</div>{tecnico}</div>") if tecnico else ""
    nota = f"<div style='font-size:.7rem;color:{c_md};margin-top:.35rem;font-style:italic;'>{nota_niveles}</div>" if nota_niveles else ""
    return (f"<div style='background:#FFFFFF;border-radius:0;border:1px solid {c_cl};overflow:hidden;margin:.55rem 0;'>"
            f"<div style='background:{c_fondo};border-bottom:2px solid {c_rojo};padding:.55rem .95rem;display:flex;gap:.5rem;'>"
            f"<span style='color:{c_os};font-weight:700;font-size:.86rem;flex:1;'>{nombre}</span>"
            f"<span style='font-size:.63rem;padding:.18rem .65rem;background:{c_rojo};color:#FFF;font-weight:700;"
            f"text-transform:uppercase;white-space:nowrap;'>{tag}</span></div><div style='padding:.75rem .95rem;'>"
            f"<div style='background:{c_fondo};border-left:2px solid {c_os};padding:.55rem .85rem;margin-bottom:.6rem;"
            f"font-size:.79rem;color:{c_os};line-height:1.65;'><div style='font-size:.63rem;font-weight:700;text-transform:uppercase;"
            f"margin-bottom:.2rem;'>Descripción accesible</div>{simple}"
            f"<div style='margin-top:.4rem;font-style:italic;color:{c_md};font-size:.78rem;'>{analogia}</div></div>{tec}"
            f"<div style='font-size:.63rem;font-weight:700;color:{c_os};text-transform:uppercase;margin-bottom:.24rem;'>Fórmula</div>"
            f"<div style='font-family:Courier New,monospace;font-size:.78rem;background:{c_fondo};border:1px solid {c_cl};"
            f"padding:.38rem .75rem;border-radius:0;margin:.1rem 0 .6rem;'>{formula}</div>"
            f"<div style='font-size:.63rem;font-weight:700;color:{c_os};text-transform:uppercase;margin:.35rem 0 .25rem;'>"
            f"Lectura de referencia</div><div>{''.join(badge(c, t) for c, t in niveles)}</div>{nota}</div></div>")


def ficha(clave):
    st.markdown(ficha_as_html(**FICHAS[clave]), unsafe_allow_html=True)


NOTA_UMBRAL = ("Rangos orientadores de uso común en análisis financiero, no normativos: "
               "deben leerse junto con el sector y el modelo de negocio de cada empresa.")
FICHAS = {
    "razon_corriente": dict(
        nombre="Razón corriente (liquidez)", tag="Liquidez",
        simple="Indica si la empresa tiene suficientes recursos de corto plazo para pagar las deudas que vencen en el próximo año.",
        analogia="Si debes $1.000.000 este año y tienes $1.420.000 disponibles, tu razón corriente es 1,42x.",
        formula="Activos corrientes ÷ Pasivos corrientes",
        tecnico="Compara los activos que se convertirán en efectivo en menos de 12 meses con las obligaciones de menos de un año.",
        niveles=[("verde", "≥ 1,5x holgura"), ("amarillo", "1,0 – 1,5x atención"), ("rojo", "< 1,0x presión de liquidez")],
        nota_niveles=NOTA_UMBRAL),
    "prueba_acida": dict(
        nombre="Prueba ácida", tag="Liquidez",
        simple="Como la razón corriente, pero sin contar los inventarios, que pueden tardar en venderse.",
        analogia="De $1.420.000 disponibles, $400.000 son mercancía por vender: tu liquidez inmediata es $1.020.000.",
        formula="(Activos corrientes − Inventarios) ÷ Pasivos corrientes",
        tecnico="La brecha entre razón corriente y prueba ácida muestra cuánto depende la empresa de vender inventario para pagar.",
        niveles=[("verde", "≥ 1,0x"), ("amarillo", "0,8 – 1,0x"), ("rojo", "< 0,8x")], nota_niveles=NOTA_UMBRAL),
    "razon_efectivo": dict(
        nombre="Razón de efectivo", tag="Liquidez",
        simple="Cuánto dinero en caja y bancos tiene la empresa frente a sus deudas de corto plazo.",
        analogia="Es el dinero disponible hoy, sin contar inventarios ni lo que te deben los clientes.",
        formula="(Efectivo + Equivalentes) ÷ Pasivos corrientes",
        tecnico="Medida conservadora de liquidez. Valores muy altos pueden indicar caja ociosa.",
        niveles=[("verde", "≥ 0,5x"), ("amarillo", "0,2 – 0,5x"), ("rojo", "< 0,2x")], nota_niveles=NOTA_UMBRAL),
    "pasivos_activos": dict(
        nombre="Pasivos / Activos (endeudamiento total)", tag="Endeudamiento",
        simple="Qué parte de todo lo que tiene la empresa está financiada con obligaciones con terceros (bancos, proveedores, empleados, impuestos).",
        analogia="Si una empresa tiene activos por $100 y debe $59 en total, su endeudamiento total es 59 %.",
        formula="Pasivos totales ÷ Activos totales × 100",
        tecnico="Calculado en el cuaderno desde los estados financieros. Es la medida usada para el endeudamiento total.",
        niveles=[("verde", "< 40 %"), ("amarillo", "40 – 60 %"), ("rojo", "> 60 %")], nota_niveles=NOTA_UMBRAL),
    "deuda_activos": dict(
        nombre="Deuda financiera / Activos (EMIS)", tag="Endeudamiento",
        simple="Qué parte de los activos está financiada con préstamos y obligaciones financieras (no incluye proveedores ni otras cuentas por pagar).",
        analogia="Una empresa puede deber mucho a proveedores y tener cero créditos bancarios: su deuda financiera sería 0 %.",
        formula="Deuda financiera (corto + largo plazo) ÷ Activos totales × 100",
        tecnico=("Indicador precalculado por EMIS. En la muestra su mediana es ≈ 7 % frente a ≈ 59 % de pasivos/activos, "
                 "y cerca de una de cada cuatro empresas no tiene deuda financiera. No se le aplican los umbrales del endeudamiento total."),
        niveles=[("gris", "Sin umbral: se compara entre sectores y años")]),
    "deuda_capital": dict(
        nombre="Deuda financiera / Patrimonio (EMIS)", tag="Endeudamiento",
        simple="Cuántos pesos de deuda financiera tiene la empresa por cada peso aportado por los dueños.",
        analogia="Si los socios aportaron $100 y la empresa debe $50 a bancos, la relación es 50 %.",
        formula="Deuda financiera ÷ Patrimonio × 100",
        tecnico="Indicador precalculado por EMIS. No es interpretable cuando el patrimonio es negativo.",
        niveles=[("gris", "Sin umbral: se compara entre sectores y años")]),
    "roa": dict(
        nombre="ROA — Rentabilidad sobre activos", tag="Rentabilidad",
        simple="Por cada $100 en activos (máquinas, inventarios, caja…), cuánta utilidad neta genera la empresa.",
        analogia="Un ROA de 5 % equivale a $5 de ganancia por cada $100 invertidos en activos.",
        formula="Utilidad neta ÷ Activos totales × 100",
        tecnico="Mide la eficiencia en el uso de los recursos, independientemente de cómo se financian.",
        niveles=[("verde", "> 10 %"), ("verde", "5 – 10 %"), ("amarillo", "2 – 5 %"), ("rojo", "< 2 %")], nota_niveles=NOTA_UMBRAL),
    "roe": dict(
        nombre="ROE — Rentabilidad sobre patrimonio", tag="Rentabilidad",
        simple="Cuánto ganan los dueños por cada peso de patrimonio.",
        analogia="Con un patrimonio de $100 millones y una utilidad de $15 millones, el ROE es 15 %.",
        formula="Utilidad neta ÷ Patrimonio × 100",
        tecnico=("Puede ser alto por buena rentabilidad o por alto endeudamiento (ver DuPont). Con patrimonio negativo no es "
                 "interpretable: una pérdida produciría un ROE positivo."),
        niveles=[("verde", "> 15 %"), ("verde", "8 – 15 %"), ("amarillo", "3 – 8 %"), ("rojo", "< 3 %")], nota_niveles=NOTA_UMBRAL),
    "margen_neto": dict(
        nombre="Margen neto", tag="Rentabilidad",
        simple="De cada $100 vendidos, cuántos quedan como ganancia final después de todos los costos, gastos, intereses e impuestos.",
        analogia="Si vendes una arepa a $1.000 y después de pagar todo te quedan $34, tu margen neto es 3,4 %.",
        formula="Utilidad neta ÷ Ventas × 100",
        tecnico="Varía mucho por sector: el comercio suele operar con márgenes bajos y alta rotación.",
        niveles=[("verde", "> 10 %"), ("verde", "4 – 10 %"), ("amarillo", "1 – 4 %"), ("rojo", "< 1 %")], nota_niveles=NOTA_UMBRAL),
    "margen_ebitda": dict(
        nombre="Margen EBITDA", tag="Rentabilidad",
        simple="Cuánto genera el negocio con su operación antes de intereses, impuestos, depreciaciones y amortizaciones.",
        analogia="Es lo que queda de la operación diaria antes de pagar créditos e impuestos.",
        formula="EBITDA ÷ Ventas × 100",
        tecnico="Aproximación a la generación operativa; no equivale al flujo de caja libre.",
        niveles=[("verde", "> 15 %"), ("verde", "8 – 15 %"), ("amarillo", "3 – 8 %"), ("rojo", "< 3 %")], nota_niveles=NOTA_UMBRAL),
    "zscore": dict(
        nombre="Z'' de Altman (calculado)", tag="Riesgo financiero",
        simple=("Un puntaje que combina liquidez, utilidades acumuladas, rentabilidad operativa y solvencia en un solo número. "
                "Un valor más alto indica una situación financiera más holgada. Es una señal de riesgo, no una probabilidad de quiebra."),
        analogia="Como un chequeo médico que combina varios exámenes en un semáforo: verde, amarillo o rojo.",
        formula="Z'' = 6,56·(CT/AT) + 3,26·(UR/AT) + 6,72·(EBIT/AT) + 1,05·(Patrimonio/Pasivos)",
        tecnico=("Versión de cuatro variables con valores en libros para empresas no manufactureras y no cotizadas "
                 "(Altman, Hartzell y Peck, 1995). Calculado en el cuaderno con los estados financieros; no se usa el puntaje de EMIS. "
                 "Requiere 'Resultados acumulados', por lo que algunas empresas no tienen Z. El modelo EMS de los mismos autores suma "
                 "3,25 a este puntaje y desplaza los puntos de corte en la misma cantidad, por lo que la clasificación no cambia."),
        niveles=[("verde", "Z'' > 2,60 Zona Segura"), ("amarillo", "1,10 – 2,60 Zona Alerta"), ("rojo", "Z'' < 1,10 Zona Crítica")],
        nota_niveles="Puntos de corte de la literatura; no fueron calibrados con insolvencias observadas en Colombia."),
    "dupont": dict(
        nombre="Descomposición DuPont del ROE", tag="Rentabilidad",
        simple="Separa el ROE en tres partes: cuánto gana por cada venta (margen), cuánto vende con sus activos (rotación) y cuánto usa deuda (apalancamiento).",
        analogia="Dos empresas pueden tener el mismo ROE: una vendiendo poco con mucho margen y otra vendiendo mucho con poco margen.",
        formula="ROE = Margen neto × Rotación de activos × Multiplicador de apalancamiento",
        tecnico=("Margen = Utilidad neta / Ventas; Rotación = Ventas / Activos; Multiplicador = Activos / Patrimonio. "
                 "Se excluyen las empresas con patrimonio ≤ 0. La 'ventaja relativa' compara cada componente del sector con la mediana del universo."),
        niveles=[("gris", "Lectura comparativa entre sectores, sin umbral")]),
    "kmeans": dict(
        nombre="Perfiles financieros (K-Means)", tag="Segmentación",
        simple="Un algoritmo agrupa empresas con indicadores parecidos (liquidez, rentabilidad y endeudamiento de 2021 a 2024) sin usar el Z''.",
        analogia="Como ordenar libros por parecido sin tener etiquetas previas: los grupos resultan de los datos y de las variables elegidas.",
        formula="Minimiza Σ ||xᵢ − μₖ||² (distancia de cada empresa al centro de su grupo), K = 3",
        tecnico=("44 indicadores (11 × 4 años), winsorizados en p1–p99, estandarizados e imputados con KNN (k = 5). "
                 "17 empresas atípicas se tratan como suplementarias: no estiman el modelo, pero se proyectan. "
                 "Silueta 0,17: estructura débil, por lo que los grupos son una segmentación descriptiva y no categorías naturales."),
        niveles=[("verde", "Alta rentabilidad y liquidez"), ("gris", "Perfil intermedio de baja deuda financiera"),
                 ("rojo", "Mayor deuda financiera y menor rentabilidad")]),
}


# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑAS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Inicio", "Resumen general", "Liquidez", "Endeudamiento y riesgo Z''", "Rentabilidad y DuPont",
                "Perfiles financieros", "Ficha de empresa", "Simulador", "Glosario"])
t_ini, t_res, t_liq, t_rie, t_ren, t_clu, t_emp, t_sim, t_glo = tabs

zonas_año, n_conz, n_sinz = zona_pct(df_año)


# ═══════════════════════════════════════════════════════════════════════════════
# INICIO — guía de uso
# ═══════════════════════════════════════════════════════════════════════════════
with t_ini:
    st.markdown(f"""
    <div style='background:#FFFFFF;border:1px solid {CLR["gris_claro"]};border-left:4px solid {CLR["rojo"]};
                border-radius:0;padding:1rem 1.4rem;margin-bottom:1rem;'>
      <div style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;font-weight:700;font-size:1.05rem;color:{CLR["gris_oscuro"]};'>
        ¿Qué es este tablero?</div>
      <div style='font-size:.85rem;color:#636363;line-height:1.65;margin-top:.3rem;'>
        Reúne en un solo lugar los indicadores de <b>liquidez</b>, <b>endeudamiento</b> y <b>rentabilidad</b> de
        {fmt(N_EMPRESAS_TOTAL, 0)} de las 3.000 empresas colombianas con mayores ingresos operativos de 2024 (fuente: EMIS), para
        compararlas entre sectores y entre 2021 y 2024. Es un <b>caso de estudio descriptivo</b>: muestra cómo están las
        empresas, no predice su futuro.<br><br>
        <b>Pregunta de investigación:</b> ¿cómo el diseño de un prototipo de herramienta de analítica descriptiva, sustentado
        en la integración y visualización de indicadores financieros, facilita la comprensión y comparación del desempeño de
        las principales empresas del sector real en Colombia (2021–2024) en liquidez, endeudamiento y rentabilidad?
      </div>
    </div>""", unsafe_allow_html=True)

    seccion("¿Qué quiere saber? — ruta rápida")
    rutas = [
        ("1. ¿Cómo se comporta la liquidez por sector?", "Pestaña <b>Liquidez</b>"),
        ("2. ¿Qué sectores están más endeudados?", "Pestaña <b>Endeudamiento y riesgo Z''</b>"),
        ("3. ¿Cuáles son las empresas y sectores más rentables?", "Pestaña <b>Rentabilidad y DuPont</b>"),
        ("4. ¿Cómo evolucionó el desempeño entre 2021 y 2024?", "Pestaña <b>Resumen general</b> y <b>Ficha de empresa</b>"),
        ("5. ¿Qué empresas muestran señales de alerta financiera?", "Pestaña <b>Endeudamiento y riesgo Z''</b>"),
        ("¿Cómo está una empresa en particular?", "Pestaña <b>Ficha de empresa</b>"),
        ("¿Cómo saldría mi empresa?", "Pestaña <b>Simulador</b>"),
        ("¿Qué significa cada indicador?", "Pestaña <b>Glosario</b>"),
    ]
    cols = st.columns(4)
    for i, (q, a) in enumerate(rutas):
        cols[i % 4].markdown(f"<div class='kpi-card kpi-info' style='min-height:110px;margin-bottom:.6rem;'>"
                             f"<div style='font-size:.8rem;font-weight:600;color:{CLR['gris_oscuro']};'>{q}</div>"
                             f"<div style='font-size:.75rem;color:#636363;margin-top:.4rem;'>{a}</div></div>", unsafe_allow_html=True)

    seccion("Cómo leer el tablero")
    c1, c2, c3 = st.columns(3)
    with c1:
        interp("Casi todas las cifras sectoriales son <b>medianas</b>: el valor de la empresa del medio. Se usan porque los datos "
               "financieros tienen valores extremos (p. ej., una razón corriente de 172 millones) que distorsionan el promedio. "
               "Al pasar el cursor sobre las barras verá también el rango P25–P75 (el 50 % central de las empresas) y el número de empresas.",
               "Medianas")
    with c2:
        interp("El semáforo de riesgo usa el <b>Z'' de Altman calculado</b> con los estados financieros: 🟢 Segura (Z'' > 2,60), "
               "🟡 Alerta (1,10–2,60) y 🔴 Crítica (< 1,10). Es una señal, no una probabilidad de quiebra. "
               f"En {año_sel}, {fmt(n_sinz, 0)} empresas de la selección no tienen Z'' porque EMIS no reporta sus resultados acumulados.",
               "Semáforo Z''")
    with c3:
        interp(f"Los sectores con menos de {N_MIN_SECTOR} empresas se ocultan en las comparaciones sectoriales (puede mostrarlos en "
               "la barra lateral). Las empresas con <b>alerta de calidad de dato</b> tienen valores implausibles en EMIS: no se "
               "eliminan, pero se excluyen de los rankings.", "Precauciones")

    seccion(f"Hallazgos principales — {año_sel} (selección actual)")
    med = {c: df_año[c].median() for c in [LIQ, PAS_ACT, ROA, ROE]}
    roa_ini = df_serie[df_serie["Año"] == 2021][ROA].median()
    roa_fin = df_serie[df_serie["Año"] == 2024][ROA].median()
    h1, h2, h3, h4 = st.columns(4)
    h1.markdown(kpi_html("Liquidez", fmt(med[LIQ], 2, "x"), "Razón corriente mediana", "kpi-info"), True)
    h2.markdown(kpi_html("Endeudamiento total", fmt(med[PAS_ACT], 1, " %"), "Pasivos/Activos mediano", "kpi-info"), True)
    h3.markdown(kpi_html("Rentabilidad", fmt(med[ROA], 1, " %"), f"ROA mediano (2021: {fmt(roa_ini, 1, ' %')} → 2024: {fmt(roa_fin, 1, ' %')})", "kpi-info"), True)
    h4.markdown(kpi_html("Zona Crítica", fmt(zonas_año['🔴 Zona Crítica'][1], 1, " %"), f"de {fmt(n_conz, 0)} empresas con Z''", "kpi-danger"), True)
    st.caption("Fuente: EMIS Platform. Cálculos propios (cuaderno V12). Los resultados dependen de las decisiones metodológicas "
               "documentadas en la pestaña Glosario.")


# ═══════════════════════════════════════════════════════════════════════════════
# RESUMEN GENERAL (Pregunta 4)
# ═══════════════════════════════════════════════════════════════════════════════
with t_res:
    pregunta(4, "¿Cómo ha evolucionado el desempeño financiero entre 2021 y 2024?")
    k = st.columns(4)
    k[0].markdown(kpi_html("Empresas en la selección", f"{fmt(df_año['Compañía'].nunique(), 0)}",
                           f"{df_año['Sección CIIU'].nunique()} sectores CIIU · {año_sel}", "kpi-info"), True)
    for col_k, z, cls in [(k[1], "🟢 Zona Segura", "kpi-safe"), (k[2], "🟡 Zona Alerta", "kpi-warn"), (k[3], "🔴 Zona Crítica", "kpi-danger")]:
        n_z, p_z = zonas_año[z]
        col_k.markdown(kpi_html(z, f"{fmt(n_z, 0)}", f"{fmt(p_z, 1, ' %')} de {fmt(n_conz, 0)} empresas con Z''", cls), True)
    st.caption(f"Los porcentajes de zona se calculan sobre las empresas con Z'' disponible; {fmt(n_sinz, 0)} empresas de la selección no lo tienen.")

    st.markdown("<br>", True)
    col_l, col_r = st.columns([1.25, 0.75])
    with col_l:
        seccion("Evolución de indicadores clave 2021–2024 (medianas)")
        fig_evol = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                                 subplot_titles=("Rentabilidad (%)", "Liquidez (x) y Z'' calculado"))
        series_txt = {}
        for met, col, color, fila in [("ROA", ROA, CLR["azul_claro"], 1), ("ROE", ROE, CLR["verde"], 1),
                                       ("Margen neto", MNE, CLR["amarillo"], 1), ("Razón corriente", LIQ, CLR["rojo"], 2),
                                       ("Z''", ZC, CLR["gris_oscuro"], 2)]:
            s = serie_medianas(df_serie, col)
            series_txt[met] = s.set_index("Año")["Mediana"]
            fig_evol.add_trace(go.Scatter(
                x=s["Año"], y=s["Mediana"], name=met, mode="lines+markers+text", line=dict(width=2.5, color=color),
                text=[fmt(v, 1) for v in s["Mediana"]], textposition="top center", textfont_size=9,
                customdata=s[["P25", "P75", "n"]].values,
                hovertemplate=f"<b>{met}</b> %{{x}}<br>Mediana: %{{y:.2f}}<br>P25–P75: %{{customdata[0]:.2f}} – %{{customdata[1]:.2f}}<br>n = %{{customdata[2]}}<extra></extra>"),
                row=fila, col=1)
        fig_evol.add_hline(y=2.60, row=2, col=1, line_dash="dash", line_color=CLR["verde"], opacity=0.4,
                           annotation_text="Z'' = 2,60", annotation_position="top left", annotation_font_size=9)
        fig_evol.update_layout(**LAYOUT_BASE, height=480)
        fig_evol.update_layout(legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))
        fig_evol.update_traces(cliponaxis=False)
        eje_años(fig_evol)
        st.plotly_chart(fig_evol, use_container_width=True)
        r, mn, lq, zz = series_txt["ROA"], series_txt["Margen neto"], series_txt["Razón corriente"], series_txt["Z''"]
        interp(f"El ROA mediano pasa de {fmt(r.get(2021), 2, ' %')} en 2021 a {fmt(r.get(2024), 2, ' %')} en 2024 "
               f"({'disminuye' if r.get(2024, np.nan) < r.get(2021, np.nan) else 'aumenta'} {fmt(abs(r.get(2024, np.nan) - r.get(2021, np.nan)), 2)} p.p.), "
               f"y el margen neto mediano, de {fmt(mn.get(2021), 2, ' %')} a {fmt(mn.get(2024), 2, ' %')}. La razón corriente mediana se mueve entre "
               f"{fmt(lq.min(), 2, 'x')} y {fmt(lq.max(), 2, 'x')}, y el Z'' mediano entre {fmt(zz.min(), 2)} y {fmt(zz.max(), 2)}, "
               f"siempre por encima del umbral de Zona Segura. Cada año incluye las empresas del sector seleccionado con dato disponible.")

    with col_r:
        seccion(f"Distribución por zona Z'' — {año_sel}")
        dz = pd.DataFrame([(z, zonas_año[z][0]) for z in ZONAS[::-1]] + [(SIN_Z, n_sinz)], columns=["Zona", "n"])
        dz = dz[dz["n"] > 0]
        fig_donut = go.Figure(go.Pie(labels=dz["Zona"], values=dz["n"], hole=0.6, sort=False, textinfo="percent",
                                     marker=dict(colors=[COLOR_ZONA[z] for z in dz["Zona"]], line=dict(color="#FFFFFF", width=2)),
                                     hovertemplate="%{label}<br><b>%{value:,} empresas</b><br>%{percent}<extra></extra>"))
        fig_donut.add_annotation(text=f"<b>{fmt(dz['n'].sum(), 0)}</b><br>empresas", x=0.5, y=0.5, showarrow=False,
                                 font_size=13, font_color=CLR["gris_oscuro"])
        apply_layout(fig_donut, height=360)
        fig_donut.update_layout(legend=dict(orientation="h", y=-0.02, yanchor="top", x=0.5, xanchor="center"), margin=dict(t=20, b=90, l=10, r=10))
        st.plotly_chart(fig_donut, use_container_width=True)
        interp("El gris corresponde a empresas sin Z'' calculado. Los porcentajes del gráfico incluyen ese grupo; "
               "los de las tarjetas superiores se calculan solo sobre empresas con Z''.")

    seccion(f"Empresas por sector — {año_sel}")
    dist_sec = df_año.groupby("Sector_Corto").size().reset_index(name="n").sort_values("n")
    fig_sec = go.Figure(go.Bar(x=dist_sec["n"], y=dist_sec["Sector_Corto"], orientation="h", text=dist_sec["n"],
                               textposition="outside", textfont_size=10,
                               marker_color=[CLR["gris_claro"] if v < N_MIN_SECTOR else CLR["azul_claro"] for v in dist_sec["n"]]))
    apply_layout(fig_sec, height=max(300, 24 * len(dist_sec)))
    st.plotly_chart(fig_sec, use_container_width=True)
    top2 = dist_sec.sort_values("n", ascending=False).head(2)
    interp(f"Comercio y manufactura dominan la muestra: {top2.iloc[0]['Sector_Corto']} ({fmt(top2.iloc[0]['n'], 0)}) y "
           f"{top2.iloc[1]['Sector_Corto']} ({fmt(top2.iloc[1]['n'], 0)}) suman el "
           f"{fmt(top2['n'].sum() / dist_sec['n'].sum() * 100, 1, ' %')} de las empresas. Las barras grises tienen menos de "
           f"{N_MIN_SECTOR} empresas." if len(top2) == 2 else "Distribución de empresas por sector.")

    seccion("Cambio 2021 → 2024 por indicador (medianas)")
    filas = []
    for etq, col in [("Razón corriente (x)", LIQ), ("Prueba ácida (x)", ACI), ("Pasivos/Activos (%)", PAS_ACT),
                     ("Deuda financiera/Activos (%)", DFA), ("ROA (%)", ROA), ("ROE (%)", ROE), ("Margen neto (%)", MNE),
                     ("Margen EBITDA (%)", MEB), ("Z'' calculado", ZC)]:
        v = {a: df_serie[df_serie["Año"] == a][col].median() for a in AÑOS}
        filas.append({"Indicador": etq, **{str(a): round(v[a], 2) for a in AÑOS}, "Cambio 2021→2024": round(v[2024] - v[2021], 2)})
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LIQUIDEZ (Pregunta 1)
# ═══════════════════════════════════════════════════════════════════════════════
with t_liq:
    pregunta(1, "¿Cómo se comporta la liquidez por sector económico?")
    liq_m, aci_m, efe_m = df_año[LIQ].median(), df_año[ACI].median(), df_año[EFE].median()
    clase = lambda v, a, b: "kpi-safe" if v >= a else ("kpi-warn" if v >= b else "kpi-danger")
    ka, kb, kc, kd = st.columns(4)
    ka.markdown(kpi_html("Razón corriente (mediana)", fmt(liq_m, 2, "x"), "Referencia: ≥ 1,5x holgura · < 1,0x presión", clase(liq_m, 1.5, 1.0)), True)
    kb.markdown(kpi_html("Prueba ácida (mediana)", fmt(aci_m, 2, "x"), "Sin inventarios", clase(aci_m, 1.0, 0.8)), True)
    kc.markdown(kpi_html("Razón de efectivo (mediana)", fmt(efe_m, 2, "x"), "Solo caja y equivalentes", clase(efe_m, 0.5, 0.2)), True)
    kd.markdown(kpi_html("Empresas con razón corriente < 1", fmt((df_año[LIQ] < 1).mean() * 100, 1, " %"),
                         f"de {fmt(df_año[LIQ].notna().sum(), 0)} con dato", "kpi-info"), True)
    with st.expander("¿Qué miden estos indicadores?"):
        e1, e2, e3 = st.columns(3)
        with e1: ficha("razon_corriente")
        with e2: ficha("prueba_acida")
        with e3: ficha("razon_efectivo")

    col_l2, col_r2 = st.columns([1.2, 0.8])
    with col_l2:
        seccion(f"Razón corriente y prueba ácida por sector — {año_sel} (medianas)")
        g1, g2 = resumen_por_grupo(df_año, "Sector_Corto", LIQ), resumen_por_grupo(df_año, "Sector_Corto", ACI)
        g1 = g1.sort_values("Mediana", ascending=False)
        g2 = g2.set_index("Sector_Corto").reindex(g1["Sector_Corto"]).reset_index()
        fig_liq = go.Figure()
        for g, nombre, color in [(g1, "Razón corriente", CLR["azul_claro"]), (g2, "Prueba ácida", CLR["verde_claro"])]:
            fig_liq.add_trace(go.Bar(name=nombre, x=g["Sector_Corto"], y=g["Mediana"], marker_color=color,
                                     text=[fmt(v, 2) for v in g["Mediana"]], textposition="outside", textfont_size=9,
                                     customdata=g[["P25", "P75", "n"]].values, hovertemplate=HOVER_MED))
        fig_liq.add_hline(y=1.0, line_dash="dash", line_color=CLR["rojo"], opacity=0.6, annotation_text="1,0x", annotation_position="top left", annotation_font_size=9)
        fig_liq.add_hline(y=1.5, line_dash="dash", line_color=CLR["verde"], opacity=0.4, annotation_text="1,5x", annotation_position="top left", annotation_font_size=9)
        fig_liq.update_layout(**LAYOUT_BASE, barmode="group", height=400, xaxis_tickangle=-35)
        st.plotly_chart(fig_liq, use_container_width=True)
        if len(g1):
            bajo1 = g1[g1["Mediana"] < 1]["Sector_Corto"].tolist()
            interp(f"En {año_sel}, la mayor liquidez mediana está en {g1.iloc[0]['Sector_Corto']} ({fmt(g1.iloc[0]['Mediana'], 2, 'x')}) y la menor en "
                   f"{g1.iloc[-1]['Sector_Corto']} ({fmt(g1.iloc[-1]['Mediana'], 2, 'x')}). "
                   + (f"Sectores con mediana por debajo de 1,0x: {', '.join(bajo1)}. " if bajo1 else "Ningún sector tiene una mediana inferior a 1,0x. ")
                   + "La distancia entre las dos barras de un sector indica cuánto de su liquidez depende de los inventarios." + nota_n(g1))

    with col_r2:
        seccion("Razón corriente vs. prueba ácida por empresa")
        sc = df_año.dropna(subset=[LIQ, ACI])
        lim_x, lim_y = sc[LIQ].quantile(.98), sc[ACI].quantile(.98)
        fig_scat = px.scatter(sc, x=LIQ, y=ACI, color="Zona", color_discrete_map=COLOR_ZONA, opacity=0.6,
                              hover_data={"Compañía": True, "Sector_Corto": True},
                              labels={LIQ: "Razón corriente (x)", ACI: "Prueba ácida (x)", "Zona": "Zona Z''"})
        fig_scat.update_traces(marker_size=5)
        fig_scat.add_vline(x=1.0, line_dash="dot", line_color=CLR["rojo"], opacity=0.5)
        fig_scat.add_hline(y=1.0, line_dash="dot", line_color=CLR["rojo"], opacity=0.5)
        apply_layout(fig_scat, height=400)
        fig_scat.update_xaxes(range=[0, lim_x]); fig_scat.update_yaxes(range=[0, lim_y])
        st.plotly_chart(fig_scat, use_container_width=True)
        cuad = ((sc[LIQ] < 1) & (sc[ACI] < 1))
        crit_cuad = (sc.loc[cuad, "Zona"] == "🔴 Zona Crítica").mean() * 100 if cuad.any() else np.nan
        crit_resto = (sc.loc[~cuad & (sc["Zona"] != SIN_Z), "Zona"] == "🔴 Zona Crítica").mean() * 100
        interp(f"{fmt(cuad.sum(), 0)} empresas ({fmt(cuad.mean() * 100, 1, ' %')}) tienen ambos indicadores por debajo de 1,0x (cuadrante inferior izquierdo). "
               f"De ellas, el {fmt(crit_cuad, 1, ' %')} está en Zona Crítica, frente al {fmt(crit_resto, 1, ' %')} del resto con Z''. "
               "Ejes acotados al percentil 98 para facilitar la lectura.")

    seccion("Evolución de la razón corriente por sector 2021–2024 (medianas, 8 sectores con más empresas)")
    top_sec = df_serie.groupby("Sector_Corto")["Compañía"].nunique().nlargest(8).index
    lev = (df_serie[df_serie["Sector_Corto"].isin(top_sec)].groupby(["Año", "Sector_Corto"])[LIQ].median().reset_index())
    fig_lev = px.line(lev, x="Año", y=LIQ, color="Sector_Corto", markers=True, color_discrete_sequence=COLOR_SEQ,
                      labels={LIQ: "Razón corriente (mediana)", "Sector_Corto": "Sector"})
    fig_lev.add_hline(y=1.0, line_dash="dash", line_color=CLR["rojo"], opacity=0.5, annotation_text=" 1,0x")
    apply_layout(fig_lev, height=360); eje_años(fig_lev)
    st.plotly_chart(fig_lev, use_container_width=True)
    camb = lev.pivot(index="Sector_Corto", columns="Año", values=LIQ)
    if 2021 in camb and 2024 in camb:
        d_c = (camb[2024] - camb[2021]).sort_values()
        interp(f"Entre 2021 y 2024, la mayor caída de la razón corriente mediana se observa en {d_c.index[0]} ({fmt(d_c.iloc[0], 2, 'x')}) "
               f"y el mayor aumento en {d_c.index[-1]} ({'+' if d_c.iloc[-1] > 0 else ''}{fmt(d_c.iloc[-1], 2, 'x')}).")


# ═══════════════════════════════════════════════════════════════════════════════
# ENDEUDAMIENTO Y RIESGO Z'' (Preguntas 2 y 5)
# ═══════════════════════════════════════════════════════════════════════════════
with t_rie:
    pregunta(2, "¿Qué sectores presentan mayor nivel de endeudamiento relativo?")
    pa_m, dfa_m = df_año[PAS_ACT].median(), df_año[DFA].median()
    sin_deuda = (df_año[DFA] == 0).mean() * 100
    z_m = df_año[ZC].median()
    kz = st.columns(4)
    kz[0].markdown(kpi_html("Pasivos / Activos (mediana)", fmt(pa_m, 1, " %"), "Endeudamiento total",
                            "kpi-safe" if pa_m < 40 else ("kpi-warn" if pa_m <= 60 else "kpi-danger")), True)
    kz[1].markdown(kpi_html("Deuda financiera / Activos", fmt(dfa_m, 1, " %"), "Mediana (EMIS)", "kpi-info"), True)
    kz[2].markdown(kpi_html("Sin deuda financiera", fmt(sin_deuda, 1, " %"), "Empresas con deuda financiera = 0", "kpi-info"), True)
    kz[3].markdown(kpi_html("Z'' calculado (mediana)", fmt(z_m, 2), "Zona Segura: > 2,60",
                            "kpi-safe" if z_m > 2.6 else ("kpi-warn" if z_m >= 1.1 else "kpi-danger")), True)
    with st.expander("¿Qué miden estos indicadores?"):
        e1, e2, e3 = st.columns(3)
        with e1: ficha("pasivos_activos")
        with e2: ficha("deuda_activos")
        with e3: ficha("deuda_capital")

    seccion(f"Endeudamiento por sector — {año_sel} (medianas)")
    gpa = resumen_por_grupo(df_año, "Sector_Corto", PAS_ACT).sort_values("Mediana", ascending=False)
    gdf = resumen_por_grupo(df_año, "Sector_Corto", DFA).set_index("Sector_Corto").reindex(gpa["Sector_Corto"]).reset_index()
    fig_end = go.Figure()
    fig_end.add_trace(go.Bar(name="Pasivos / Activos (%)", x=gpa["Sector_Corto"], y=gpa["Mediana"], marker_color=CLR["gris_oscuro"],
                             text=[fmt(v, 1) for v in gpa["Mediana"]], textposition="outside", textfont_size=9,
                             customdata=gpa[["P25", "P75", "n"]].values, hovertemplate=HOVER_MED))
    fig_end.add_trace(go.Bar(name="Deuda financiera / Activos (%)", x=gdf["Sector_Corto"], y=gdf["Mediana"], marker_color=CLR["amarillo"],
                             text=[fmt(v, 1) for v in gdf["Mediana"]], textposition="outside", textfont_size=9,
                             customdata=gdf[["P25", "P75", "n"]].values, hovertemplate=HOVER_MED))
    fig_end.add_hline(y=60, line_dash="dash", line_color=CLR["rojo"], opacity=0.5, annotation_text="60 % (endeudamiento total alto)", annotation_position="top left", annotation_font_size=9)
    fig_end.update_layout(**LAYOUT_BASE, barmode="group", height=400, xaxis_tickangle=-35, yaxis_title="%")
    st.plotly_chart(fig_end, use_container_width=True)
    if len(gpa):
        sobre60 = gpa[gpa["Mediana"] > 60]["Sector_Corto"].tolist()
        gdf_ok = gdf.dropna(subset=["Mediana"]).sort_values("Mediana", ascending=False)
        interp(f"En {año_sel}, el mayor endeudamiento total mediano está en {gpa.iloc[0]['Sector_Corto']} ({fmt(gpa.iloc[0]['Mediana'], 1, ' %')}) y el menor en "
               f"{gpa.iloc[-1]['Sector_Corto']} ({fmt(gpa.iloc[-1]['Mediana'], 1, ' %')}). "
               + (f"Sectores con pasivos superiores al 60 % de los activos: {', '.join(sobre60)}. " if sobre60 else "")
               + (f"La mayor deuda financiera relativa está en {gdf_ok.iloc[0]['Sector_Corto']} ({fmt(gdf_ok.iloc[0]['Mediana'], 1, ' %')}). " if len(gdf_ok) else "")
               + "La deuda financiera es solo una parte de los pasivos: un sector puede tener pasivos altos con proveedores y poca deuda bancaria." + nota_n(gpa))

    st.markdown("---")
    pregunta(5, "¿Existen empresas con señales de alerta financiera (Z'' de Altman)?")
    kz2 = st.columns(4)
    for col_k, z, cls in [(kz2[0], "🟢 Zona Segura", "kpi-safe"), (kz2[1], "🟡 Zona Alerta", "kpi-warn"), (kz2[2], "🔴 Zona Crítica", "kpi-danger")]:
        col_k.markdown(kpi_html(z, f"{fmt(zonas_año[z][0], 0)}", f"{fmt(zonas_año[z][1], 1, ' %')} de {fmt(n_conz, 0)} con Z''", cls), True)
    kz2[3].markdown(kpi_html(SIN_Z, f"{fmt(n_sinz, 0)}", "Falta 'Resultados acumulados' en EMIS", "kpi-info"), True)
    with st.expander("¿Qué es el Z'' de Altman y por qué se usa el calculado y no el de EMIS?"):
        e1, e2 = st.columns([1, 1])
        with e1: ficha("zscore")
        with e2:
            interp("El Z'' se calcula en el cuaderno con una fórmula conocida y los estados financieros de EMIS. Así se puede "
                   "<b>descomponer en X1–X4</b> y explicar por qué cada empresa está en su zona. EMIS reporta otro puntaje "
                   "con un método no documentado. Ambos ordenan a las empresas de forma muy parecida (Spearman ≈ 0,98, zona "
                   "coincidente en el 89–93 % de los casos), pero <b>no son iguales</b>: en 2024, con las mismas empresas, EMIS ubica el "
                   "21,8 % en Zona Crítica y el cálculo propio el 17,7 %. El costo de la decisión es la cobertura: el Z'' calculado "
                   "no está disponible para el 12 % de las empresas.", "Decisión metodológica")

    col_zl, col_zr = st.columns([1.1, 0.9])
    with col_zl:
        seccion(f"% de empresas en Zona Crítica por sector — {año_sel}")
        con_z = df_año[df_año["Zona"] != SIN_Z]
        rz = con_z.groupby("Sector_Corto").agg(n=("Zona", "size"), crit=("Zona", lambda s: (s == "🔴 Zona Crítica").mean() * 100),
                                               z_med=(ZC, "median")).reset_index()
        cob = df_año.groupby("Sector_Corto").size().rename("total")
        rz = rz.join(cob, on="Sector_Corto")
        rz["sin_z"] = (1 - rz["n"] / rz["total"]) * 100
        if ocultar_peq:
            rz = rz[rz["n"] >= N_MIN_SECTOR]
        rz = rz.sort_values("crit", ascending=True)
        fig_cr = go.Figure(go.Bar(x=rz["crit"], y=rz["Sector_Corto"], orientation="h", marker_color=CLR["rojo"],
                                  text=[f"{v:.1f} %" for v in rz["crit"]], textposition="outside", textfont_size=9,
                                  customdata=rz[["n", "z_med", "sin_z"]].values,
                                  hovertemplate="<b>%{y}</b><br>% en Zona Crítica: %{x:.1f}<br>Empresas con Z'': %{customdata[0]}"
                                                "<br>Z'' mediano: %{customdata[1]:.2f}<br>% sin Z'': %{customdata[2]:.0f}<extra></extra>"))
        apply_layout(fig_cr, height=max(320, 26 * len(rz)))
        fig_cr.update_xaxes(title_text="% de empresas con Z'' en Zona Crítica")
        st.plotly_chart(fig_cr, use_container_width=True)
        if len(rz):
            alta_falta = rz[rz["sin_z"] > 30]["Sector_Corto"].tolist()
            interp(f"En {año_sel}, las mayores proporciones en Zona Crítica están en {rz.iloc[-1]['Sector_Corto']} ({fmt(rz.iloc[-1]['crit'], 1, ' %')}) y "
                   f"{rz.iloc[-2]['Sector_Corto'] if len(rz) > 1 else ''} ({fmt(rz.iloc[-2]['crit'] if len(rz) > 1 else np.nan, 1, ' %')}). "
                   + (f"Cuidado: en {', '.join(alta_falta)} más del 30 % de las empresas no tiene Z'', por lo que el porcentaje puede no representar al sector. " if alta_falta else "")
                   + "Pase el cursor para ver el número de empresas y la cobertura.")

    with col_zr:
        seccion(f"Zonas Z'' por tamaño empresarial — {año_sel} (UVT del año)")
        tz = con_z.groupby(["Tamaño (UVT del año)", "Zona"]).size().reset_index(name="n")
        tz["pct"] = tz["n"] / tz.groupby("Tamaño (UVT del año)")["n"].transform("sum")
        tot_t = con_z.groupby("Tamaño (UVT del año)").size()
        tz["Tamaño"] = tz["Tamaño (UVT del año)"].map(lambda t: f"{t} (n={fmt(tot_t[t], 0)})")
        fig_rk = px.bar(tz, x="Tamaño", y="pct", color="Zona", barmode="stack", color_discrete_map=COLOR_ZONA,
                        category_orders={"Zona": ZONAS[::-1]}, labels={"pct": "Proporción", "Zona": "Zona Z''"})
        fig_rk.update_layout(**LAYOUT_BASE, height=300, yaxis_tickformat=".0%")
        st.plotly_chart(fig_rk, use_container_width=True)
        interp("El tamaño se calcula con los ingresos y el UVT de cada año. Como la muestra son las mayores empresas por ingresos "
               "de 2024, en ese año casi todas son 'Grandes'; cambie el año en la barra lateral para ver más medianas. "
               "Los grupos con menos de 30 empresas son solo referenciales.")

        seccion("Empresas con el Z'' más bajo (sin alerta de calidad de dato)")
        low = con_z[~con_z["Alerta_Calidad_Dato"]].nsmallest(12, ZC)[["Compañía", "Sector_Corto", ZC]]
        fig_qui = go.Figure(go.Bar(x=low[ZC], y=low["Compañía"].str[:35], orientation="h", marker_color=CLR["rojo"],
                                   text=[fmt(v, 2) for v in low[ZC]], textposition="outside", textfont_size=9,
                                   customdata=low[["Sector_Corto"]].values,
                                   hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Z'' = %{x:.2f}<extra></extra>"))
        apply_layout(fig_qui, height=360); fig_qui.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_qui, use_container_width=True)
        interp("Lista descriptiva para explorar casos. Un Z'' muy bajo es una señal que conviene revisar con los estados financieros "
               "de la empresa; no implica que la empresa esté en quiebra.")

    st.markdown("---")
    seccion("Mapa de calor: % de empresas en Zona Crítica por sector y año")
    serie_z = df_serie[df_serie["Zona"] != SIN_Z]
    heat = serie_z.groupby(["Sector_Corto", "Año"]).agg(crit=("Zona", lambda s: (s == "🔴 Zona Crítica").mean() * 100), n=("Zona", "size")).reset_index()
    if ocultar_peq:
        heat = heat[heat["n"] >= N_MIN_SECTOR]
    hp = heat.pivot(index="Sector_Corto", columns="Año", values="crit")
    hn = heat.pivot(index="Sector_Corto", columns="Año", values="n").reindex_like(hp)
    if año_sel in hp.columns:
        hp = hp.sort_values(año_sel, ascending=False); hn = hn.reindex(hp.index)
    fig_heat = go.Figure(go.Heatmap(z=hp.values, x=[str(c) for c in hp.columns], y=hp.index.tolist(),
                                    colorscale=[[0, "#FFFFFF"], [0.5, CLR["amarillo"]], [1, CLR["rojo"]]],
                                    text=np.round(hp.values, 1), texttemplate="%{text}%", textfont_size=10, customdata=hn.values,
                                    hovertemplate="%{y} — %{x}<br><b>%{z:.1f} % en Zona Crítica</b><br>n con Z'' = %{customdata}<extra></extra>",
                                    colorbar=dict(title=dict(text="% Crítica", side="right"))))
    fig_heat.update_layout(**LAYOUT_BASE, height=max(360, 26 * len(hp)))
    fig_heat.update_xaxes(type="category")
    st.plotly_chart(fig_heat, use_container_width=True)
    interp("Cada celda es el porcentaje de empresas con Z'' del sector que están en Zona Crítica ese año. Celdas que se mantienen "
           "altas en los cuatro años indican sectores donde el riesgo es persistente en la muestra.")

    st.markdown("---")
    seccion(f"¿Qué componente del Z'' es más bajo en cada sector? — {año_sel}")
    zona_dec = st.radio("Zona analizada", ZONAS + ["Todas las zonas"], index=0, horizontal=True)
    comp = {"X1 — Liquidez (CT/AT)": "Contrib_X1", "X2 — Utilidades acumuladas (UR/AT)": "Contrib_X2",
            "X3 — Rentabilidad operativa (EBIT/AT)": "Contrib_X3", "X4 — Solvencia (Pat/Pas)": "Contrib_X4"}
    base_dec = con_z if zona_dec == "Todas las zonas" else con_z[con_z["Zona"] == zona_dec]
    dec = base_dec.groupby("Sector_Corto")[list(comp.values())].median().rename(columns={v: k for k, v in comp.items()})
    dec["n"] = base_dec.groupby("Sector_Corto").size()
    dec = dec[dec["n"] >= N_MIN_SECTOR]
    if len(dec):
        fig_dec = go.Figure()
        for i, c in enumerate(comp):
            fig_dec.add_trace(go.Bar(name=c, x=[f"{s} (n={n})" for s, n in zip(dec.index, dec["n"])], y=dec[c],
                                     marker_color=[CLR["azul_claro"], CLR["gris_oscuro"], CLR["rojo"], CLR["verde"]][i]))
        fig_dec.add_hline(y=0, line_color="#1A1A1A", line_width=0.8)
        fig_dec.update_layout(**LAYOUT_BASE, barmode="group", height=400, xaxis_tickangle=-35, yaxis_title="Contribución mediana al Z''")
        st.plotly_chart(fig_dec, use_container_width=True)
        dec["Componente más bajo"] = dec[list(comp)].idxmin(axis=1)
        cuenta = dec["Componente más bajo"].str[:2].value_counts()
        interp(f"Se muestran solo sectores con al menos {N_MIN_SECTOR} empresas en la zona elegida. El componente más bajo es "
               + ", ".join(f"{k} en {v} sector(es)" for k, v in cuenta.items())
               + ". X1 bajo indica capital de trabajo neto estrecho o negativo; X2 bajo, pérdidas acumuladas. "
                 "Como la zona se define con la suma de estos componentes, la lectura es descriptiva, no causal.")
        st.dataframe(dec[["n", "Componente más bajo"] + list(comp)].round(3), use_container_width=True)
    else:
        interp(f"No hay sectores con al menos {N_MIN_SECTOR} empresas en la zona seleccionada con los filtros actuales.")

    st.markdown("---")
    seccion("Matriz de transición de zonas Z'' — 2021 → 2024")
    w = wide_raw.copy()
    if sector_sel:
        w = w[w["Sector_Corto"].isin(sector_sel)]
    w = w[w["Zona_ZScore 2021"].notna() & w["Zona_ZScore 2024"].notna()]
    if len(w):
        z21, z24 = w["Zona_ZScore 2021"].map(MAPA_ZONA), w["Zona_ZScore 2024"].map(MAPA_ZONA)
        cnt = pd.crosstab(z21, z24).reindex(index=ZONAS, columns=ZONAS, fill_value=0)
        pct = cnt.div(cnt.sum(axis=1).replace(0, np.nan), axis=0) * 100
        ct1, ct2 = st.columns([1.2, 0.8])
        with ct1:
            fig_mtz = go.Figure(go.Heatmap(z=pct.values, x=ZONAS, y=ZONAS, zmin=0, zmax=100,
                                           colorscale=[[0, "#FFFFFF"], [1, CLR["gris_oscuro"]]],
                                           text=[[f"{p:.1f} %<br>({fmt(c, 0)})" for p, c in zip(rp, rc)] for rp, rc in zip(pct.values, cnt.values)],
                                           texttemplate="%{text}", textfont=dict(size=12),
                                           hovertemplate="2021: %{y}<br>2024: %{x}<br>%{text}<extra></extra>",
                                           colorbar=dict(title=dict(text="% de la fila"))))
            fig_mtz.update_layout(**LAYOUT_BASE, height=340, xaxis_title="Zona en 2024", yaxis_title="Zona en 2021")
            st.plotly_chart(fig_mtz, use_container_width=True)
        with ct2:
            orden = {z: i for i, z in enumerate(ZONAS)}
            mov = np.where(z24.map(orden) > z21.map(orden), "Mejoró", np.where(z24.map(orden) < z21.map(orden), "Empeoró", "Permaneció"))
            mc = pd.Series(mov).value_counts().reindex(["Mejoró", "Permaneció", "Empeoró"]).fillna(0).astype(int)
            fig_mov = go.Figure(go.Bar(x=mc.index, y=mc.values, marker_color=[CLR["verde"], CLR["gris_oscuro"], CLR["rojo"]],
                                       text=[f"{fmt(v, 0)}<br>({v / mc.sum() * 100:.1f} %)" for v in mc.values], textposition="outside"))
            apply_layout(fig_mov, height=340); fig_mov.update_yaxes(title_text="Empresas")
            st.plotly_chart(fig_mov, use_container_width=True)
        interp(f"Con {fmt(len(w), 0)} empresas con zona en 2021 y 2024: el {fmt(pct.loc['🔴 Zona Crítica', '🔴 Zona Crítica'], 1, ' %')} de las que estaban "
               f"en Zona Crítica en 2021 sigue en ella en 2024, y la Zona Segura retiene al {fmt(pct.loc['🟢 Zona Segura', '🟢 Zona Segura'], 1, ' %')}. "
               "Son frecuencias observadas en esta muestra, no probabilidades. La muestra solo incluye empresas que en 2024 siguen entre "
               "las de mayores ingresos, por lo que no refleja a las que salieron del mercado.")


# ═══════════════════════════════════════════════════════════════════════════════
# RENTABILIDAD Y DUPONT (Pregunta 3)
# ═══════════════════════════════════════════════════════════════════════════════
with t_ren:
    pregunta(3, "¿Cuáles son las empresas y los sectores más rentables en términos de ROA, ROE y margen neto?")
    kr = st.columns(4)
    roe_validas = df_año[~df_año["Patrimonio_Negativo"]][ROE]
    kr[0].markdown(kpi_html("ROA (mediana)", fmt(df_año[ROA].median(), 1, " %"), "Utilidad / activos", "kpi-info"), True)
    kr[1].markdown(kpi_html("ROE (mediana)", fmt(roe_validas.median(), 1, " %"), "Excluye patrimonio ≤ 0", "kpi-info"), True)
    kr[2].markdown(kpi_html("Margen neto (mediana)", fmt(df_año[MNE].median(), 1, " %"), "De cada $100 vendidos", "kpi-info"), True)
    kr[3].markdown(kpi_html("Empresas con pérdidas", fmt((df_año[ROA] < 0).mean() * 100, 1, " %"), "ROA negativo", "kpi-info"), True)
    with st.expander("¿Qué miden estos indicadores?"):
        e1, e2 = st.columns(2)
        with e1: ficha("roa"); ficha("margen_neto")
        with e2: ficha("roe"); ficha("margen_ebitda")

    col_r1, col_r2 = st.columns([1.1, 0.9])
    with col_r1:
        seccion(f"ROA y margen neto por sector — {año_sel} (medianas)")
        gr = resumen_por_grupo(df_año, "Sector_Corto", ROA).sort_values("Mediana", ascending=False)
        gm = resumen_por_grupo(df_año, "Sector_Corto", MNE).set_index("Sector_Corto").reindex(gr["Sector_Corto"]).reset_index()
        fig_rent = go.Figure()
        fig_rent.add_trace(go.Bar(name="ROA (%)", x=gr["Sector_Corto"], y=gr["Mediana"], marker_color=CLR["azul_claro"],
                                  text=[fmt(v, 1) for v in gr["Mediana"]], textposition="outside", textfont_size=9,
                                  customdata=gr[["P25", "P75", "n"]].values, hovertemplate=HOVER_MED))
        fig_rent.add_trace(go.Scatter(name="Margen neto (%)", x=gm["Sector_Corto"], y=gm["Mediana"], mode="lines+markers",
                                      line=dict(color=CLR["amarillo"], width=2.5), marker_size=8,
                                      customdata=gm[["P25", "P75", "n"]].values, hovertemplate=HOVER_MED))
        apply_layout(fig_rent, height=400); fig_rent.update_layout(xaxis_tickangle=-35, yaxis_title="%")
        st.plotly_chart(fig_rent, use_container_width=True)
        if len(gr):
            interp(f"En {año_sel}, el ROA mediano más alto está en {gr.iloc[0]['Sector_Corto']} ({fmt(gr.iloc[0]['Mediana'], 1, ' %')}, n = {gr.iloc[0]['n']}) "
                   f"y el más bajo en {gr.iloc[-1]['Sector_Corto']} ({fmt(gr.iloc[-1]['Mediana'], 1, ' %')}). Un margen alto con ROA bajo indica "
                   "muchos activos por cada peso vendido (baja rotación); lo contrario, un negocio de volumen." + nota_n(gr))

    with col_r2:
        seccion(f"15 empresas con mayor ROA — {año_sel}")
        st.caption("Excluye empresas con alerta de calidad de dato o con patrimonio ≤ 0.")
        top = df_año[~df_año["Alerta_Calidad_Dato"] & ~df_año["Patrimonio_Negativo"]].nlargest(15, ROA)
        fig_top = go.Figure(go.Bar(x=top[ROA], y=top["Compañía"].str[:32], orientation="h", marker_color=CLR["verde"],
                                   text=[fmt(v, 1, " %") for v in top[ROA]], textposition="outside", textfont_size=9,
                                   customdata=top[["Sector_Corto", ROE, MNE]].values,
                                   hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>ROA: %{x:.1f} %<br>ROE: %{customdata[1]:.1f} %"
                                                 "<br>Margen neto: %{customdata[2]:.1f} %<extra></extra>"))
        apply_layout(fig_top, height=430); fig_top.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")
    seccion(f"Descomposición DuPont del ROE por sector — {año_sel} (medianas, patrimonio > 0)")
    dp = df_año[~df_año["Patrimonio_Negativo"]].dropna(subset=["DuPont_Margen", "DuPont_Rotacion", "DuPont_Apalancamiento"])
    univ = dp[["DuPont_Margen", "DuPont_Rotacion", "DuPont_Apalancamiento"]].median()
    dps = dp.groupby("Sector_Corto").agg(n=("ROE_DuPont_pct", "size"), Margen=("DuPont_Margen", "median"),
                                         Rotacion=("DuPont_Rotacion", "median"), Apalancamiento=("DuPont_Apalancamiento", "median"),
                                         ROE=("ROE_DuPont_pct", "median"))
    if ocultar_peq:
        dps = dps[dps["n"] >= N_MIN_SECTOR]

    def _ventaja(r):
        if r["Margen"] <= 0:
            return "Margen neto ≤ 0"
        rel = np.log(np.array([r["Margen"], r["Rotacion"], r["Apalancamiento"]]) / univ.values)
        return ["Margen neto", "Rotación de activos", "Apalancamiento"][int(np.argmax(rel))]
    if len(dps):
        dps["Mayor ventaja relativa"] = dps.apply(_ventaja, axis=1)
        dps = dps.sort_values("ROE", ascending=False)
        fig_dp = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.03,
                               subplot_titles=("ROE (%)", "Margen neto (%)", "Rotación de activos (x)", "Apalancamiento (x)"))
        for j, (col, mult, color, ref) in enumerate([("ROE", 1, CLR["rojo"], None), ("Margen", 100, CLR["azul_claro"], univ["DuPont_Margen"] * 100),
                                                     ("Rotacion", 1, CLR["verde_claro"], univ["DuPont_Rotacion"]),
                                                     ("Apalancamiento", 1, CLR["amarillo"], univ["DuPont_Apalancamiento"])], start=1):
            fig_dp.add_trace(go.Bar(x=dps[col] * mult, y=dps.index, orientation="h", marker_color=color, showlegend=False,
                                    text=[fmt(v, 2) for v in dps[col] * mult], textposition="outside", textfont_size=8,
                                    hovertemplate="%{y}: %{x:.2f}<extra></extra>"), row=1, col=j)
            if ref is not None:
                fig_dp.add_vline(x=ref, line_dash="dot", line_color="#1A1A1A", opacity=0.5, row=1, col=j)
        fig_dp.update_layout(**LAYOUT_BASE, height=max(380, 26 * len(dps)))
        fig_dp.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_dp, use_container_width=True)
        st.dataframe(dps[["n", "ROE", "Mayor ventaja relativa"]].round(2), use_container_width=True)
        cnt_v = dps["Mayor ventaja relativa"].value_counts()
        interp("Cada componente tiene su propia escala; la línea punteada es la mediana del universo. La 'mayor ventaja relativa' es el "
               "componente en el que el sector más supera esa mediana: " + ", ".join(f"{k} ({v} sectores)" for k, v in cnt_v.items())
               + f". El sector con mayor ROE mediano es {dps.index[0]} ({fmt(dps.iloc[0]['ROE'], 1, ' %')}, n = {int(dps.iloc[0]['n'])}).")
    with st.expander("¿Cómo funciona la descomposición DuPont?"):
        ficha("dupont")


# ═══════════════════════════════════════════════════════════════════════════════
# PERFILES FINANCIEROS (K-Means)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="⏳ Proyectando empresas en el plano PCA…")
def coordenadas_pca(_wide, _pp, _pca):
    X = _wide[_pp["features"]].clip(_pp["lim_inf"], _pp["lim_sup"], axis=1)
    Xz = pd.DataFrame(_pp["scaler_pre"].transform(X), columns=_pp["features"], index=X.index)
    Xs = _pp["scaler_final"].transform(_pp["imputador_knn"].transform(Xz))
    return _pca.transform(Xs), _pca.explained_variance_ratio_

with t_clu:
    st.markdown(f"<div class='interp-box'><b>¿Qué muestra esta pestaña?</b> Un algoritmo (K-Means) agrupó a las empresas según sus "
                f"11 indicadores de liquidez, rentabilidad y endeudamiento de 2021 a 2024, <b>sin usar el Z''</b>. El resultado son tres "
                f"perfiles financieros. Las 17 empresas con valores extremos que distorsionaban el modelo (p. ej., Federación Colombiana de "
                f"Municipios y Metro de Bogotá) se trataron como <b>suplementarias</b>: no participaron en la construcción del modelo, pero "
                f"se muestran en el gráfico (estrellas) y siguen en el resto del tablero.</div>", unsafe_allow_html=True)
    wc = wide_raw.copy()
    if sector_sel:
        wc = wc[wc["Sector_Corto"].isin(sector_sel)]
    faltan = [k for k in ["preproceso_clustering", "pca_2d"] if k not in modelos]
    if not faltan:
        coords, var = coordenadas_pca(wide_raw, modelos["preproceso_clustering"], modelos["pca_2d"])
        wide_pca = wide_raw.assign(PC1=coords[:, 0], PC2=coords[:, 1])
        wc = wide_pca.loc[wc.index]
        seccion(f"Mapa de perfiles financieros (PCA: el plano resume el {fmt(var.sum() * 100, 1)} % de la variación)")
        fig_pca = go.Figure()
        for lab, g in wc[wc["Rol_Modelo_Multivariado"] == "Activa"].groupby("Cluster_Label"):
            fig_pca.add_trace(go.Scatter(x=g["PC1"], y=g["PC2"], mode="markers", name=f"{lab} ({fmt(len(g), 0)})",
                                         marker=dict(size=5, color=COLOR_CLUSTER.get(lab, CLR["gris_medio"]), opacity=0.55),
                                         customdata=g[["Compañía", "Sector_Corto"]].values,
                                         hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"))
        sup = wc[wc["Rol_Modelo_Multivariado"] == "Suplementaria"]
        if len(sup):
            fig_pca.add_trace(go.Scatter(x=sup["PC1"], y=sup["PC2"], mode="markers", name=f"Suplementarias ({len(sup)})",
                                         marker=dict(symbol="star", size=13, color="#1A1A1A"),
                                         customdata=sup[["Compañía", "Sector_Corto", "Cluster_Label"]].values,
                                         hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Más cercana a: %{customdata[2]}<extra></extra>"))
        apply_layout(fig_pca, height=480)
        fig_pca.update_xaxes(title_text=f"PC1 ({fmt(var[0] * 100, 1)} %) — a la derecha: más márgenes y liquidez, menos deuda")
        fig_pca.update_yaxes(title_text=f"PC2 ({fmt(var[1] * 100, 1)} %) — arriba: más deuda financiera, menos liquidez")
        st.plotly_chart(fig_pca, use_container_width=True)
        interp("Cada punto es una empresa; las cercanas tienen perfiles parecidos. El plano es una vista parcial, porque resume cerca del 40 % "
               "de la variación: grupos que se superponen aquí pueden diferenciarse en otras dimensiones. Las suplementarias se ubican "
               "después de acotar sus valores extremos, y su grupo asignado es solo referencial.")
    else:
        st.info("No se encontraron los artefactos del modelo en 'modelos_exportados/' (" + ", ".join(faltan) +
                "). Se muestran solo los perfiles calculados en el cuaderno.")

    seccion(f"Perfil mediano de cada grupo — {año_sel} (empresas activas)")
    act = df_año[df_año["Rol_Modelo_Multivariado"] == "Activa"]
    perf_cols = {LIQ: "Razón corriente (x)", ROA: "ROA (%)", ROE: "ROE (%)", MOP: "Margen operacional (%)", MNE: "Margen neto (%)",
                 DFA: "Deuda financiera/Activos (%)", PAS_ACT: "Pasivos/Activos (%)", ZC: "Z'' calculado"}
    perf = act.groupby("Cluster_Label")[list(perf_cols)].median().rename(columns=perf_cols)
    perf.insert(0, "Empresas", act.groupby("Cluster_Label").size())
    perf.index.name = "Perfil financiero"
    st.dataframe(perf.round(2), use_container_width=True)

    seccion(f"Grupos vs. zonas Z'' — {año_sel}")
    cz = act[act["Zona"] != SIN_Z]
    if len(cz):
        ctab = pd.crosstab(cz["Cluster_Label"], cz["Zona"], normalize="index").reindex(columns=ZONAS[::-1], fill_value=0) * 100
        fig_cz = go.Figure()
        for z in ZONAS[::-1]:
            fig_cz.add_trace(go.Bar(name=z, y=ctab.index, x=ctab[z], orientation="h", marker_color=COLOR_ZONA[z],
                                    text=[f"{v:.0f} %" for v in ctab[z]], textposition="inside"))
        fig_cz.update_layout(**LAYOUT_BASE, barmode="stack", height=260, xaxis_title="% de empresas del grupo")
        st.plotly_chart(fig_cz, use_container_width=True)
        interp("Los grupos se formaron sin usar el Z'', así que esta es una comparación externa. La asociación es moderada: "
               "el grupo de alta rentabilidad está casi todo en Zona Segura, mientras que el de mayor deuda financiera se reparte "
               "entre las tres zonas. Los perfiles complementan al Z'', no lo reemplazan.")
    if len(sup := df_año[df_año["Rol_Modelo_Multivariado"] == "Suplementaria"]):
        with st.expander(f"Empresas suplementarias en la selección ({len(sup)})"):
            st.dataframe(sup[["Compañía", "Sector_Corto", "Cluster_Label", ROA, MNE, LIQ]].rename(columns={"Cluster_Label": "Grupo más cercano"}),
                         use_container_width=True, hide_index=True)
    with st.expander("Metodología de la segmentación"):
        ficha("kmeans")


# ═══════════════════════════════════════════════════════════════════════════════
# FICHA DE EMPRESA
# ═══════════════════════════════════════════════════════════════════════════════
with t_emp:
    st.markdown("<div class='interp-box'><b>Ficha de empresa:</b> seleccione una empresa para ver su evolución 2021–2024 frente a la "
                "mediana de su sector y del universo. Puede escribir parte del nombre en el buscador.</div>", unsafe_allow_html=True)
    empresas = sorted(panel_raw["Compañía"].unique())
    emp = st.selectbox("Empresa", empresas, index=None, placeholder="Escriba el nombre de la empresa…")
    if emp:
        e = panel_raw[panel_raw["Compañía"] == emp].sort_values("Año")
        sec = e["Sector_Corto"].iloc[0]
        e_sel = e[e["Año"] == año_sel]
        st.markdown(f"""
        <div style='background:#FFFFFF;border:1px solid {CLR["gris_claro"]};border-left:4px solid {CLR["rojo"]};border-radius:0;
                    padding:.8rem 1.2rem;margin:.6rem 0;'>
          <div style='font-weight:700;font-size:1.05rem;color:{CLR["gris_oscuro"]};'>{emp}</div>
          <div style='font-size:.8rem;color:#636363;margin-top:.2rem;'>Sector: <b>{e["Sección CIIU"].iloc[0]}</b> ·
          Perfil financiero: <b>{e["Cluster_Label"].iloc[0]}</b> ({e["Rol_Modelo_Multivariado"].iloc[0].lower()} en el modelo) ·
          Tamaño {año_sel}: <b>{e_sel["Tamaño (UVT del año)"].iloc[0] if len(e_sel) else "—"}</b></div></div>""", unsafe_allow_html=True)
        zc = st.columns(4)
        for i, a in enumerate(AÑOS):
            r = e[e["Año"] == a]
            if len(r):
                z, zona = r[ZC].iloc[0], r["Zona"].iloc[0]
                cls = {"🟢 Zona Segura": "kpi-safe", "🟡 Zona Alerta": "kpi-warn", "🔴 Zona Crítica": "kpi-danger"}.get(zona, "kpi-info")
                alerta = " · ⚠️ alerta de calidad de dato" if r["Alerta_Calidad_Dato"].iloc[0] else ""
                zc[i].markdown(kpi_html(f"Z'' {a}", fmt(z, 2), zona + alerta, cls), True)
        ind = st.selectbox("Indicador", ["Razón corriente (x)", "Prueba ácida (x)", "Pasivos/Activos (%)", "Deuda financiera/Activos (%)",
                                         "ROA (%)", "ROE (%)", "Margen neto (%)", "Margen EBITDA (%)", "Z'' calculado"], index=0)
        col_ind = {"Razón corriente (x)": LIQ, "Prueba ácida (x)": ACI, "Pasivos/Activos (%)": PAS_ACT, "Deuda financiera/Activos (%)": DFA,
                   "ROA (%)": ROA, "ROE (%)": ROE, "Margen neto (%)": MNE, "Margen EBITDA (%)": MEB, "Z'' calculado": ZC}[ind]
        med_sec = panel_raw[panel_raw["Sector_Corto"] == sec].groupby("Año")[col_ind].median()
        med_uni = panel_raw.groupby("Año")[col_ind].median()
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=e["Año"], y=e[col_ind], name=emp[:30], mode="lines+markers+text", line=dict(color=CLR["rojo"], width=3),
                                   text=[fmt(v, 2) for v in e[col_ind]], textposition="top center"))
        fig_e.add_trace(go.Scatter(x=med_sec.index, y=med_sec.values, name=f"Mediana {sec}", mode="lines+markers",
                                   line=dict(color=CLR["azul_claro"], dash="dot")))
        fig_e.add_trace(go.Scatter(x=med_uni.index, y=med_uni.values, name="Mediana del universo", mode="lines",
                                   line=dict(color=CLR["gris_medio"], dash="dash")))
        apply_layout(fig_e, title=ind, height=380); eje_años(fig_e)
        st.plotly_chart(fig_e, use_container_width=True)
        v_e = e.set_index("Año")[col_ind]
        if v_e.notna().sum() >= 2:
            pos = "por encima" if v_e.get(año_sel, np.nan) > med_sec.get(año_sel, np.nan) else "por debajo"
            interp(f"En {año_sel}, {emp} registra {fmt(v_e.get(año_sel), 2)} en {ind.lower()}, {pos} de la mediana de su sector "
                   f"({fmt(med_sec.get(año_sel), 2)}). Entre el primer y el último año con dato, el indicador pasa de "
                   f"{fmt(v_e.dropna().iloc[0], 2)} a {fmt(v_e.dropna().iloc[-1], 2)}.")
        seccion("Componentes del Z'' por año (contribución ponderada)")
        comp_e = e.set_index("Año")[["Contrib_X1", "Contrib_X2", "Contrib_X3", "Contrib_X4"]]
        if comp_e.notna().any().any():
            fig_ce = go.Figure()
            for c, nom, col in [("Contrib_X1", "X1 Liquidez", CLR["azul_claro"]), ("Contrib_X2", "X2 Utilidades acumuladas", CLR["gris_oscuro"]),
                                ("Contrib_X3", "X3 Rentabilidad operativa", CLR["rojo"]), ("Contrib_X4", "X4 Solvencia", CLR["verde"])]:
                fig_ce.add_trace(go.Bar(name=nom, x=comp_e.index, y=comp_e[c], marker_color=col))
            fig_ce.add_hline(y=0, line_color="#1A1A1A", line_width=0.8)
            fig_ce.update_layout(**LAYOUT_BASE, barmode="relative", height=320)
            eje_años(fig_ce)
            st.plotly_chart(fig_ce, use_container_width=True)
            interp("La suma de las cuatro barras de cada año es el Z''. Una barra negativa resta puntaje: por ejemplo, X1 negativo indica "
                   "capital de trabajo neto negativo.")
        else:
            interp("Esta empresa no tiene Z'' calculado porque EMIS no reporta sus resultados acumulados.")
        tabla_e = e.set_index("Año")[[LIQ, ACI, PAS_ACT, DFA, ROA, ROE, MNE, MEB, ZC]].T
        tabla_e.index = ["Razón corriente (x)", "Prueba ácida (x)", "Pasivos/Activos (%)", "Deuda financiera/Activos (%)", "ROA (%)",
                         "ROE (%)", "Margen neto (%)", "Margen EBITDA (%)", "Z'' calculado"]
        tabla_e.columns = [str(c) for c in tabla_e.columns]
        st.dataframe(tabla_e.round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULADOR
# ═══════════════════════════════════════════════════════════════════════════════
with t_sim:
    st.markdown(f"""
    <div style='background:#FFFFFF;border:1px solid {CLR["gris_claro"]};border-radius:0;padding:1rem 1.4rem;margin-bottom:1rem;
                border-left:4px solid {CLR["rojo"]};'>
      <div style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;font-weight:700;font-size:1.05rem;color:{CLR["gris_oscuro"]};'>Simulador financiero</div>
      <div style='font-size:.8rem;color:#636363;line-height:1.55;'>Ingrese las cifras de una empresa (millones de COP) para calcular sus
      indicadores, su Z'' de Altman y su descomposición DuPont, y compararla con las empresas del sector en {año_sel}.
      Es un ejercicio descriptivo: no predice el futuro de la empresa.</div></div>""", True)
    seccion("Datos financieros de la empresa")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("**Balance general**")
        activos_totales = st.number_input("Activos totales", min_value=0.0, value=5000.0, step=100.0, format="%.1f")
        pasivos_totales = st.number_input("Pasivos totales", min_value=0.0, value=2500.0, step=100.0, format="%.1f")
        patrimonio = st.number_input("Patrimonio total", value=2500.0, step=100.0, format="%.1f")
        ut_retenidas = st.number_input("Utilidades retenidas (resultados acumulados)", value=500.0, step=50.0, format="%.1f")
        deuda_fin = st.number_input("Deuda financiera (bancos y bonos)", min_value=0.0, value=800.0, step=50.0, format="%.1f")
    with f2:
        st.markdown("**Estado de resultados**")
        ventas = st.number_input("Ingresos operativos", min_value=0.0, value=8000.0, step=100.0, format="%.1f")
        utilidad_neta = st.number_input("Utilidad neta", value=400.0, step=50.0, format="%.1f")
        ebit = st.number_input("EBIT (utilidad operativa)", value=600.0, step=50.0, format="%.1f")
        ebitda = st.number_input("EBITDA", value=900.0, step=50.0, format="%.1f")
    with f3:
        st.markdown("**Corto plazo**")
        activos_corrientes = st.number_input("Activos corrientes", min_value=0.0, value=2000.0, step=100.0, format="%.1f")
        pasivos_corrientes = st.number_input("Pasivos corrientes", min_value=0.0, value=1200.0, step=100.0, format="%.1f")
        inventarios = st.number_input("Inventarios", min_value=0.0, value=400.0, step=50.0, format="%.1f")
        efectivo = st.number_input("Efectivo y equivalentes", min_value=0.0, value=300.0, step=50.0, format="%.1f")
        sector_sim = st.selectbox("Sector de comparación", sorted(panel_raw["Sector_Corto"].unique()))
    if abs(activos_totales - pasivos_totales - patrimonio) > 0.01 * max(activos_totales, 1):
        st.warning("Activos ≠ Pasivos + Patrimonio: revise las cifras del balance.")
    calcular = st.button("Calcular diagnóstico", type="primary", use_container_width=True)

    if calcular:
        div = lambda a, b: a / b if b else np.nan
        capital_trabajo = activos_corrientes - pasivos_corrientes
        rc, pa_, re_ = div(activos_corrientes, pasivos_corrientes), div(activos_corrientes - inventarios, pasivos_corrientes), div(efectivo, pasivos_corrientes)
        pas_act, dfa_ = div(pasivos_totales, activos_totales) * 100, div(deuda_fin, activos_totales) * 100
        roa_, roe_ = div(utilidad_neta, activos_totales) * 100, div(utilidad_neta, patrimonio) * 100
        mn_, meb_, mop_ = div(utilidad_neta, ventas) * 100, div(ebitda, ventas) * 100, div(ebit, ventas) * 100
        X = [div(capital_trabajo, activos_totales), div(ut_retenidas, activos_totales), div(ebit, activos_totales), div(patrimonio, pasivos_totales)]
        pesos = [6.56, 3.26, 6.72, 1.05]
        zscore = float(np.nansum([w * x for w, x in zip(pesos, X)]))
        zona_txt = "🟢 Zona Segura" if zscore > 2.60 else ("🟡 Zona Alerta" if zscore >= 1.10 else "🔴 Zona Crítica")
        zcol = COLOR_ZONA[zona_txt]
        st.markdown(f"""
        <div style='background:#FFFFFF;border:1px solid {zcol};border-left:6px solid {zcol};border-radius:0;padding:1rem 1.4rem;
                    margin-bottom:1rem;text-align:center;'>
          <div style='font-size:1.8rem;font-weight:700;color:{zcol};'>{zona_txt}</div>
          <div style='font-size:1.25rem;font-weight:600;color:#1A1A1A;'>Z'' = {zscore:.3f}</div>
          <div style='font-size:.78rem;color:#636363;margin-top:.3rem;'>Capital de trabajo calculado = activos corrientes − pasivos corrientes =
          {fmt(capital_trabajo, 1)} millones COP</div></div>""", True)
        r1, r2, r3 = st.columns(3)
        with r1:
            seccion("Componentes del Z''")
            comp_df = pd.DataFrame({"Componente": ["X1 Capital de trabajo / Activos", "X2 Utilidades retenidas / Activos",
                                                   "X3 EBIT / Activos", "X4 Patrimonio / Pasivos"],
                                    "Valor": np.round(X, 4), "Peso": pesos})
            comp_df["Contribución"] = (comp_df["Valor"] * comp_df["Peso"]).round(3)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            debil = comp_df.loc[comp_df["Contribución"].idxmin(), "Componente"]
            interp(f"El componente con menor aporte es <b>{debil}</b>.")
        with r2:
            seccion("Indicadores calculados")
            st.dataframe(pd.DataFrame({"Indicador": ["Razón corriente (x)", "Prueba ácida (x)", "Razón de efectivo (x)", "Pasivos/Activos (%)",
                                                     "Deuda financiera/Activos (%)", "ROA (%)", "ROE (%)", "Margen neto (%)", "Margen EBITDA (%)"],
                                       "Valor": np.round([rc, pa_, re_, pas_act, dfa_, roa_, roe_, mn_, meb_], 2)}),
                         use_container_width=True, hide_index=True)
        with r3:
            seccion("Descomposición DuPont")
            m_dp, r_dp, a_dp = div(utilidad_neta, ventas), div(ventas, activos_totales), div(activos_totales, patrimonio)
            st.markdown(f"<div class='interp-box' style='text-align:center;'>ROE = {fmt(m_dp * 100, 2, ' %')} × {fmt(r_dp, 2, 'x')} × "
                        f"{fmt(a_dp, 2, 'x')} = <b>{fmt(m_dp * r_dp * a_dp * 100, 2, ' %')}</b></div>", True)
            ref = panel_raw[(panel_raw["Año"] == año_sel) & ~panel_raw["Patrimonio_Negativo"]][["DuPont_Margen", "DuPont_Rotacion", "DuPont_Apalancamiento"]].median()
            if patrimonio > 0 and m_dp > 0:
                rel = np.log(np.array([m_dp, r_dp, a_dp]) / ref.values)
                comp_v = ["margen neto", "rotación de activos", "apalancamiento"][int(np.argmax(rel))]
                interp(f"Frente a la mediana del universo en {año_sel}, la mayor ventaja relativa de la empresa está en el <b>{comp_v}</b>.")
            else:
                interp("Con patrimonio ≤ 0 o margen neto ≤ 0 la descomposición DuPont no es interpretable.")

        if "arbol_decision_zscore" in modelos and "features_arbol" in modelos:
            vals = {LIQ: rc, ACI: pa_, EFE: re_, DFA: dfa_, DCAP: div(deuda_fin, patrimonio) * 100,
                    DNE: div(deuda_fin - efectivo, ebitda), ROA: roa_, ROE: roe_, MOP: mop_, MNE: mn_, MEB: meb_}
            X_arb = pd.DataFrame([{f: vals.get(f.rsplit(" ", 1)[0], np.nan) for f in modelos["features_arbol"]}])
            try:
                zona_arbol = modelos["arbol_decision_zscore"].predict(X_arb)[0]
                interp(f"Con sus indicadores, el mapa de reglas (árbol de decisión de 3 niveles) ubica a la empresa en <b>{MAPA_ZONA.get(zona_arbol, zona_arbol)}</b>. "
                       "El árbol es un resumen aproximado de la regla del Z'' (coincide en ≈ 72 % de las empresas de la muestra); "
                       "la clasificación de referencia es el Z'' calculado arriba.", "Mapa de reglas")
            except Exception as ex:
                st.caption(f"No fue posible aplicar el árbol exportado: {ex}")

        seccion(f"Comparación con las empresas de {sector_sim} en {año_sel}")
        ref_sec = panel_raw[(panel_raw["Año"] == año_sel) & (panel_raw["Sector_Corto"] == sector_sim) & ~panel_raw["Alerta_Calidad_Dato"]]
        filas = []
        for etq, col, val in [("Razón corriente (x)", LIQ, rc), ("Pasivos/Activos (%)", PAS_ACT, pas_act), ("ROA (%)", ROA, roa_),
                              ("Margen neto (%)", MNE, mn_), ("Z'' calculado", ZC, zscore)]:
            s = ref_sec[col].dropna()
            filas.append({"Indicador": etq, "Su empresa": round(val, 2) if pd.notna(val) else None,
                          "Mediana del sector": round(s.median(), 2) if len(s) else None,
                          "Rango P25–P75 del sector": f"{s.quantile(.25):.2f} – {s.quantile(.75):.2f}" if len(s) else "—",
                          "Percentil de su empresa en el sector": f"{(s < val).mean() * 100:.0f}" if len(s) and pd.notna(val) else "—"})
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
        interp(f"El percentil indica qué porcentaje de las {fmt(len(ref_sec), 0)} empresas del sector tiene un valor menor que el de su empresa. "
               "En Pasivos/Activos, un percentil alto significa más endeudamiento que la mayoría.")

        seccion("Señales identificadas")
        senales = [("🟢 El Z'' está en Zona Segura." if zscore > 2.6 else
                    ("🟡 El Z'' está en Zona Alerta: conviene revisar los componentes con menor aporte." if zscore >= 1.1 else
                     "🔴 El Z'' está en Zona Crítica: los componentes con menor aporte señalan dónde se concentra la debilidad."))]
        if pd.notna(rc) and rc < 1:
            senales.append("🔴 Razón corriente inferior a 1,0x: los activos corrientes no cubren los pasivos corrientes.")
        elif pd.notna(rc) and rc < 1.5:
            senales.append("🟡 Razón corriente entre 1,0x y 1,5x: cobertura de corto plazo ajustada.")
        if pd.notna(pas_act) and pas_act > 60:
            senales.append("🔴 Pasivos superiores al 60 % de los activos: endeudamiento total alto.")
        if pd.notna(roa_) and roa_ < 2:
            senales.append("🟡 ROA inferior a 2 %: baja utilidad por cada peso de activos.")
        if pd.notna(a_dp) and a_dp > 3:
            senales.append("🟡 Multiplicador de apalancamiento mayor a 3x: el ROE depende en buena medida del endeudamiento.")
        for s_ in senales:
            interp(s_, "Señal")
        st.caption("Las señales se basan en rangos orientadores generales y no sustituyen un análisis financiero de la empresa.")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOSARIO Y METODOLOGÍA
# ═══════════════════════════════════════════════════════════════════════════════
with t_glo:
    buscar = st.text_input("🔍 Buscar indicador", placeholder="Ej.: ROA, liquidez, Z''…")
    CATEGORIAS = {"Liquidez": ["razon_corriente", "prueba_acida", "razon_efectivo"],
                  "Endeudamiento": ["pasivos_activos", "deuda_activos", "deuda_capital"],
                  "Rentabilidad": ["roa", "roe", "margen_neto", "margen_ebitda", "dupont"],
                  "Riesgo y segmentación": ["zscore", "kmeans"]}
    t_ = buscar.lower().strip()
    for cat, claves in CATEGORIAS.items():
        cl = [k for k in claves if not t_ or t_ in (FICHAS[k]["nombre"] + FICHAS[k]["tag"] + FICHAS[k]["simple"]).lower()]
        if not cl:
            continue
        seccion(cat)
        st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:.8rem;margin-bottom:1.2rem;'>"
                    + "".join(ficha_as_html(**FICHAS[k]) for k in cl) + "</div>", unsafe_allow_html=True)

    st.markdown("---")
    seccion("Metodología en breve")
    st.markdown(f"""
    <div class='interp-box' style='line-height:1.8;'>
    <b>Universo:</b> 3.000 empresas con mayores ingresos operativos de 2024 (EMIS) → {fmt(N_EMPRESAS_TOTAL, 0)} empresas tras exigir activos, pasivos e
    ingresos en los cuatro años, sección CIIU identificada y como máximo 50 % de indicadores faltantes. No se eliminan empresas por valores extremos.<br>
    <b>Sector:</b> letra del primer código CIIU entre paréntesis de 'Sector (Local)'.<br>
    <b>Tamaño:</b> Decreto 957 de 2019 con los ingresos y el UVT de cada año.<br>
    <b>Estadístico central:</b> mediana y rango P25–P75, porque los indicadores tienen valores extremos que distorsionan el promedio.
    Las medianas sectoriales se interpretan solo con al menos {N_MIN_SECTOR} empresas.<br>
    <b>Riesgo:</b> Z'' de Altman calculado (no el de EMIS); sin imputación; porcentajes sobre empresas con Z''.<br>
    <b>Perfiles:</b> K-Means (K = 3) sobre 44 indicadores; 17 empresas suplementarias por un criterio de influencia (&gt; 5 % de la varianza de un eje del PCA).<br>
    <b>Calidad de datos:</b> alertas por ROA &gt; 100 %, ingresos &gt; 50 veces los activos, margen neto EMIS inconsistente o razón corriente &gt; 100.<br>
    <b>Alcance:</b> caso de estudio descriptivo; los resultados dependen de estas decisiones y describen a las mayores empresas, no al conjunto del país.
    </div>""", unsafe_allow_html=True)

    seccion("Términos clave")
    TERMINOS = [
        ("Mediana", "Valor de la empresa que queda en el medio al ordenar a todas. A diferencia del promedio, no se mueve por unos pocos valores extremos."),
        ("Rango P25–P75", "Intervalo donde se ubica el 50 % central de las empresas. Muestra cuán parecidas o distintas son entre sí."),
        ("Capital de trabajo", "Activos corrientes menos pasivos corrientes: recursos de corto plazo que quedan después de cubrir las deudas de corto plazo."),
        ("Utilidades retenidas", "Ganancias acumuladas que no se repartieron como dividendos (en EMIS, 'Resultados acumulados')."),
        ("EBIT / EBITDA", "Utilidad operativa antes de intereses e impuestos; el EBITDA, además, antes de depreciaciones y amortizaciones."),
        ("Sección CIIU", "Letra de la Clasificación Industrial Internacional Uniforme que agrupa actividades económicas (C = manufactura, G = comercio…)."),
        ("Decreto 957 de 2019", "Norma colombiana que clasifica el tamaño de las empresas por ingresos en UVT según el macrosector."),
        ("Empresa suplementaria", "Empresa con valores tan extremos que, por sí sola, distorsionaba el modelo de perfiles. No participa en su construcción, pero se proyecta y se muestra."),
        ("Alerta de calidad de dato", "Marca de empresas-año con cifras implausibles en EMIS. Se conservan, pero se excluyen de los rankings."),
    ]
    g1, g2 = st.columns(2)
    for i, (term, defn) in enumerate(TERMINOS):
        (g1 if i % 2 == 0 else g2).markdown(
            f"<div style='background:#FFFFFF;border-radius:0;padding:.7rem 1rem;border-left:3px solid {CLR['rojo']};margin-bottom:.6rem;'>"
            f"<div style='color:{CLR['rojo']};font-weight:600;font-size:.85rem;'>{term}</div>"
            f"<div style='color:#1A1A1A;font-size:.8rem;line-height:1.6;'>{defn}</div></div>", unsafe_allow_html=True)

    seccion("Referencias")
    refs = [
        "Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. <i>The Journal of Finance, 23</i>(4), 589–609.",
        "Altman, E. I., Hartzell, J. y Peck, M. (1995). <i>Emerging markets corporate bonds: A scoring system</i>. Salomon Brothers. (Z'' de cuatro variables y modelo EMS).",
        "Altman, E. I. y Hotchkiss, E. (2006). <i>Corporate financial distress and bankruptcy</i> (3.ª ed.). Wiley.",
        "Breiman, L., Friedman, J., Olshen, R. y Stone, C. (1984). <i>Classification and regression trees</i>. Wadsworth.",
        "Brigham, E. F. y Houston, J. F. (2014). <i>Fundamentals of financial management</i>. Cengage Learning.",
        "Chapman, P. et al. (2000). <i>CRISP-DM 1.0: Step-by-step data mining guide</i>. SPSS.",
        "Few, S. (2012). <i>Show me the numbers</i> (2.ª ed.). Analytics Press.",
        "Husson, F., Lê, S. y Pagès, J. (2017). <i>Exploratory multivariate analysis by example using R</i> (2.ª ed.). CRC Press. (Individuos suplementarios en PCA).",
        "Jolliffe, I. T. (2002). <i>Principal component analysis</i> (2.ª ed.). Springer.",
        "MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. <i>Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1</i>, 281–297.",
        "Ministerio de Comercio, Industria y Turismo (2019). <i>Decreto 957 de 2019</i>, criterios de clasificación de las micro, pequeñas, medianas y grandes empresas.",
        "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. <i>Journal of Computational and Applied Mathematics, 20</i>, 53–65.",
        "Troyanskaya, O. et al. (2001). Missing value estimation methods for DNA microarrays. <i>Bioinformatics, 17</i>(6), 520–525. (Imputación KNN).",
        "Tufte, E. R. (2001). <i>The visual display of quantitative information</i> (2.ª ed.). Graphics Press.",
    ]
    for r_ in refs:
        st.markdown(f"<div style='margin-bottom:.45rem;font-size:.8rem;color:#1A1A1A;line-height:1.6;padding-left:1.2rem;text-indent:-1.2rem;'>{r_}</div>", True)


# ─────────────────────────────────────────────────────────────────────────────
# PIE DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='border-top:1px solid {CLR["gris_claro"]};margin:2rem 0 .8rem;padding-top:.9rem;display:flex;
            justify-content:space-between;flex-wrap:wrap;gap:.5rem;'>
  <div style='font-size:.72rem;color:#636363;line-height:1.75;'>
    <span style='color:{CLR["rojo"]};font-weight:700;'>Universidad del Valle</span> · Maestría en Analítica e Inteligencia de Negocios<br>
    Trabajo de grado — caso de estudio de analítica descriptiva del desempeño financiero (2021–2024)</div>
  <div style='font-size:.7rem;color:#636363;text-align:right;line-height:1.7;'>
    Z'' de Altman calculado · Perfiles K-Means (K = 3) · Metodología CRISP-DM<br>
    Fuente: EMIS Platform · {fmt(N_EMPRESAS_TOTAL, 0)} empresas · {N_SECTORES_TOTAL} secciones CIIU · COP millones</div>
</div>""", unsafe_allow_html=True)
