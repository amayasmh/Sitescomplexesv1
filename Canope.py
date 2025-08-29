import io
import json
import gzip
import requests
import pandas as pd
import geopandas as gpd
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional

import folium  # on intègre la carte en HTML statique (pas de st_folium)
from folium.plugins import MarkerCluster

st.set_page_config(page_title="Carte des communes par lot & Sites Complexes", layout="wide")
# Titre de l'application
st.title("Carte des communes par lot & Sites Complexes")
st.caption("Le fichier local 'Communes_notifiees.csv' est chargé automatiquement. "
           "Choisissez un lot (obligatoire), puis affinez par département et commune. "
           "Cliquez sur **Valider** pour appliquer les filtres.")

# -----------------------------
# SOURCES
# -----------------------------
CSV_PATH = "Communes_notifiees.csv"
COMMUNES_GEOJSON_URL = (
    "https://adresse.data.gouv.fr/data/contours-administratifs/2023/geojson/communes-100m.geojson.gz"
)

# -----------------------------
# HELPERS & CACHE


def _make_label_html() -> str:
    return f"""
    <span style="display:inline-block;color:white;border-radius:50%;width:24px;height:24px;line-height:24px;text-align:center;font-size:14px;font-weight:bold;margin-right:8px;border:1px solid #888;vertical-align:middle;"></span>
    """

@st.cache_data(show_spinner=False)
def load_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "code_insee" in df.columns:
        df["code_insee"] = df["code_insee"].astype(str).str.zfill(5)
    return df

@st.cache_data(show_spinner=True)
def fetch_communes_geojson() -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(COMMUNES_GEOJSON_URL)
    except Exception:
        r = requests.get(COMMUNES_GEOJSON_URL, timeout=60)
        r.raise_for_status()
        with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as f:
            data = json.loads(f.read().decode("utf-8"))
        gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
    keep = [c for c in ["code", "nom", "geometry"] if c in gdf.columns]
    gdf = gdf[keep].rename(columns={"code": "code_insee", "nom": "nom_commune_ref"})
    gdf["code_insee"] = gdf["code_insee"].astype(str).str.zfill(5)
    return gdf

@st.cache_data(show_spinner=False)
def load_raw_centers(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def _detect_lat_lon_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    lat_candidates = ["lat", "latitude", "Latitude", "LAT", "Lat"]
    lon_candidates = ["lon", "lng", "longitude", "Longitude", "LON", "Long", "LONG"]
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    return lat_col, lon_col

@st.cache_data(show_spinner=False)
def load_raw_centers(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def _detect_lat_lon_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    lat_candidates = [
        "lat", "latitude", "Latitude", "LAT", "Lat"
    ]
    lon_candidates = [
        "lon", "lng", "longitude", "Longitude", "LON", "Long", "LONG"
    ]
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    return lat_col, lon_col

@st.cache_data(show_spinner=False)
def build_filtered_gdf(df: pd.DataFrame, _communes_gdf: gpd.GeoDataFrame,
                       lot: Optional[str], dep: Optional[str], communes_filter: list[str]) -> gpd.GeoDataFrame:
    # éviter le hash de GeoDataFrame en paramètre cache
    communes_gdf = _communes_gdf
    base = df.copy()
    if lot:
        base = base[base["lot"].astype(str) == str(lot)]
    else:
        return gpd.GeoDataFrame(columns=["code_insee", "geometry"], geometry="geometry", crs="EPSG:4326")
    if dep and "code_dep" in base.columns:
        base = base[base["code_dep"].astype(str) == str(dep)]
    if communes_filter:
        base = base[base["nom_commune"].isin(communes_filter)]
    merged = communes_gdf.merge(base, on="code_insee", how="inner")
    return merged

def _normalize_filters(lot, dep, communes):
    return {
        "lot": lot or None,
        "dep": dep or None,
        "communes": tuple(sorted(communes or [])),
    }

# -----------------------------
# LOAD DATA
# -----------------------------
try:
    df = load_csv(CSV_PATH)
except Exception as e:
    st.error(f"Erreur lors de la lecture du CSV '{CSV_PATH}' : {e}")
    st.stop()

required_cols = {"code_insee", "nom_commune", "lot"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Colonnes manquantes dans le CSV : {', '.join(sorted(missing))}")
    st.stop()

with st.spinner("Téléchargement des contours de communes (IGN/BAN)…"):
    communes_gdf = fetch_communes_geojson()

# -----------------------------
# SESSION STATE (applied vs pending)
# -----------------------------
if "applied" not in st.session_state:
    st.session_state.applied = _normalize_filters(None, None, [])

# -----------------------------
# SIDEBAR (FORM = pas de rerun tant que pas validé)
# -----------------------------
with st.sidebar:
    
    st.image("images/logo_807.png", width=90) 

    st.header("Sites Complexes")
    show_raw_centers = st.checkbox("Afficher tout les centres commerciaux", value=True)
    st.divider()
    st.markdown("### Filtres (validation étape par étape)")

    # Init session state (lots, dep, communes)
    if "applied" not in st.session_state or not isinstance(st.session_state.applied, dict):
        st.session_state.applied = {"lots": tuple(), "dep": None, "communes": tuple()}

    lots_all = sorted(df["lot"].dropna().astype(str).unique().tolist())

    # Étape 1: Lots (multisélection)
    with st.form("lots_form"):
        sel_lots = st.multiselect("1️⃣ Lots (obligatoire)", options=lots_all, default=list(st.session_state.applied.get("lots", ())))
        submit_lots = st.form_submit_button("Valider les lots", use_container_width=True)
        if submit_lots:
            st.session_state.applied = {"lots": tuple(sel_lots), "dep": None, "communes": tuple()}

    applied = st.session_state.applied
    applied_lots = tuple(applied.get("lots", ()))

    # Étape 2: Département (dépend des lots)
    if applied_lots:
        deps = (
            df[df["lot"].astype(str).isin(list(applied_lots))]
            .get("code_dep", pd.Series(dtype=str))
            .dropna().astype(str).sort_values().unique().tolist()
        )
        with st.form("dep_form"):
            sel_dep = st.selectbox("2️⃣ Département (optionnel)", options=[""] + deps, index=([""] + deps).index(applied.get("dep") or ""))
            submit_dep = st.form_submit_button("Valider le département", use_container_width=True)
            if submit_dep:
                st.session_state.applied = {"lots": applied_lots, "dep": (sel_dep or None), "communes": tuple()}

    # Étape 3: Communes (dépend des lots et du département)
    applied = st.session_state.applied
    applied_lots = tuple(applied.get("lots", ()))
    applied_dep = applied.get("dep")
    if applied_lots:
        df_scope = df[df["lot"].astype(str).isin(list(applied_lots))]
        if applied_dep:
            df_scope = df_scope[df_scope["code_dep"].astype(str) == str(applied_dep)]
        communes_options = sorted(df_scope.get("nom_commune", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        with st.form("communes_form"):
            sel_communes = st.multiselect("3️⃣ Communes (optionnel)", options=communes_options, default=list(applied.get("communes", ())))
            submit_communes = st.form_submit_button("Valider les communes", use_container_width=True)
            if submit_communes:
                st.session_state.applied = {"lots": applied_lots, "dep": applied_dep, "communes": tuple(sel_communes)}

# filtres appliqués
applied = st.session_state.applied
applied_lots = tuple(applied.get("lots", ()))
applied_dep = applied.get("dep")
applied_communes = list(applied.get("communes", ()))

# Palette de couleurs (sans rouge ni jaune) et mapping lot -> couleur
palette_no_red_yellow = [
    "#1f77b4",  # bleu
    "#2ca02c",  # vert
    "#9467bd",  # violet
    "#17becf",  # cyan
    "#e377c2",  # rose
    "#7f7f7f",  # gris
    "#8c564b",  # brun
    "#386cb0",  # bleu soutenu
    "#4daf4a",  # vert moyen
    "#984ea3",  # violet moyen
]

lot_to_color = {}
if applied_lots:
    for idx, lot_val in enumerate(applied_lots):
        lot_to_color[str(lot_val)] = palette_no_red_yellow[idx % len(palette_no_red_yellow)]

# -----------------------------
# GÉO + JOINTURE (uniquement avec filtres appliqués)
# -----------------------------
gdf = gpd.GeoDataFrame(columns=["code_insee", "geometry"], geometry="geometry", crs="EPSG:4326")
if applied_lots:
    base = df[df["lot"].astype(str).isin(list(applied_lots))].copy()
    if applied_dep and "code_dep" in base.columns:
        base = base[base["code_dep"].astype(str) == str(applied_dep)]
    if applied_communes:
        base = base[base["nom_commune"].isin(applied_communes)]
    gdf = communes_gdf.merge(base, on="code_insee", how="inner")

# -----------------------------
# CARTE (HTML statique => pas d'événements => pas de rerun au survol)
# -----------------------------
left, right = st.columns([8, 2], gap="large")

with left:
    

    # Construire et afficher la carte même sans lot sélectionné
    m = folium.Map(location=[46.6, 2.4], zoom_start=6, tiles="cartodbpositron")
  

    # Centres commerciaux (clusters jaunes fixes, pas de lignes)
    if show_raw_centers:
        try:
            df_raw = load_raw_centers("data_raw.csv")
            lat_col, lon_col = _detect_lat_lon_columns(df_raw)
            name_col = next((c for c in ["nom", "name", "Nom", "centre", "Centre"] if c in df_raw.columns), None)
            if lat_col and lon_col:
                mc = MarkerCluster(
                    name="Centres commerciaux",
                    icon_create_function=(
                        """
                        function (cluster) {
                            var count = cluster.getChildCount();
                            return new L.DivIcon({
                                html: '<div style=\"background:#FFC107; color:#000; border:1px solid #B38600; border-radius:20px; width:40px; height:40px; line-height:40px; text-align:center; font-weight:600;\">' + count + '</div>',
                                className: 'marker-cluster',
                                iconSize: new L.Point(40, 40)
                            });
                        }
                        """
                    ),
                    options={
                        'showCoverageOnHover': False,
                        'spiderLegPolylineOptions': {'opacity': 0, 'weight': 0}
                    }
                ).add_to(m)
                for _, r in df_raw.dropna(subset=[lat_col, lon_col]).iterrows():
                    try:
                        lat = float(r[lat_col])
                        lon = float(r[lon_col])
                    except Exception:
                        continue
                    # Champs potentiels pour l'info-bulle et le popup
                    nom_cc = str(r.get(name_col, "Centre")) if name_col else str(r.get("nom", "Centre"))
                    type_cc = str(r.get("type", r.get("Type", "-")))
                    adresse = str(r.get("adresse_complete", r.get("adresse", "-")))
                    cp = str(r.get("code_postal", r.get("cp", "")))
                    ville = str(r.get("ville", r.get("commune", "")))
                    proprietaire = str(r.get("proprietaire", r.get("Propriétaire", "-")))
                    nb_boutiques = r.get("nb_boutiques", r.get("Nb_boutiques", r.get("magasins", r.get("shops", "-"))))
                    anciennete_val = r.get("anciennete", r.get("age", "-"))
                    superficie = r.get("surface_gla", r.get("superficie", r.get("surface", "-")))

                    # Tooltip (survol) en texte simple multi-lignes
                    tooltip_lines = [
                        f"{nom_cc}",
                        f"Type : {type_cc}",
                        f"Adresse : {adresse} - {cp} {ville}",
                        f"Propriétaire : {proprietaire}",
                        f"Boutiques : {nb_boutiques}",
                        f"Ancienneté : {anciennete_val}",
                        f"Superficie : {superficie}",
                    ]
                    tooltip_text = "\n".join([str(x) for x in tooltip_lines if x is not None])

                    # Style App.py: lecture des champs spécifiques si disponibles
                    # Adresse complète (adresse1..3)
                    if all(col in df_raw.columns for col in ["adresse1", "adresse2", "adresse3"]):
                        adresse_parts = [r.get(f"adresse{i}") for i in range(1, 4) if pd.notna(r.get(f"adresse{i}"))]
                        adresse_complete = ", ".join([str(x) for x in adresse_parts]) if adresse_parts else adresse
                    else:
                        adresse_complete = adresse

                    # Champs alignés App.py
                    ville_app = r.get("nom_ville", ville)
                    code_postal_app = r.get("code_postal", cp)
                    anciennete_app = r.get("nb_annees_ouverture", anciennete_val)
                    nb_boutiques_app = r.get("nb_boutiques", nb_boutiques)
                    type_cc_app = r.get("typologie_cc_long", type_cc)
                    proprietaire_app = r.get("gestionnaires", proprietaire)
                    superficie_app = r.get("surface_gla", superficie)

                    # Mise en forme superficie comme App.py (1 234 m²)
                    try:
                        superficie_app = f"{int(float(superficie_app)):,} m²".replace(",", " ")
                    except Exception:
                        pass

                   

                    # HTML au survol (même logique que App.py : HTML mis dans tooltip)
                    popup_html = f"""
                    
                    <b>{nom_cc}</b><br>
                    Type : {type_cc_app}<br>
                    Adresse : {adresse_complete} - {code_postal_app} {ville_app}<br>
                    Propriétaire : {proprietaire_app}<br>
                    Boutiques : {nb_boutiques_app}<br>
                    Ancienneté : {anciennete_app} ans<br>
                    Superficie : {superficie_app}
                    """
                    folium.Marker(
                        location=[lat, lon],
                        tooltip=popup_html,
                        icon=folium.Icon(color="red", icon="shopping-cart", prefix="fa")
                    ).add_to(mc)
        except Exception as e:
            st.warning(f"Impossible d'afficher les centres (data_raw.csv): {e}")

    # Couche communes si filtre renvoie des données (tooltip simple + champs Canopée)
    if not gdf.empty:
        # Formater les dates en français: 31 jan 2025
        def _format_date_fr(x):
            try:
                ts = pd.to_datetime(x, errors="coerce", dayfirst=False)
                if pd.isna(ts):
                    return x
                mois = [
                    "jan", "fév", "mar", "avr", "mai", "juin",
                    "juil", "aoû", "sep", "oct", "nov", "déc"
                ]
                return f"{ts.day} {mois[ts.month-1]} {ts.year}"
            except Exception:
                return x

        for _col in ["fermeture_commerciale", "fermeture_technique"]:
            if _col in gdf.columns:
                gdf[_col] = gdf[_col].apply(_format_date_fr)

        desired_fields = [
            "nom_commune", "code_insee", "nom_departement",
            "lot", "fermeture_commerciale", "fermeture_technique", "code_oi", "nom_oi",
        ]
        tooltip_fields = [c for c in desired_fields if c in gdf.columns]
        aliases_by_field = {
            "nom_commune": "Commune",
            "code_insee": "INSEE",
            "nom_departement": "Département",
            "lot": "Lot",
            "fermeture_commerciale": "Fermeture commerciale",
            "fermeture_technique": "Fermeture technique",
            "code_oi": "Code OI",
            "nom_oi": "Nom OI",
        }
        aliases = [aliases_by_field.get(f, f) for f in tooltip_fields]
        def _style_function(feature):
            lot_val = str(feature.get("properties", {}).get("lot"))
            color = lot_to_color.get(lot_val, "#555555")
            return {
                "fillColor": color,
                "color": color,
                "weight": 1,
                "fillOpacity": 0.45,
            }

        folium.GeoJson(
            data=json.loads(gdf.to_json()),
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=aliases,
                sticky=True
            ) if tooltip_fields else None,
            name=("Lots " + ", ".join(applied_lots)) if len(applied_lots) > 1 else (f"Lot {applied_lots[0]}" if applied_lots else "Communes"),
            style_function=_style_function,
        ).add_to(m)



    folium.LayerControl(collapsed=True).add_to(m)

    # Rendu HTML "statique" : pas d'événements renvoyés à Streamlit
    map_html = m.get_root().render()
    components.html(map_html, width=1300, height=650, scrolling=False)

with right:
    if applied_lots and not gdf.empty:
        st.subheader("Légende des lots")
        for lot_val in applied_lots:
            color = lot_to_color.get(str(lot_val), "#555555")
            st.markdown(
                f"<div style='display:flex;align-items:center;margin:6px 0;'>"
                f"<span style='display:inline-block;width:14px;height:14px;border:1px solid #666;background:{color};margin-right:8px;'></span>"
                f"Lot {lot_val}"
                f"</div>",
                unsafe_allow_html=True,
            )

st.divider()
st.markdown(
    "- **Pas de `st_folium`** : la carte est intégrée en HTML → **aucun rerun** sur survol/zoom.\n"
    "- Les filtres ne s’appliquent **qu’au clic** sur **Valider**.\n"
    "- Chargement CSV/GeoJSON et jointures **en cache** pour de meilleures perfs."
)
