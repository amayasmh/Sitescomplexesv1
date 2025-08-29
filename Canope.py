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
st.title("Carte des communes par lot & Sites Complexes")

# -----------------------------
# SOURCES
# -----------------------------
CSV_PATH = "Communes_notifiees.csv"
COMMUNES_GEOJSON_URL = (
    "https://adresse.data.gouv.fr/data/contours-administratifs/2023/geojson/communes-100m.geojson.gz"
)

# -----------------------------
# HELPERS & CACHE
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "code_insee" in df.columns:
        df["code_insee"] = df["code_insee"].astype(str).str.upper().str.strip()
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
    gdf["code_insee"] = gdf["code_insee"].astype(str).str.upper().str.strip()
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
def ensure_centers_have_insee(df_raw: pd.DataFrame, _communes_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Garantit une colonne code_insee dans data_raw :
       - si déjà présente → normalise
       - sinon tente de l’inférer via sjoin (lat/lon -> commune)
    """
    df_raw = df_raw.copy()

    # 1) Détection colonne INSEE existante
    insee_candidates = ["code_insee", "code_commune_insee", "code_commune", "INSEE", "insee"]
    found_col = next((c for c in insee_candidates if c in df_raw.columns), None)
    if found_col:
        df_raw["code_insee"] = df_raw[found_col].astype(str).str.upper().str.strip()

    # 2) Inférence via lon/lat si besoin
    if "code_insee" not in df_raw.columns or df_raw["code_insee"].isna().all():
        lat_col, lon_col = _detect_lat_lon_columns(df_raw)
        if lat_col and lon_col:
            try:
                _df = df_raw.dropna(subset=[lat_col, lon_col]).copy()
                pts = gpd.GeoDataFrame(
                    _df,
                    geometry=gpd.points_from_xy(
                        _df[lon_col].astype(float),
                        _df[lat_col].astype(float),
                    ),
                    crs="EPSG:4326",
                )
                communes_geo = _communes_gdf.set_crs("EPSG:4326", allow_override=True)
                within = gpd.sjoin(pts, communes_geo[["code_insee", "geometry"]], how="left", predicate="within")
                df_raw.loc[within.index, "code_insee"] = within["code_insee"].values
            except Exception:
                pass

    # 3) Normalisation finale
    if "code_insee" in df_raw.columns:
        df_raw["code_insee"] = df_raw["code_insee"].astype(str).str.upper().str.strip()

    return df_raw

def _normalize_filters(lot, dep, communes):
    return {
        "lot": lot or None,
        "dep": dep or None,
        "communes": tuple(sorted(communes or [])),
    }

def _fit_map_to_gdf_or_points(m, gdf: gpd.GeoDataFrame,
                              df_points: Optional[pd.DataFrame],
                              lat_col: Optional[str],
                              lon_col: Optional[str]) -> None:
    """Zoom prioritairement sur l'emprise des polygones; sinon sur l'emprise des points."""
    # 1) Polygones
    if gdf is not None and not gdf.empty and gdf.geometry.notna().any():
        try:
            minx, miny, maxx, maxy = gdf.total_bounds
            m.fit_bounds([[miny, minx], [maxy, maxx]])
            return
        except Exception:
            pass
    # 2) Points
    if df_points is not None and lat_col and lon_col:
        try:
            pts = df_points.dropna(subset=[lat_col, lon_col])
            if not pts.empty:
                lats = pts[lat_col].astype(float)
                lons = pts[lon_col].astype(float)
                m.fit_bounds([[lats.min(), lons.min()], [lats.max(), lons.max()]])
        except Exception:
            pass

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
# SESSION STATE
# -----------------------------
if "applied" not in st.session_state:
    st.session_state.applied = {"lots": tuple(), "dep": None, "communes": tuple()}
if "widgets_nonce" not in st.session_state:
    st.session_state.widgets_nonce = 0

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.image("images/logo_807.png", width=90)
    st.header("Sites Complexes")
    show_raw_centers = st.checkbox("Afficher tout les centres commerciaux", value=True)
    st.divider()
    st.markdown("### Filtres (validation étape par étape)")

    # Reset filters
    if st.session_state.get("reset_filters", False):
        st.session_state.applied = {"lots": tuple(), "dep": None, "communes": tuple()}
        st.session_state.widgets_nonce = int(st.session_state.widgets_nonce) + 1
        st.session_state.reset_filters = False

    lots_all = sorted(df["lot"].dropna().astype(str).unique().tolist())

    # Étape 1: Lots
    with st.form("lots_form"):
        sel_lots = st.multiselect(
            "1️⃣ Lots (obligatoire)",
            options=lots_all,
            default=list(st.session_state.applied.get("lots", ())),
            key=f"lots_ms_{st.session_state.widgets_nonce}"
        )
        submit_lots = st.form_submit_button("Valider les lots", use_container_width=True)
        if submit_lots:
            st.session_state.applied = {"lots": tuple(sel_lots), "dep": None, "communes": tuple()}

    applied = st.session_state.applied
    applied_lots = tuple(applied.get("lots", ()))

    # Étape 2: Département (optionnel)
    if applied_lots:
        deps = (
            df[df["lot"].astype(str).isin(list(applied_lots))]
            .get("code_dep", pd.Series(dtype=str))
            .dropna().astype(str).sort_values().unique().tolist()
        )
        with st.form("dep_form"):
            sel_dep = st.selectbox(
                "2️⃣ Département (optionnel)",
                options=[""] + deps,
                index=([""] + deps).index(applied.get("dep") or ""),
                key=f"dep_select_{st.session_state.widgets_nonce}"
            )
            submit_dep = st.form_submit_button("Valider le département", use_container_width=True)
            if submit_dep:
                st.session_state.applied = {"lots": applied_lots, "dep": (sel_dep or None), "communes": tuple()}

    # Étape 3: Communes (optionnel)
    applied = st.session_state.applied
    applied_lots = tuple(applied.get("lots", ()))
    applied_dep = applied.get("dep")
    if applied_lots:
        df_scope = df[df["lot"].astype(str).isin(list(applied_lots))]
        if applied_dep:
            df_scope = df_scope[df_scope["code_dep"].astype(str) == str(applied_dep)]
        communes_options = sorted(df_scope.get("nom_commune", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        with st.form("communes_form"):
            sel_communes = st.multiselect(
                "3️⃣ Communes (optionnel)",
                options=communes_options,
                default=list(applied.get("communes", ())),
                key=f"communes_ms_{st.session_state.widgets_nonce}"
            )
            submit_communes = st.form_submit_button("Valider les communes", use_container_width=True)
            if submit_communes:
                st.session_state.applied = {"lots": applied_lots, "dep": applied_dep, "communes": tuple(sel_communes)}

    st.divider()
    if st.button("Réinitialiser les filtres", use_container_width=True):
        st.session_state.reset_filters = True
        st.rerun()

# Filtres appliqués
applied = st.session_state.applied
applied_lots = tuple(applied.get("lots", ()))
applied_dep = applied.get("dep")
applied_communes = list(applied.get("communes", ()))

# Palette et mapping lot -> couleur
palette_no_red_yellow = [
    "#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#e377c2",
    "#7f7f7f", "#8c564b", "#386cb0", "#4daf4a", "#984ea3",
]
lot_to_color = {}
if applied_lots:
    for idx, lot_val in enumerate(applied_lots):
        lot_to_color[str(lot_val)] = palette_no_red_yellow[idx % len(palette_no_red_yellow)]

# -----------------------------
# GÉO + JOINTURE (polygones communes via INSEE)
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
# CARTE
# -----------------------------
left, right = st.columns([8, 2], gap="large")

with left:
    m = folium.Map(location=[46.6, 2.4], zoom_start=6, tiles="cartodbpositron")

    # Centres commerciaux (Markers) — FILTRE PAR COMMUNE (INSEE) quand un lot est appliqué
    filtered_points_for_zoom = None
    lat_col_for_zoom = None
    lon_col_for_zoom = None

    if show_raw_centers:
        try:
            df_raw = load_raw_centers("data_raw.csv")
            lat_col, lon_col = _detect_lat_lon_columns(df_raw)
            name_col = next((c for c in ["nom", "name", "Nom", "centre", "Centre"] if c in df_raw.columns), None)

            # Communes INSEE autorisées selon les filtres
            allowed_insee: set[str] = set()
            if applied_lots:
                df_scope = df[df["lot"].astype(str).isin(list(applied_lots))]
                if applied_dep and "code_dep" in df_scope.columns:
                    df_scope = df_scope[df_scope["code_dep"].astype(str) == str(applied_dep)]
                if applied_communes:
                    df_scope = df_scope[df_scope["nom_commune"].isin(applied_communes)]
                if "code_insee" in df_scope.columns:
                    allowed_insee = set(df_scope["code_insee"].dropna().astype(str).str.upper().unique().tolist())

            # S’assurer que data_raw possède code_insee (ou l’inférer)
            df_raw = ensure_centers_have_insee(df_raw, communes_gdf)

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
                    options={'showCoverageOnHover': False, 'spiderLegPolylineOptions': {'opacity': 0, 'weight': 0}}
                ).add_to(m)

                # Conserver les points filtrés pour l'auto-zoom (fallback)
                lat_col_for_zoom, lon_col_for_zoom = lat_col, lon_col
                df_points = df_raw.dropna(subset=[lat_col, lon_col]).copy()
                if applied_lots and allowed_insee:
                    df_points = df_points[df_points["code_insee"].astype(str).str.upper().isin(allowed_insee)]
                filtered_points_for_zoom = df_points

                # Markers
                for _, r in df_points.iterrows():
                    try:
                        lat = float(r[lat_col]); lon = float(r[lon_col])
                    except Exception:
                        continue

                    # Champs tooltip/popup
                    nom_cc = str(r.get(name_col, "Centre")) if name_col else str(r.get("nom", "Centre"))
                    type_cc = str(r.get("type", r.get("Type", "-")))
                    adresse = str(r.get("adresse_complete", r.get("adresse", "-")))
                    cp = str(r.get("code_postal", r.get("cp", "")))
                    ville = str(r.get("ville", r.get("commune", "")))
                    proprietaire = str(r.get("gestionnaires", r.get("Propriétaire", "-")))
                    nb_boutiques = r.get("nb_boutiques", r.get("Nb_boutiques", r.get("magasins", r.get("shops", "-"))))
                    anciennete_val = r.get("nb_annees_ouverture", r.get("anciennete", r.get("age", "-")))
                    superficie = r.get("surface_gla", r.get("superficie", r.get("surface", "-")))

                    # Adresse 1..3 si dispo
                    if all(col in df_raw.columns for col in ["adresse1", "adresse2", "adresse3"]):
                        parts = [r.get(f"adresse{i}") for i in range(1, 4) if pd.notna(r.get(f"adresse{i}"))]
                        adresse_complete = ", ".join([str(x) for x in parts]) if parts else adresse
                    else:
                        adresse_complete = adresse

                    ville_app = r.get("nom_ville", ville)
                    code_postal_app = r.get("code_postal", cp)
                    type_cc_app = r.get("typologie_cc_long", type_cc)
                    proprietaire_app = r.get("gestionnaires", proprietaire)
                    superficie_app = r.get("surface_gla", superficie)

                    # Mise en forme surface
                    try:
                        superficie_app = f"{int(float(superficie_app)):,} m²".replace(",", " ")
                    except Exception:
                        pass

                    popup_html = f"""
                    <b>{nom_cc}</b><br>
                    Type : {type_cc_app}<br>
                    Adresse : {adresse_complete} - {code_postal_app} {ville_app}<br>
                    Propriétaire : {proprietaire_app}<br>
                    Boutiques : {nb_boutiques}<br>
                    Ancienneté : {anciennete_val} ans<br>
                    Superficie : {superficie_app}
                    """
                    folium.Marker(
                        location=[lat, lon],
                        tooltip=popup_html,
                        icon=folium.Icon(color="red", icon="shopping-cart", prefix="fa")
                    ).add_to(mc)
        except Exception as e:
            st.warning(f"Impossible d'afficher les centres (data_raw.csv): {e}")

    # Couche communes (polygones) si filtres → déjà par INSEE
    if not gdf.empty:
        def _format_date_fr(x):
            try:
                ts = pd.to_datetime(x, errors="coerce", dayfirst=False)
                if pd.isna(ts):
                    return x
                mois = ["jan", "fév", "mar", "avr", "mai", "juin", "juil", "aoû", "sep", "oct", "nov", "déc"]
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
            return {"fillColor": color, "color": color, "weight": 1, "fillOpacity": 0.45}

        folium.GeoJson(
            data=json.loads(gdf.to_json()),
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=aliases, sticky=True) if tooltip_fields else None,
            name=("Lots " + ", ".join(applied_lots)) if len(applied_lots) > 1 else (f"Lot {applied_lots[0]}" if applied_lots else "Communes"),
            style_function=_style_function,
        ).add_to(m)

    # === Auto-zoom sur la zone filtrée (département/communes) ===
    try:
        # on zoome si on a au moins un lot sélectionné (et potentiellement dep/communes)
        if applied_lots and (applied_dep or applied_communes):
            _fit_map_to_gdf_or_points(
                m,
                gdf=gdf,  # polygones déjà filtrés par dep/communes
                df_points=filtered_points_for_zoom,
                lat_col=lat_col_for_zoom,
                lon_col=lon_col_for_zoom
            )
    except Exception:
        # ne bloque pas l'affichage si échec du fit
        pass

    folium.LayerControl(collapsed=True).add_to(m)

    # Rendu HTML "statique"
    map_html = m.get_root().render()
    components.html(map_html, width=1800, height=800, scrolling=False)

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
