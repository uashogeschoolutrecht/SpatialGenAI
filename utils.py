"""utils.py

Small utilities for generating, serialising and manipulating simple themed
polygon prototypes (for prototyping geospatial workflows and LLM-driven
geometry operations).

We only really use cut_polygon() from here in sketchpad_agents.ipynb, also
cut_polygon_with_buffer() could be useful later but needed some extra work.
The rest are utility functions used earlier in the process for archived code
to learn how polygons work and generating sample data for experimentation.

Conventions summary (short):
- WGS84 coordinates: (lon, lat)
- RD coordinates: EPSG:28992 (meters) for metric ops
- Use `.buffer(0)` defensively to clean geometries when needed
"""

import os
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union

# Geospatial
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from pyproj import CRS, Transformer
from lxml import etree as ET

# Coordinate reference systems (we are using Amersfoort / RD, but WGS84 is often used in other data sources)
CRS_WGS84 = CRS.from_epsg(4326)
CRS_RDNEW = CRS.from_epsg(28992)  # Amersfoort / RD New

TRANSFORMER_TO_RD = Transformer.from_crs(CRS_WGS84, CRS_RDNEW, always_xy=True)
TRANSFORMER_TO_WGS84 = Transformer.from_crs(CRS_RDNEW, CRS_WGS84, always_xy=True)

MAX_RADIUS_M = 1000.0

@dataclass
class ThemedPolygon:
    name: str
    theme: str
    polygon: Union[Polygon, MultiPolygon]
    properties: Dict[str, Any]

def wgs84_to_rd(lon: float, lat: float) -> Tuple[float, float]:
    return TRANSFORMER_TO_RD.transform(lon, lat)

def rd_to_wgs84(x: float, y: float) -> Tuple[float, float]:
    return TRANSFORMER_TO_WGS84.transform(x, y)

def generate_regular_polygon(center_xy: Tuple[float, float], radius_m: float, n: int, angle_deg: float = 0.0) -> Polygon:
    cx, cy = center_xy
    pts = []
    for i in range(n):
        theta = math.radians(angle_deg + (360.0 * i / n))
        x = cx + radius_m * math.cos(theta)
        y = cy + radius_m * math.sin(theta)
        pts.append((x, y))
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return Polygon(pts)

def cut_polygon(base_polygon: Union[Polygon, MultiPolygon], reference_polygon: Union[Polygon, MultiPolygon]) -> Union[Polygon, MultiPolygon]:
    """
    Subtract `reference_polygon` from `base_polygon` and return the result.
    Ensures invalid geometries are fixed using a small buffer(0) operation.
    """
    if not base_polygon.is_valid:
        base_polygon = base_polygon.buffer(0)
    if not reference_polygon.is_valid:
        reference_polygon = reference_polygon.buffer(0)
    return base_polygon.difference(reference_polygon)

def cut_with_buffer(base_polygon: Union[Polygon, MultiPolygon],
                    reference_polygon: Union[Polygon, MultiPolygon],
                    extra_meters: float,
                    resolution: int = 16,
                    cap_style: str = "round",
                    join_style: str = "round") -> Union[Polygon, MultiPolygon]:
    """
    Subtract `reference_polygon` buffered by `extra_meters` from `base_polygon`.

    Parameters
    - base_polygon: Polygon or MultiPolygon in RD coordinates (meters).
    - reference_polygon: Polygon or MultiPolygon in RD coordinates (meters).
    - extra_meters: distance in meters to expand the reference polygon before
      subtraction. Must be >= 0. If 0, this behaves like `cut_polygon`.
    - resolution: number of segments used to approximate a quarter circle when
      buffering (passed to Shapely's buffer).
    - cap_style: one of 'round', 'flat', 'square' to control buffer end caps.
    - join_style: one of 'round', 'mitre', 'bevel' to control joins between
      segments.

    Returns: resulting geometry (Polygon or MultiPolygon) in RD coordinates.

    Notes:
    - Input geometries are cleaned with buffer(0) if invalid before operations.
    - The buffer is performed in the same CRS as the inputs (expected RD meters).
    """
    if extra_meters < 0:
        raise ValueError("extra_meters must be >= 0")

    # Map sensible names to Shapely integers
    cap_map = {"round": 1, "flat": 2, "square": 3}
    join_map = {"round": 1, "mitre": 2, "bevel": 3}

    if cap_style not in cap_map:
        raise ValueError(f"cap_style must be one of {list(cap_map.keys())}")
    if join_style not in join_map:
        raise ValueError(f"join_style must be one of {list(join_map.keys())}")

    cap_val = cap_map[cap_style]
    join_val = join_map[join_style]

    if not base_polygon.is_valid:
        base_polygon = base_polygon.buffer(0)
    if not reference_polygon.is_valid:
        reference_polygon = reference_polygon.buffer(0)

    # If no extra buffer requested, use regular cut
    # Kind of superfluous?
    if extra_meters == 0:
        return cut_polygon(base_polygon, reference_polygon)

    buffered_ref = reference_polygon.buffer(extra_meters, resolution=resolution, cap_style=cap_val, join_style=join_val)

    result = base_polygon.difference(buffered_ref)

    if not result.is_valid:
        result = result.buffer(0)

    return result

def ensure_within_radius(poly: Polygon, center_xy: Tuple[float, float], max_radius_m: float) -> Polygon:
    """
    Ensure all exterior and interior coordinates of `poly` lie within
    `max_radius_m` from `center_xy`. If any coordinate lies further away,
    the entire polygon is scaled (about the provided center) so that the
    farthest point lies exactly at `max_radius_m`.

    Notes:
    - `poly` is expected to be a Shapely Polygon (not a MultiPolygon).
    - Interiors (holes) are scaled using the same factor so topology is
      preserved.
    - This function returns a new Polygon object; it does not modify the
      original.
    """
    cx, cy = center_xy
    max_dist = 0.0
    # consider exterior
    for x, y in poly.exterior.coords:
        d = math.hypot(x - cx, y - cy)
        max_dist = max(max_dist, d)
    # and interiors
    for interior in poly.interiors:
        for x, y in interior.coords:
            d = math.hypot(x - cx, y - cy)
            max_dist = max(max_dist, d)

    if max_dist <= max_radius_m or max_dist == 0:
        return poly

    scale = max_radius_m / max_dist
    def scale_ring(coords):
        scaled = []
        for x, y in coords:
            sx = cx + (x - cx) * scale
            sy = cy + (y - cy) * scale
            scaled.append((sx, sy))
        # ensure ring is closed
        if scaled[0] != scaled[-1]:
            scaled.append(scaled[0])
        return scaled

    exterior_scaled = scale_ring(list(poly.exterior.coords))
    interiors_scaled = [scale_ring(list(interior.coords)) for interior in poly.interiors]
    return Polygon(exterior_scaled, interiors=[ring for ring in interiors_scaled if len(ring) >= 4])

def polygon_or_multi_to_poslist_wgs84(geom: Union[Polygon, MultiPolygon]) -> List[List[Tuple[float, float]]]:
    """
    Convert a Polygon or MultiPolygon (in RD coordinates) to a list of rings
    expressed in WGS84 (lon, lat). The return value is a list of rings where
    the first ring is the outer ring for a polygon, followed by any interior
    rings. For a MultiPolygon, rings are appended for each polygon in sequence.

    This is a small helper retained for tooling and tests; it may be removed
    later if unused by callers.
    """
    rings: List[List[Tuple[float, float]]] = []
    def ring_to_lonlat(coords):
        return [rd_to_wgs84(x, y) for x, y in coords]
    if isinstance(geom, Polygon):
        rings.append(ring_to_lonlat(list(geom.exterior.coords)))
        for interior in geom.interiors:
            rings.append(ring_to_lonlat(list(interior.coords)))
    else:
        for poly in geom.geoms:
            rings.append(ring_to_lonlat(list(poly.exterior.coords)))
            for interior in poly.interiors:
                rings.append(ring_to_lonlat(list(interior.coords)))
    return rings

def polygon_to_gml(themed_poly: ThemedPolygon) -> bytes:
    """
    Serialize a ThemedPolygon into a small GML document.

    Implementation notes:
    - The top-level FeatureCollection and featureMember are placed in the
      GML namespace for better compatibility with GML consumers.
    - Geometries are written with coordinates in EPSG:4326 (lon lat).
    - ThemedPolygon and its metadata elements are intentionally not
      namespace-qualified (they are simple application-level tags).

    Returns: bytes (UTF-8 encoded XML with declaration)
    """
    ns_gml = "http://www.opengis.net/gml"
    ns_xsi = "http://www.w3.org/2001/XMLSchema-instance"
    NSMAP = {"gml": ns_gml, "xsi": ns_xsi}

    # FeatureCollection and featureMember MUST be in the GML namespace.
    feature = ET.Element(f"{{{ns_gml}}}FeatureCollection", nsmap=NSMAP)
    feature_member = ET.SubElement(feature, f"{{{ns_gml}}}featureMember")

    # Application-level feature element (no GML namespace) containing metadata
    feat = ET.SubElement(feature_member, "ThemedPolygon")
    ET.SubElement(feat, "name").text = themed_poly.name
    ET.SubElement(feat, "theme").text = themed_poly.theme

    props_el = ET.SubElement(feat, "properties")
    for k, v in themed_poly.properties.items():
        prop = ET.SubElement(props_el, k)
        prop.text = json.dumps(v) if isinstance(v, (dict, list)) else str(v)

    geom = themed_poly.polygon
    def write_polygon(parent_el, poly: Polygon, poly_id: str):
        poly_el = ET.SubElement(parent_el, f"{{{ns_gml}}}Polygon", attrib={f"{{{ns_gml}}}id": poly_id, "srsName": "urn:ogc:def:crs:EPSG::4326"})
        # exterior
        exterior = ET.SubElement(poly_el, f"{{{ns_gml}}}exterior")
        lr = ET.SubElement(exterior, f"{{{ns_gml}}}LinearRing")
        poslist = ET.SubElement(lr, f"{{{ns_gml}}}posList")
        ext_lonlat = [rd_to_wgs84(x, y) for x, y in poly.exterior.coords]
        poslist.text = " ".join([f"{lon:.8f} {lat:.8f}" for lon, lat in ext_lonlat])
        # holes
        for interior in poly.interiors:
            interior_el = ET.SubElement(poly_el, f"{{{ns_gml}}}interior")
            lr_i = ET.SubElement(interior_el, f"{{{ns_gml}}}LinearRing")
            poslist_i = ET.SubElement(lr_i, f"{{{ns_gml}}}posList")
            int_lonlat = [rd_to_wgs84(x, y) for x, y in interior.coords]
            poslist_i.text = " ".join([f"{lon:.8f} {lat:.8f}" for lon, lat in int_lonlat])

    if isinstance(geom, Polygon):
        write_polygon(feat, geom, themed_poly.name)
    else:
        ms = ET.SubElement(feat, f"{{{ns_gml}}}MultiSurface", attrib={"srsName": "urn:ogc:def:crs:EPSG::4326"})
        for idx, poly in enumerate(geom.geoms):
            sm = ET.SubElement(ms, f"{{{ns_gml}}}surfaceMember")
            write_polygon(sm, poly, f"{themed_poly.name}_{idx}")

    return ET.tostring(feature, pretty_print=True, xml_declaration=True, encoding="UTF-8")

def make_polygon(name: str, theme: str, polygon: Union[Polygon, MultiPolygon], properties: Optional[Dict[str, Any]] = None, out_dir: str = "test_polygons") -> str:
    """
    Write a ThemedPolygon to disk as a GML file and return the file path.

    The function will create `out_dir` if needed. The `name` is used for the
    filename; callers should provide a simple ASCII name without path
    separators. This function does not currently sanitize names aggressively.
    """
    if properties is None:
        properties = {}
    os.makedirs(out_dir, exist_ok=True)
    tp = ThemedPolygon(name=name, theme=theme, polygon=polygon, properties=properties)
    gml_bytes = polygon_to_gml(tp)
    filepath = os.path.join(out_dir, f"{name}.gml")
    with open(filepath, "wb") as f:
        f.write(gml_bytes)
    return filepath

def load_gml_polygon(filepath: str) -> Tuple[Union[Polygon, MultiPolygon], Dict[str, Any]]:
    """
    Load a GML file created by `polygon_to_gml` and return a Shapely geometry
    (Polygon or MultiPolygon) in RD coordinates and the properties dict with
    keys 'name' and 'theme'.
    """
    ns_gml = "http://www.opengis.net/gml"
    tree = ET.parse(filepath)
    root = tree.getroot()

    name_el = root.find(".//name")
    theme_el = root.find(".//theme")
    props = {
        "name": name_el.text if name_el is not None else "",
        "theme": theme_el.text if theme_el is not None else ""
    }

    ms = root.find(f".//{{{ns_gml}}}MultiSurface")
    polys: List[Polygon] = []
    # Parse GML Polygon posList (lon,lat), convert points to RD meters, build a Shapely Polygon and append to polys
    # Bit of a mess but keeps dependencies light without using GeoPandas for simple GML reading
    if ms is not None:
        for surface_member in ms.findall(f".//{{{ns_gml}}}surfaceMember"):
            poly_el = surface_member.find(f".//{{{ns_gml}}}Polygon")
            if poly_el is None:
                continue
            posList_el = poly_el.find(f".//{{{ns_gml}}}exterior/{{{ns_gml}}}LinearRing/{{{ns_gml}}}posList")
            if posList_el is None or not posList_el.text:
                continue
            nums = [float(v) for v in posList_el.text.strip().split()]
            if len(nums) % 2 != 0 or len(nums) < 8:
                raise ValueError(f"Invalid coordinate list in {filepath}")
            coords_lonlat = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
            coords_xy = [wgs84_to_rd(lon, lat) for lon, lat in coords_lonlat]
            interiors_xy = []
            for interior in poly_el.findall(f".//{{{ns_gml}}}interior"):
                pos_i = interior.find(f".//{{{ns_gml}}}LinearRing/{{{ns_gml}}}posList")
                if pos_i is not None and pos_i.text:
                    nums_i = [float(v) for v in pos_i.text.strip().split()]
                    if len(nums_i) % 2 != 0:
                        raise ValueError(f"Invalid interior coordinate list in {filepath}")
                    coords_lonlat_i = [(nums_i[i], nums_i[i+1]) for i in range(0, len(nums_i), 2)]
                    interiors_xy.append([wgs84_to_rd(lon, lat) for lon, lat in coords_lonlat_i])
            poly = Polygon(coords_xy, holes=[ring for ring in interiors_xy if len(ring) >= 4])
            polys.append(poly)
    else:
        # Single polygon
        poly_el = root.find(f".//{{{ns_gml}}}Polygon")
        if poly_el is None:
            raise ValueError(f"No gml:Polygon found in {filepath}")
        posList_el = poly_el.find(f".//{{{ns_gml}}}exterior/{{{ns_gml}}}LinearRing/{{{ns_gml}}}posList")
        if posList_el is None or not posList_el.text:
            raise ValueError(f"No coordinates found in {filepath}")
        nums = [float(v) for v in posList_el.text.strip().split()]
        if len(nums) % 2 != 0 or len(nums) < 8:
            raise ValueError(f"Invalid coordinate list in {filepath}")
        coords_lonlat = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
        coords_xy = [wgs84_to_rd(lon, lat) for lon, lat in coords_lonlat]
        interiors_xy = []
        for interior in poly_el.findall(f".//{{{ns_gml}}}interior"):
            pos_i = interior.find(f".//{{{ns_gml}}}LinearRing/{{{ns_gml}}}posList")
            if pos_i is not None and pos_i.text:
                nums_i = [float(v) for v in pos_i.text.strip().split()]
                if len(nums_i) % 2 != 0:
                    raise ValueError(f"Invalid interior coordinate list in {filepath}")
                coords_lonlat_i = [(nums_i[i], nums_i[i+1]) for i in range(0, len(nums_i), 2)]
                interiors_xy.append([wgs84_to_rd(lon, lat) for lon, lat in coords_lonlat_i])
        poly = Polygon(coords_xy, holes=[ring for ring in interiors_xy if len(ring) >= 4])
        polys = [poly]

    geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    return geom, props


def generate_sample_polygons(center_wgs84: Tuple[float, float] = (5.1214, 52.0907), max_radius_m: float = MAX_RADIUS_M) -> List[ThemedPolygon]:
    """
    Generate a set of sample themed polygons around a given WGS84 center point, which defaults to Utrecht in WGS84 format (lon, lat).
    """
    cx, cy = wgs84_to_rd(*center_wgs84)
    specs = [
        ("forest_zone", "forest", 120, 80, 220, 8, 10),
        ("factory_zone", "factory", -150, 60, 240, 9, 0),
        ("pond_zone", "pond", 60, -130, 180, 10, 20),
        ("park_zone", "park", -60, -100, 200, 8, 15),
        ("residential_zone", "residential", 200, -40, 230, 8, 5),
        ("commercial_zone", "commercial", -220, -30, 210, 9, -10),
    ]
    themed_polys: List[ThemedPolygon] = []
    for name, theme, dx, dy, radius, sides, rot in specs:
        center = (cx + dx, cy + dy)
        poly = generate_regular_polygon(center, radius, sides, rot)
        poly = ensure_within_radius(poly, (cx, cy), max_radius_m)
        props = {
            "description": f"{theme.capitalize()} area near Utrecht center",
            "priority": {"forest": 2, "pond": 3, "park": 2, "factory": 1, "residential": 2, "commercial": 1}.get(theme, 1),
        }
        themed_polys.append(ThemedPolygon(name=name, theme=theme, polygon=poly, properties=props))
    union_poly = unary_union([tp.polygon for tp in themed_polys]).buffer(80)
    union_poly = ensure_within_radius(union_poly, (cx, cy), max_radius_m)
    themed_polys.append(ThemedPolygon(
        name="city_boundary",
        theme="city",
        polygon=union_poly,
        properties={"description": "City boundary encompassing prototypes", "priority": 0}
    ))
    return themed_polys

def write_sample_gmls(polys: List[ThemedPolygon], out_dir: str = "test_polygons") -> List[str]:
    paths = []
    for tp in polys:
        paths.append(make_polygon(tp.name, tp.theme, tp.polygon, tp.properties, out_dir))
    return paths
