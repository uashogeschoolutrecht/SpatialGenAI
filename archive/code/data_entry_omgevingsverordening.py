import os
import glob
from lxml import etree
import psycopg2

# --- Namespaces for XML parsing ---
namespaces = {
    'ow-dc': 'http://www.geostandaarden.nl/imow/bestanden/deelbestand',
    'sl': 'http://www.geostandaarden.nl/bestanden-ow/standlevering-generiek',
    'regels': 'http://www.geostandaarden.nl/imow/regels',
    'rol': 'http://www.geostandaarden.nl/imow/regelsoplocatie',
    'rg': 'http://www.geostandaarden.nl/imow/regelingsgebied',
    'l': 'http://www.geostandaarden.nl/imow/locatie',
    'ga': 'http://www.geostandaarden.nl/imow/gebiedsaanwijzing',
    'k': 'http://www.geostandaarden.nl/imow/kaart',
    'kaart': 'http://www.geostandaarden.nl/imow/kaart',
    'p': 'http://www.geostandaarden.nl/imow/pons',
    'vt': 'http://www.geostandaarden.nl/imow/vrijetekst',
    's': 'http://www.geostandaarden.nl/imow/symbolisatie',                               
    'sym': 'http://www.geostandaarden.nl/imow/symbolisatie',
    'da': 'http://www.geostandaarden.nl/imow/datatypenalgemeen',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
    'xlink': 'http://www.w3.org/1999/xlink',
    'ow': 'http://www.geostandaarden.nl/imow/owobject',
    'op': 'http://www.geostandaarden.nl/imow/opobject',
    'xs': 'http://www.w3.org/2001/XMLSchema',
    'basisgeo' : 'http://www.geostandaarden.nl/basisgeometrie/1.0',
    'gml' : 'http://www.opengis.net/gml/3.2',
    'geo' : 'https://standaarden.overheid.nl/stop/imop/geo/'
}

# --- DB connection ---
conn = psycopg2.connect(
    dbname="spatial_planning"
)
cur = conn.cursor()

# --- Load regels ---
def load_regels(xml_path, regeltekst_map, tekst_map):
    print(f"Loading regels from {xml_path} ...")
    tree = etree.parse(xml_path)
    # Use xpath to find all Instructieregel elements at any depth
    instructieregels = tree.xpath('.//regels:Instructieregel', namespaces=namespaces)
    print(f"Found {len(instructieregels)} instructieregels.")
    for i, regel in enumerate(instructieregels, 1):
        print(f"Loop {i}/{len(instructieregels)}: extracting ids...")
        import sys; sys.stdout.flush()
        regel_id = regel.findtext("regels:identificatie", namespaces=namespaces)
        print(f"  regel_id: {regel_id}")
        regeltekst_ref = regel.find("regels:artikelOfLid/regels:RegeltekstRef", namespaces=namespaces)
        regeltekst_id = regeltekst_ref.get("{http://www.w3.org/1999/xlink}href") if regeltekst_ref is not None else None
        print(f"  regeltekst_id: {regeltekst_id}")
        regeltekst = regeltekst_map.get(regeltekst_id)
        print(f"  regeltekst: {regeltekst[:60] if regeltekst else None}")
        tekst = tekst_map.get(regeltekst_id)
        print(f"  tekst: {tekst[:60] if tekst else None}")
        import sys; sys.stdout.flush()
        try:
            print(f"  About to insert into DB...")
            import sys; sys.stdout.flush()
            cur.execute(
                "INSERT INTO regels (id, regeltekst, tekst) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (regel_id, regeltekst, tekst)
            )
            print(f"  Inserted.")
            import sys; sys.stdout.flush()
        except Exception as e:
            print(f"Error inserting instructieregel {i} (id={regel_id}): {e}")
            import sys; sys.stdout.flush()
        if i % 10 == 0 or i == len(instructieregels):
            print(f"Progress: {i}/{len(instructieregels)} instructieregels processed...")
            import sys; sys.stdout.flush()
    print("Finished loading instructieregels.")
    import sys; sys.stdout.flush()

# --- Build regeltekst and tekst maps ---
def build_regeltekst_map(xml_path):
    print(f"Building regeltekst map from {xml_path} ...")
    tree = etree.parse(xml_path)
    regeltekst_map = {el.findtext("regels:identificatie", namespaces=namespaces): el.findtext("regels:tekst", namespaces=namespaces) for el in tree.findall(".//regels:Regeltekst", namespaces=namespaces)}
    print(f"Regeltekst map contains {len(regeltekst_map)} entries.")
    return regeltekst_map

def build_tekst_map(xml_path):
    print(f"Building tekst map from {xml_path} ...")
    tree = etree.parse(xml_path)
    # Use the correct namespace and element for Tekst.xml
    tekst_ns = {'tekst': 'https://standaarden.overheid.nl/stop/imop/tekst/'}
    tekst_elements = tree.findall('.//tekst:Al', namespaces=tekst_ns)
    # Map by index, as there is no clear ID in Tekst.xml
    tekst_map = {str(i): el.text for i, el in enumerate(tekst_elements)}
    print(f"Tekst map contains {len(tekst_map)} entries.")
    return tekst_map

# --- Load gebieden ---
def load_gebieden(gebieden_xml, gml_base_path, gml_index):
    print(f"Loading gebieden from {gebieden_xml} ...")
    tree = etree.parse(gebieden_xml)
    gebieden = tree.findall(".//l:Gebied", namespaces=namespaces)
    print(f"Found {len(gebieden)} gebieden.")
    for i, gebied in enumerate(gebieden, 1):
        gebied_id = gebied.findtext("l:identificatie", namespaces=namespaces)
        naam = gebied.findtext("l:noemer", namespaces=namespaces)
        # Find GML reference
        geom_ref = gebied.find("l:geometrie/l:GeometrieRef", namespaces=namespaces)
        gml_id = geom_ref.get("{http://www.w3.org/1999/xlink}href") if geom_ref is not None else None
        gml, theme = find_gml_by_id(gml_id, gml_index, print_status=(i % 100 == 0 or i == len(gebieden)))
        cur.execute(
            "INSERT INTO gebieden (id, naam, gml, theme) VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
            (gebied_id, naam, gml, theme)
        )
        if i % 100 == 0 or i == len(gebieden):
            print(f"Progress: {i}/{len(gebieden)} gebieden inserted...")
    print("Finished loading gebieden.")

def find_gml_by_id(gml_id, gml_index, print_status=False):
    # Use the pre-built index for fast lookup
    if gml_id in gml_index:
        gml_file = gml_index[gml_id]['file_path']
        theme = gml_index[gml_id]['theme']
        with open(gml_file) as f:
            gml_content = f.read()
        if print_status:
            print(f"Found GML for id {gml_id} in file {gml_file} (theme: {theme})")
        return gml_content, theme
    if print_status:
        print(f"Warning: GML id {gml_id} not found in any GML file.")
    return None, None


# --- Utility: Build GML index ---
import multiprocessing
from functools import partial

def _process_gml_file(gml_file, namespaces):
    local_index = {}
    try:
        tree = etree.parse(gml_file)
        elements_with_id = tree.findall(".//basisgeo:id", namespaces=namespaces)
        theme = os.path.basename(os.path.dirname(gml_file))
        for el in elements_with_id:
            local_index[el.text] = {'file_path': gml_file, 'theme': theme}
    except Exception as e:
        print(f"Error parsing {gml_file}: {e}")
    return local_index

def build_gml_index(gml_base_path):
    print("Indexing all GML files for fast lookup (parallel) ...")
    gml_index = {}
    gml_files = glob.glob(os.path.join(gml_base_path, "**/*.gml"), recursive=True)
    print(f"Found {len(gml_files)} GML files to process.")
    total = len(gml_files)
    chunk = max(1, total // 20)  # Print every 5% or at least every file if few files
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = []
        for idx, local_index in enumerate(pool.imap(partial(_process_gml_file, namespaces=namespaces), gml_files), 1):
            results.append(local_index)
            if idx % chunk == 0 or idx == total:
                print(f"Indexed {idx}/{total} files ({(idx/total)*100:.1f}%)...")
    for local_index in results:
        gml_index.update(local_index)
    print(f"Indexed {len(gml_index)} GML IDs from {len(gml_files)} files.")
    return gml_index

# --- Main ---
if __name__ == "__main__":
    # Paths (update as needed)
    base = "data/spatial_genai_4_spatial_fase2/data_fase_2/akn_nl_act_pv26_2022_omgevingsverordening_download DSO/"
    print("Starting data entry script...")
    regeltekst_map = build_regeltekst_map(f"{base}OW-bestanden/regelteksten.xml")
    tekst_map = build_tekst_map(f"{base}Regeling/Tekst.xml")
    load_regels(f"{base}OW-bestanden/instructieregels.xml", regeltekst_map, tekst_map)
    gml_index = build_gml_index(base)
    load_gebieden(f"{base}OW-bestanden/gebieden.xml", base, gml_index)
    print("Committing changes to the database...")
    conn.commit()
    print("All done! Closing connection.")
    cur.close()
    conn.close()