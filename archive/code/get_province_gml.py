import geopandas as gpd
import pandas as pd
import requests
import io

##################################################################################################
"""
Below is a script to download a gml file containing the boundaries of the Province of Utrecht.
This was not used for the final version of the project, but is provided here for reference.
"""
##################################################################################################

def get_wfs_features(wfs_url, type_name, output_format='application/json'):
    """
    Downloads features from a WFS layer using the server's specified parameters.
    """
    params = {
        'service': 'WFS',
        'version': '2.0.0',
        'request': 'GetFeature',
        'typeName': type_name,
        'outputFormat': output_format,
    }
    
    try:
        print(f"  - Sending WFS 2.0.0 request for layer: {type_name}")
        r = requests.get(wfs_url, params=params, verify=True, timeout=60)
        r.raise_for_status()

        if not r.content:
            print("  - Error: Received an empty response from the server.")
            return None

        gdf = gpd.read_file(io.BytesIO(r.content))
        print(f"  - Success: Read {len(gdf)} features.")
        return gdf
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP Error making WFS request: {e}")
        print(f"Full response text that caused the error: {r.text}")
    except Exception as e:
        print(f"An error occurred while processing the data with GeoPandas: {e}")
    return None

def main():
    """
    Main function to download, process, and save administrative boundaries.
    """
    WFS_URL = 'https://service.pdok.nl/cbs/gebiedsindelingen/2023/wfs/v1_0'
    
    PROVINCE_LAYER = 'gebiedsindelingen:provincie_gegeneraliseerd'
    MUNICIPALITY_LAYER = 'gebiedsindelingen:gemeente_gegeneraliseerd'
    
    PROVINCE_NAME = 'Utrecht'
    TARGET_CRS = 'EPSG:28992'
    OUTPUT_FILENAME = 'utrecht_boundaries.gml'

    print("--- Starting Geospatial Data Processing (Final Version) ---")

    print(f"\nDownloading provincial boundaries...")
    provinces_gdf = get_wfs_features(WFS_URL, PROVINCE_LAYER)
    if provinces_gdf is None or provinces_gdf.empty: return

    print(f"\nDownloading municipal boundaries...")
    municipalities_gdf = get_wfs_features(WFS_URL, MUNICIPALITY_LAYER)
    if municipalities_gdf is None or municipalities_gdf.empty: return
        
    print(f"\nSelecting the province of '{PROVINCE_NAME}'...")
    utrecht_province = provinces_gdf[provinces_gdf['statnaam'] == PROVINCE_NAME].copy()
    if utrecht_province.empty:
        print(f"Province '{PROVINCE_NAME}' not found.")
        return
    print(f"Found feature for '{PROVINCE_NAME}'.")

    print("Performing spatial join to find municipalities within Utrecht...")
    utrecht_municipalities_joined = gpd.sjoin(municipalities_gdf, utrecht_province, how="inner", predicate="within")
    print(f"Found {len(utrecht_municipalities_joined)} municipalities within Utrecht.")
    
    # Create the final municipality dataframe by selecting and renaming the correct columns
    utrecht_municipalities = gpd.GeoDataFrame({
        'naam': utrecht_municipalities_joined['statnaam_left'],
        'type': 'Gemeente',
        'statcode': utrecht_municipalities_joined['statcode_left'],
        'geometry': utrecht_municipalities_joined.geometry
    }, crs=municipalities_gdf.crs)

    utrecht_province['type'] = 'Provincie'
    utrecht_province.rename(columns={'statnaam': 'naam'}, inplace=True)
    final_columns = ['naam', 'type', 'statcode', 'geometry']
    combined_gdf = pd.concat(
        [utrecht_province[final_columns], utrecht_municipalities[final_columns]], 
        ignore_index=True
    )
    print("\nCombined province and municipalities into a single dataset.")
    print("Final data preview:")
    print(combined_gdf.head())

    print(f"\nReprojecting geometries to {TARGET_CRS} (Amersfoort / RD New)...")
    combined_gdf_rd = combined_gdf.to_crs(TARGET_CRS)

    print("Rounding coordinates to 1 meter precision...")
    combined_gdf_rd.geometry = combined_gdf_rd.geometry.set_precision(1.0)

    print(f"\nWriting the final data to '{OUTPUT_FILENAME}'...")
    try:
        combined_gdf_rd.to_file(OUTPUT_FILENAME, driver='GML')
        print(f"Successfully created '{OUTPUT_FILENAME}'.")
    except Exception as e:
        print(f"Error writing to GML file: {e}")

    print("\n--- Geospatial Data Processing Complete ---")

if __name__ == '__main__':
    main()