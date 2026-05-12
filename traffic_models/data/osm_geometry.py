import time
import xml.etree.ElementTree as ET

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString


def get_osm_way_geometries(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract unique OSM way IDs from an OSMnx graph and retrieve their geometries using Overpy.

    Args:
        osmnx_extract_path (str): Path to the OSMnx GraphML file

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing OSM way IDs and their geometries
    """
    
    if edges_gdf["osmid"].dtype == object:
        # some edges have multiple osmids stored as list
        edges_gdf = edges_gdf.explode("osmid")
    unique_osmids = edges_gdf['osmid'].unique()

    # Filter out any None or NaN values
    unique_osmids = [osmid for osmid in unique_osmids if pd.notna(osmid)]

    # Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Process OSM IDs in batches of 500
    all_ways_data = []
    batch_size = 500
    
    for i in range(0, len(unique_osmids), batch_size):
        batch_osmids = unique_osmids[i:i + batch_size]
        
        # Build Overpass query for current batch
        osmid_list = ','.join(map(str, batch_osmids))
        query = f"""
        [out:xml];
        (
          way(id:{osmid_list});
        );
        out geom;
        """

        # Execute the query for this batch
        response = requests.post(overpass_url, data={'data': query})
        response.raise_for_status()
        time.sleep(1)
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Extract way data from XML
        for way in root.findall('way'):
            way_id_str = way.get('id')
            if not way_id_str:
                continue
            way_id = int(way_id_str)
            
            # Extract node coordinates
            coordinates = []
            for nd in way.findall('nd'):
                lat_str = nd.get('lat')
                lon_str = nd.get('lon')
                if lat_str and lon_str:
                    lat = float(lat_str)
                    lon = float(lon_str)
                    coordinates.append((lon, lat))
            
            # Extract tags
            tags = {}
            for tag in way.findall('tag'):
                key = tag.get('k')
                value = tag.get('v')
                if key and value:
                    tags[key] = value
            
            if coordinates:  # Only add ways with valid coordinates
                all_ways_data.append({
                    'osmid': way_id,
                    'geometry': LineString(coordinates),
                    'tags': tags
                })

    way_data = all_ways_data

    gdf = gpd.GeoDataFrame(way_data, crs='EPSG:4326')
    assert len(gdf) == len(unique_osmids), "Mismatch in number of OSM IDs and retrieved geometries."
    return gdf