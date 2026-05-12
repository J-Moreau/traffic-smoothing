import osmnx as ox

from traffic_models.data.osm_geometry import get_osm_way_geometries

I880_OSMNX_EXTRACT = "data/mobilecentury/I880_osmnx_extract.graphml"
I880_OSM_POLYLINES = "data/mobilecentury/I880_osm_polylines.geojson"

def main():
    G = ox.graph.graph_from_bbox(
            # bbox,
            (-122.103, 37.51, -121.97, 37.66),
            # (37.50,-122.1,37.657,-121.0),
            network_type="drive",
            custom_filter=[
                '["highway"~"motorway"]["ref"~"I 880"]["hov"!~"designated"]',
                '(1032904190)', # split on ramp and off ramp in two around Fremont Blvd
                '["highway"~"motorway_link"]["ref"!~"CA 92"]',
            ],
            simplify=True,
        )
    ox.io.save_graphml(G, I880_OSMNX_EXTRACT)

    # pull a geodataframe with way geometries
    ways_gdf = get_osm_way_geometries(ox.graph_to_gdfs(G, nodes=False, edges=True))
    ways_gdf.to_file(I880_OSM_POLYLINES, driver="GeoJSON")

if __name__ == '__main__':
    main()