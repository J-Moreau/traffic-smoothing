import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from geopandas import GeoSeries
from geopandas.geodataframe import GeoDataFrame
from shapely import LineString, MultiLineString, Point
from shapely.ops import linemerge, substring, unary_union


def order_road_segments(
    geodataframe: GeoDataFrame, linestring: LineString | MultiLineString, crs: str
) -> GeoDataFrame:
    """Assign a road_id to each road segment based on position along the motorway."""
    geodataframe = (
        geodataframe.assign(
            x_meters=lambda df: linestring.line_locate_point(
                GeoSeries(
                    df.geometry.to_crs(crs).apply(
                        lambda g: Point(g.coords[0])
                        if isinstance(g, LineString)
                        else Point(g.geoms[0].coords[0])
                        if isinstance(g, MultiLineString)
                        else g
                    ),
                    crs=crs,
                )
            ),
            x=lambda df: df.geometry.to_crs(crs).centroid.x,
            y=lambda df: df.geometry.to_crs(crs).centroid.y,
        )
        .sort_values("x_meters")
        .assign(road_id=lambda df: df.x_meters.rank(method="first").astype("int32"))
    )
    return geodataframe


def analyze_connected_components_direction(
    G: nx.DiGraph, connected_components: list[set]
) -> list[tuple]:
    """
    Analyze connected components to infer direction vectors.

    Returns:
    list of tuples (delta_lat, delta_lon)
    """
    directions = [
        (
            sum(
                G.nodes[v]["y"] - G.nodes[u]["y"]
                for u, v, *_ in G.subgraph(component).edges()
            ),
            sum(
                G.nodes[v]["x"] - G.nodes[u]["x"]
                for u, v, *_ in G.subgraph(component).edges()
            ),
        )
        for component in connected_components
    ]
    return directions


def extract_one_way_segment(
    nodes_gdf, edges_gdf, crs
) -> tuple[GeoDataFrame, GeoDataFrame, LineString | MultiLineString]:
    """Extract the motorway segment that is the most northbound from the graph. direction = North"""
    motorway_nodes_gdf = nodes_gdf[nodes_gdf.is_motorway]
    motorway_graph = ox.graph_from_gdfs(
        motorway_nodes_gdf, edges_gdf[~edges_gdf.is_ramp]
    )

    components = list(nx.weakly_connected_components(motorway_graph))
    directions = analyze_connected_components_direction(motorway_graph, components)
    component_idx = max(range(len(directions)), key=lambda i: directions[i][0])
    component = components[component_idx]
    one_way_nodes_gdf = motorway_nodes_gdf[motorway_nodes_gdf.index.isin(component)]
    one_way_edges_gdf = edges_gdf[
        edges_gdf.index.get_level_values(0).isin(component)
        & edges_gdf.index.get_level_values(1).isin(component)
    ]

    one_way_linestring = linemerge(unary_union(one_way_edges_gdf.geometry.to_crs(crs)))
    # project each motorway node onto the merged LineString and store distance + fraction along the line
    return one_way_nodes_gdf, one_way_edges_gdf, one_way_linestring


def categorize_road_segments(G) -> tuple[GeoDataFrame, GeoDataFrame]:
    """
    Categorize road segments into motorway, on-ramps, off-ramps, and regular roads.
    extended off-ramps and on-ramps are also identified by iteratively extending ramps
    returns nodes_gdf, edges_gdf
    with new columns:
    Nodes:
    - is_motorway
    - is_off_ramp
    - is_on_ramp
    - is_extended_off_ramp
    - is_extended_on_ramp
    Edges:
    - is_ramp
    - is_off_ramp
    - is_on_ramp
    - is_extended_off_ramp
    - is_extended_on_ramp
    - category: "on_ramp", "off_ramp", "road"
    """
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(
        G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True
    )
    cats = edges_gdf["highway"].fillna("unknown").astype(str)

    motorway_edges = edges_gdf[cats.isin(["motorway", "['motorway_link', 'motorway']"])]
    motorway_nodes = set(motorway_edges.index.get_level_values(0)).union(
        set(motorway_edges.index.get_level_values(1))
    )

    edges_gdf = edges_gdf.assign(
        is_ramp=lambda df: ~df.highway.astype(str).isin(
            ["motorway", "['motorway_link', 'motorway']"]
        ),
        start_node=lambda df: df.index.get_level_values(0),
        end_node=lambda df: df.index.get_level_values(1),
    )

    motorway_nodes = set(edges_gdf[~edges_gdf.is_ramp].start_node).union(
        edges_gdf[~edges_gdf.is_ramp].end_node
    )
    off_ramp_nodes = set(edges_gdf[edges_gdf.is_ramp].start_node).intersection(
        motorway_nodes
    )
    on_ramp_nodes = set(edges_gdf[edges_gdf.is_ramp].end_node).intersection(
        motorway_nodes
    )
    nodes_gdf = nodes_gdf.assign(
        is_motorway=lambda df: df.index.isin(motorway_nodes),
        is_off_ramp=lambda df: df.index.isin(off_ramp_nodes),
        is_on_ramp=lambda df: df.index.isin(on_ramp_nodes),
    )
    edges_gdf = edges_gdf.assign(
        is_motorway=cats.isin(["motorway", "['motorway_link', 'motorway']"]),
        is_off_ramp=lambda df: df.is_ramp
        & df.start_node.isin(off_ramp_nodes)
        & ~df.end_node.isin(motorway_nodes),
        is_on_ramp=lambda df: df.is_ramp
        & df.end_node.isin(on_ramp_nodes)
        & ~df.start_node.isin(motorway_nodes),
    )
    # iteratively extend on-ramps and off-ramps unless we reach some node we have already seen
    extended_off_ramp_nodes = set(edges_gdf[edges_gdf.is_off_ramp].end_node)
    extended_on_ramp_nodes = set(edges_gdf[edges_gdf.is_on_ramp].start_node)

    # hop ten times
    for _ in range(10):
        extended_off_ramp_nodes |= set(
            edges_gdf.loc[
                lambda df: df.start_node.isin(extended_off_ramp_nodes)
            ].end_node
        )
        extended_on_ramp_nodes |= set(
            edges_gdf.loc[
                lambda df: df.end_node.isin(extended_on_ramp_nodes)
            ].start_node
        )
        common_nodes = extended_off_ramp_nodes.intersection(extended_on_ramp_nodes)
        extended_off_ramp_nodes.difference_update(
            off_ramp_nodes | on_ramp_nodes | motorway_nodes | common_nodes
        )
        extended_on_ramp_nodes.difference_update(
            off_ramp_nodes | on_ramp_nodes | motorway_nodes | common_nodes
        )

    nodes_gdf = nodes_gdf.assign(
        is_extended_off_ramp=lambda df: df.index.isin(extended_off_ramp_nodes),
        is_extended_on_ramp=lambda df: df.index.isin(extended_on_ramp_nodes),
        category=lambda df: np.select(
            [
                df.is_on_ramp | df.is_extended_on_ramp,
                df.is_off_ramp | df.is_extended_off_ramp,
                df.is_motorway,
            ],
            ["on_ramp", "off_ramp", "motorway"],
            default="other",
        ),
    )

    edges_gdf = edges_gdf.assign(
        is_extended_off_ramp=lambda df: df.start_node.isin(extended_off_ramp_nodes),
        is_extended_on_ramp=lambda df: df.end_node.isin(extended_on_ramp_nodes),
        category=lambda df: np.select(
            [
                df.is_motorway,
                df.is_on_ramp | df.is_extended_on_ramp,
                df.is_off_ramp | df.is_extended_off_ramp,
            ],
            ["motorway", "on_ramp", "off_ramp"],
            default="other",
        ),
    )

    return nodes_gdf, edges_gdf


def duplicate_on_and_off_nodes(
    nodes_gdf: GeoDataFrame, edges_gdf: GeoDataFrame
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """
    Find nodes that are both on-ramps and off-ramps and separate them into two nodes.
    """

    common_nodes = set(
        edges_gdf.loc[lambda df: df.is_on_ramp]["start_node"]
    ).intersection(set(edges_gdf.loc[lambda df: df.is_off_ramp]["end_node"]))
    # Create copies to avoid inplace modification
    nodes_gdf_new = nodes_gdf.copy()
    edges_gdf_new = edges_gdf.copy()

    for node in common_nodes:
        new_node_id = -node
        node_data = nodes_gdf.loc[node].copy()
        nodes_gdf_new.loc[new_node_id] = node_data
        nodes_gdf_new.at[node, "category"] = "extended_off_ramp"
        nodes_gdf_new.at[node, "is_extended_off_ramp"] = True
        nodes_gdf_new.at[new_node_id, "category"] = "extended_on_ramp"
        nodes_gdf_new.at[new_node_id, "is_extended_on_ramp"] = True
        affected_on_ramp_edge = edges_gdf_new[
            (edges_gdf_new["start_node"] == node) & (edges_gdf_new["is_on_ramp"])
        ].index
        edges_gdf_new.loc[affected_on_ramp_edge, "start_node"] = new_node_id
        # Update the index for affected edge
        idx = affected_on_ramp_edge[0]
        edges_gdf_new.set_index(
            edges_gdf_new.index.map(
                lambda x: (new_node_id, x[1], x[2]) if x == idx else x
            ),
            inplace=True,
        )
    # set crs, the copy operation removes it
    nodes_gdf_new.geometry.set_crs(
        nodes_gdf.geometry.crs, allow_override=True, inplace=True
    )
    edges_gdf_new.geometry.set_crs(
        edges_gdf.geometry.crs, allow_override=True, inplace=True
    )
    return nodes_gdf_new, edges_gdf_new


def split_line(line: LineString, max_length_meters: float = 100) -> list[LineString]:
    """
    Split a LineString into segments of maximum length.

    Args:
        line: The LineString to split
        max_length_meters: Maximum length per segment in meters

    Returns:
        List of LineString segments
    """
    if line.length <= max_length_meters:
        return [line]

    n_slices = int(np.ceil(line.length / max_length_meters))
    return [
        LineString(
            substring(
                line,
                start_dist=i / n_slices,
                end_dist=(i + 1) / n_slices,
                normalized=True,
            )
        )
        for i in range(n_slices)
    ]


def split_long_edges(
    nodes_gdf: GeoDataFrame,
    edges_gdf: GeoDataFrame,
    max_length_meters: float = 500,
    target_crs: str = "EPSG:3310",
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """
    Split edges with LineString geometries that are too long into multiple edges.
    (generated with Claude)
    Args:
        nodes_gdf: GeoDataFrame of nodes
        edges_gdf: GeoDataFrame of edges with MultiIndex (u, v, key)
        max_length_meters: Maximum allowed edge length in meters
        crs: Coordinate reference system for length calculation

    Returns:
        Tuple of (updated_nodes_gdf, updated_edges_gdf) with long edges split
    """
    # Make copies to avoid modifying originals
    # Ensure CRS is preserved
    nodes_gdf_new = nodes_gdf.copy().to_crs(target_crs)
    edges_gdf_new = (
        edges_gdf.copy().to_crs(target_crs).assign(distance_from_start_of_split=np.nan)
    )

    # Find edges that are too long
    long_edges = edges_gdf_new[edges_gdf_new.geometry.length > max_length_meters]

    if len(long_edges) == 0:
        return nodes_gdf, edges_gdf

    # Keep track of the maximum node ID to create new nodes
    max_node_id = max(
        nodes_gdf_new.index.max(),
        edges_gdf_new.index.get_level_values(0).max(),
        edges_gdf_new.index.get_level_values(1).max(),
    )

    new_node_counter = max_node_id + 1
    # Process each long edge

    for edge_idx, edge_row in long_edges.iterrows():
        # Split the geometry using the split_line function
        line_segments = split_line(edge_row.geometry, max_length_meters)

        if len(line_segments) <= 1:
            continue  # Edge doesn't need splitting

        # Remove the original edge
        edges_gdf_new = edges_gdf_new.drop(edge_idx)
        start_node = nodes_gdf_new.loc[edge_idx[0]]
        new_edges, new_nodes = split_one_edge_into_multiple(
            line_segments, start_node, edge_row, edge_idx, new_node_counter
        )
        new_node_counter += len(new_nodes)
        for node_idx, new_node_data in new_nodes:
            nodes_gdf_new.loc[node_idx] = new_node_data
        for edge_idx, new_edge_data in new_edges:
            edges_gdf_new.loc[edge_idx] = new_edge_data

    # add crs to newly added geometries and convert back to original crs
    edges_gdf_new = edges_gdf_new.set_crs(target_crs, allow_override=True).to_crs(
        edges_gdf.crs
    )
    nodes_gdf_new = nodes_gdf_new.set_crs(target_crs, allow_override=True).to_crs(
        nodes_gdf.crs
    )

    return nodes_gdf_new, edges_gdf_new


def split_one_edge_into_multiple(
    line_segments: list[LineString],
    node_data: GeoSeries,
    edge_data: GeoSeries,
    edge_idx: tuple[int, int, int],
    new_node_counter: int,
):
    # Create intermediate nodes and edges
    prev_node, last_node, key = edge_idx

    cumul_length = 0.0
    new_edges = []
    new_nodes = []

    for i, segment in enumerate(line_segments):
        if i == len(line_segments) - 1:
            # Last segment connects to original end node
            next_node = last_node
        else:
            # Create intermediate node at the end of this segment
            next_node = new_node_counter + i

            # Add the intermediate node
            # Position it at the end of the current segment
            segment_end_point = segment.coords[-1]

            # Create node data by copying from start node and updating position
            new_node_data = node_data.copy()
            new_node_data["geometry"] = pd.Series(
                [segment_end_point], dtype="object"
            ).iloc[0]
            if hasattr(new_node_data, "geometry"):
                new_node_data["geometry"] = Point(segment_end_point)

            # Update coordinates
            new_node_data["x"] = segment_end_point[0]
            new_node_data["y"] = segment_end_point[1]

            # Add to nodes_gdf_new
            new_nodes.append((next_node, new_node_data))

        # Create the new edge

        new_edge_data = edge_data.copy()
        new_edge_data["geometry"] = segment
        new_edge_data["length"] = segment.length
        new_edge_data["start_node"] = prev_node
        new_edge_data["end_node"] = next_node
        new_edge_data["distance_from_start_of_split"] = cumul_length
        # Add the new edge with a new key to avoid conflicts
        new_edge_idx = (prev_node, next_node, key)
        new_edges.append((new_edge_idx, new_edge_data))
        cumul_length += segment.length
        prev_node = next_node
    return new_edges, new_nodes
