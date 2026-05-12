import json
from typing import Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import polars as pl
import shapely
from shapely.ops import linemerge, unary_union
from tqdm import tqdm

from traffic_models.data.osm_network import (
    analyze_connected_components_direction,
    categorize_road_segments,
    duplicate_on_and_off_nodes,
    extract_one_way_segment,
    order_road_segments,
    split_long_edges,
)
from traffic_models.data.run_mobile_century_pull_graph import I880_OSMNX_EXTRACT
from traffic_models.experiment.probe_experiment import ProbeExperimentConfig
from traffic_models.sim import DiscretizationGrid, TimeGrid

CALIFORNIA_EPSG = "EPSG:3310"
MAX_LENGTH_SPLIT_ROADS = 500

def create_graph_edges_gdf(
    G: nx.MultiDiGraph,
    max_length_split_roads: float = MAX_LENGTH_SPLIT_ROADS,
    include_extended_ramps: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes_gdf, edges_gdf = categorize_road_segments(G)
    nodes_gdf, edges_gdf = duplicate_on_and_off_nodes(nodes_gdf, edges_gdf)
    nodes_gdf, edges_gdf = split_long_edges(
        nodes_gdf,
        edges_gdf,
        target_crs=CALIFORNIA_EPSG,
        max_length_meters=max_length_split_roads,
    )
    _, _, one_way_linestring = extract_one_way_segment(
        nodes_gdf, edges_gdf, CALIFORNIA_EPSG
    )

    def ban_extended_ramps(df: gpd.GeoDataFrame) -> pd.Series:
        if include_extended_ramps:
            return np.ones(len(df), dtype=bool)
        return ~(df.is_extended_on_ramp | df.is_extended_off_ramp)

    # clean edges and nodes to only keep those connected
    edges_gdf = (
        edges_gdf.loc[lambda df: df.category != "other"]
        .loc[ban_extended_ramps]
        .loc[
            lambda df: (
                df.index.get_level_values(0).isin(nodes_gdf.index)
                & df.index.get_level_values(1).isin(nodes_gdf.index)
            )
        ]
        .pipe(order_road_segments, one_way_linestring, CALIFORNIA_EPSG)
        .sort_values(by="x_meters")
    )
    nodes_gdf = (
        nodes_gdf.loc[
            nodes_gdf.index.isin(edges_gdf.index.get_level_values(0))
            | nodes_gdf.index.isin(edges_gdf.index.get_level_values(1))
        ]
        .pipe(order_road_segments, one_way_linestring, CALIFORNIA_EPSG)
        .sort_values(by="x_meters")
    )
    return edges_gdf, nodes_gdf


def one_way_I880_graph(
    direction: Literal["NORTHBOUND", "SOUTHBOUND"],
    osmnx_extract_path: str,
    max_length_split_roads: float = MAX_LENGTH_SPLIT_ROADS,
):
    G = ox.load_graphml(osmnx_extract_path)
    edges_gdf, nodes_gdf = create_graph_edges_gdf(G, max_length_split_roads)
    nodeG = ox.graph_from_gdfs(nodes_gdf, edges_gdf)

    # convert to line graph (where nodes are edges_gdf rows)
    nodeG = nx.DiGraph(nodeG)
    G = nx.line_graph(nodeG)
    G.add_nodes_from((node, nodeG.edges[node]) for node in G.nodes())

    # first sort by road_id (position along the highway)
    sign = 1 if direction == "NORTHBOUND" else -1
    road_id = np.array([data["road_id"] for u, data in G.nodes(data=True)])
    G = nx.relabel_nodes(
        G, dict(zip(G.nodes(), np.argsort(np.argsort(sign * road_id))))
    )
    # nodes are now (0, 1 , ..., N) but both directions are still in the graph

    # take only one direction
    connected_components = list(nx.weakly_connected_components(G))
    directions = analyze_connected_components_direction(G, connected_components)
    component_idx = max(range(len(directions)), key=lambda i: directions[i][0] * sign)
    one_way = connected_components[component_idx]

    G = G.subgraph(sorted(one_way)).copy()
    # nodes are now something like (23, 45, 67, ...) depending on which nodes were kept
    # then re-index nodes from 0 to N in the direction of travel
    G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G.nodes())))))

    # delete dangling ramps at the boundaries
    G = clean_graph_boundaries(G)
    G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G.nodes())))))

    if direction == "NORTHBOUND":
        # delete nodes on the lower part of the road that is not well measured
        G = G.subgraph(
            [n for n, data in G.nodes.data() if data["x_meters"] > 9_000]
        ).copy()
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G.nodes())))))
        POSTMILE_START = 15.242*1609.34 # convert miles to meters
        nx.set_node_attributes(
            G,
            {n: data["x_meters"] + POSTMILE_START for n, data in G.nodes(data=True)},
            "x_meters",
        )
    if direction == "SOUTHBOUND":
        POSTMILE_END = 27.535**1609.34 # convert miles to meters
        nx.set_node_attributes(
            G,
            {n: data["x_meters"] - POSTMILE_END for n, data in G.nodes(data=True)},
            "x_meters",
        )
    return G


def clean_graph_boundaries(G: nx.DiGraph):
    """
    Ensure that there is no on-ramps and off-ramps at the boundaries of the motorway sections of the graph.
    Remove the corresponding edges and nodes from the graph.
    """
    edges = np.array(list(G.edges()))
    cell_categories = np.array([data["category"] for u, data in G.nodes(data=True)])
    motorway_edges = edges[
        (cell_categories[edges[:, 0]] == "motorway")
        & (cell_categories[edges[:, 1]] == "motorway")
    ]
    motorway_nodes = np.nonzero(cell_categories == "motorway")[0]
    motorway_outputs = np.setdiff1d(motorway_nodes, motorway_edges[:, 0])
    motorway_inputs = np.setdiff1d(motorway_nodes, motorway_edges[:, 1])
    invalid_boundaries = np.concatenate(
        [list(nx.descendants(G, n)) for n in motorway_outputs]
        + [list(nx.ancestors(G, n)) for n in motorway_inputs]
    )
    G_copy = G.copy()
    G_copy.remove_nodes_from(invalid_boundaries)
    return G_copy


def mobile_century_ramp_indexes(grid: DiscretizationGrid, x_min_meters:float):
    G = one_way_I880_graph(
        direction="NORTHBOUND",
        osmnx_extract_path=I880_OSMNX_EXTRACT,
        max_length_split_roads=10000, # don't split roads, to keep the on off ramps as one node
    )
    nodes_table = pd.DataFrame(
            [{"node": n, **attrs} for n, attrs in G.nodes(data=True)]).assign(
        x_meters=lambda df: df["x_meters"]
    )
    on_ramps = nodes_table.loc[lambda df: df["category"]=="on_ramp"]["x_meters"].to_numpy()
    off_ramps = nodes_table.loc[lambda df: df["category"]=="off_ramp"]["x_meters"].to_numpy()
    # motorway_nodes = nodes_table.loc[lambda df: df["category"]=="motorway"]["x_meters"].to_numpy()
    on_ramps_index = ((on_ramps - x_min_meters)/grid.dx_meters).astype(int)
    off_ramps_index = ((off_ramps - x_min_meters)/grid.dx_meters).astype(int)
    # delete on and off ramps that are outside the grid
    on_ramps_index = on_ramps_index[(on_ramps_index >= 0) & (on_ramps_index < grid.n_cells-1)]
    off_ramps_index = off_ramps_index[(off_ramps_index >= 0) & (off_ramps_index < grid.n_cells-1)]
    return on_ramps_index,off_ramps_index
