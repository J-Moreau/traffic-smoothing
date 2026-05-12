from traffic_models.data.mobile_century_mapped import one_way_I880_graph
from traffic_models.data.run_mobile_century_pull_graph import I880_OSMNX_EXTRACT


def test_one_way_I880_graph():
    G = one_way_I880_graph(direction="NORTHBOUND", osmnx_extract_path=I880_OSMNX_EXTRACT, max_length_split_roads=500)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0