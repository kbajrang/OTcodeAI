import networkx as nx
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_code_graph(parsed_units: list[dict]) -> nx.DiGraph:
    """Create a directed graph from parsed code units."""
    graph = nx.DiGraph()
    for unit in parsed_units:
        node_id = unit.get("id")
        if not node_id:
            continue
        graph.add_node(node_id, **unit)
        for edge in unit.get("edges", []):
            graph.add_edge(edge["source"], edge["target"], type=edge.get("type"))
    logger.info("Graph nodes=%s edges=%s", graph.number_of_nodes(), graph.number_of_edges())
    return graph
