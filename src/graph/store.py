import os
import pickle
import networkx as nx
from src.config.settings import settings


def save_graph(graph: nx.DiGraph) -> None:
    os.makedirs(os.path.dirname(settings.graph_path), exist_ok=True)
    with open(settings.graph_path, "wb") as f:
        pickle.dump(graph, f)


def load_graph() -> nx.DiGraph:
    with open(settings.graph_path, "rb") as f:
        return pickle.load(f)
