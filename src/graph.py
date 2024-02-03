import networkx as nx

from scipy.spatial import Voronoi
from shapely.geometry import Point, LineString


def dfs_sum_weights(node_weights, graph, node, visited):
    node_point = Point(node)
    visited.add(node)

    neighbors = graph.neighbors(node)
    unvisited_neighbors = [n for n in neighbors if n not in visited]

    total = 0
    for n in unvisited_neighbors:
        n_point = Point(n)
        distance = n_point.distance(node_point)
        total += dfs_sum_weights(node_weights, graph, n, visited) + distance

    node_weights[node] = total
    return total


def get_heaviest_path(graph, node_weights, node, visited):
    visited.add(node)

    neighbors = graph.neighbors(node)
    unvisited_neighbors = [n for n in neighbors if n not in visited]

    # find unvisited neighbor with max node weight
    max_weight = 0
    max_weight_node = None
    for n in unvisited_neighbors:
        if node_weights[n] >= max_weight:
            max_weight = node_weights[n]
            max_weight_node = n

    if max_weight_node is None:
        return [node]

    # recurse, and return given list and the element containing current node
    return get_heaviest_path(graph, node_weights, max_weight_node, visited) + [node]


def make_skeleton_graph_from_poly(polygon, graph):
    vor = Voronoi(polygon.exterior.coords)

    # convert the diagram into lines
    for vert in vor.ridge_vertices:
        if -1 not in vert:
            points = vor.vertices[vert]
            edge = LineString(points)
            # if the edge is inside the polygon, add it to the graph
            if polygon.contains(edge):
                graph.add_edge(
                    (points[0][0], points[0][1]), (points[1][0], points[1][1])
                )

    return graph


def find_graph_medial_axis(graph):
    center = nx.center(graph)[0]
    node_weights = {}
    dfs_sum_weights(node_weights, graph, center, set())

    neighbors = graph.neighbors(center)

    # get the two neighbors with the highest weights
    neighbor_weights = [(n, node_weights[n]) for n in neighbors]
    neighbor_weights.sort(key=lambda x: x[1], reverse=True)

    # get the two heaviest paths
    heaviest_paths = []
    for n, _ in neighbor_weights[:2]:
        heaviest_paths.append(get_heaviest_path(graph, node_weights, n, set([center])))

    return LineString(heaviest_paths[0] + [center] + heaviest_paths[1][::-1])
