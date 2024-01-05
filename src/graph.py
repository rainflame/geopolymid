from shapely.geometry import Point

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