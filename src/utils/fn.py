from dataclasses import dataclass
from typing import List, Tuple


def spans2tree(spans: List[Tuple[int, int]]):
    # [inclusive beginning, exclusive end)

    index = list(range(len(spans)))
    index.sort(key=lambda x: (spans[x][0], -spans[x][1]))

    sorted_spans = []
    for i in index:
        span = spans[i]
        sorted_spans.append((span[0], span[1] - 1))

    parent = [i - 1 for i in range(len(sorted_spans))]
    endpoint = [-1 for _ in range(len(sorted_spans))]

    prev_left = sorted_spans[0][0]
    prev_left_i = 0

    for i, (s, e) in enumerate(sorted_spans[1:], start=1):

        if s == prev_left:
            continue

        endpoint[prev_left_i] = i - 1

        possible_parent_start = prev_left_i
        possible_parent_end = i - 1
        while sorted_spans[possible_parent_start][1] < e:
            possible_parent_start = parent[possible_parent_start]
            possible_parent_end = endpoint[possible_parent_start]

        possible_parent_end += 1
        while (possible_parent_end - possible_parent_start) > 1:
            cursor = (possible_parent_start + possible_parent_end) // 2
            v = sorted_spans[cursor][1]
            if v < e:
                possible_parent_end = cursor
            elif v == e:
                possible_parent_start = cursor
                break
            else:
                possible_parent_start = cursor

        prev_left = s
        prev_left_i = i
        parent[i] = possible_parent_start
    inv_index = list(range(len(spans)))
    inv_index.sort(key=lambda i: index[i])
    reordered_parent = [index[parent[i]] if parent[i] != -1 else -1 for i in inv_index]
    return reordered_parent


@dataclass
class Tree:
    left: "Tree" = None
    right: "Tree" = None
    id: int = None


def span_parents_to_traversable_tree(spans, parents):
    # assume spans are ordered by (width, start point)
    children_array = [[] for _ in range(len(parents))]
    for i, p in enumerate(parents):
        if p != -1:  # -1 is root
            children_array[p].append(i)
    tree_nodes = []
    for i, children in enumerate(children_array):
        assert len(children) <= 2
        left, right = None, None
        for c in children:
            if spans[c][0] == spans[i][0]:
                left = tree_nodes[c]
            elif spans[c][1] == spans[i][1]:
                right = tree_nodes[c]
        tree_nodes.append(Tree(left, right, i))
    return tree_nodes[-1]
