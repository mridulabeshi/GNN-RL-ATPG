from pathlib import Path
import json
import random

import torch
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CIRCUIT = "picorv32"

FEATURE_FILE = Path(
    f"data/features/{CIRCUIT}_features.pt"
)

FAULT_FILE = Path(
    f"data/fault_list/{CIRCUIT}_faults.json"
)

OUTPUT_DIR = Path("data/evaluation_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of nodes shown in the visual graph.
# Keep this around 100-300 for readability.
NUM_VISUAL_NODES = 150

# Reproducible visualization
SEED = 42

random.seed(SEED)


# ============================================================
# Load circuit
# ============================================================

print("=" * 60)
print("Loading actual PicoRV32 circuit")
print("=" * 60)

data = torch.load(
    FEATURE_FILE,
    map_location="cpu",
    weights_only=False
)

x = data.x
edge_index = data.edge_index

num_nodes = x.shape[0]
num_edges = edge_index.shape[1]

print("Circuit nodes :", num_nodes)
print("Circuit edges :", num_edges)
print("Features      :", x.shape[1])


# ============================================================
# Build actual circuit graph
# ============================================================

G = nx.DiGraph()

for node_id in range(num_nodes):
    G.add_node(node_id)

for i in range(num_edges):
    src = int(edge_index[0, i])
    dst = int(edge_index[1, i])

    G.add_edge(src, dst)


# ============================================================
# Select a connected region
# ============================================================

def connected_region(graph, start_node, max_nodes):

    visited = set()
    queue = [start_node]

    while queue and len(visited) < max_nodes:

        node = queue.pop(0)

        if node in visited:
            continue

        visited.add(node)

        neighbours = set(graph.successors(node))
        neighbours.update(graph.predecessors(node))

        for n in neighbours:
            if n not in visited:
                queue.append(n)

    return visited


start_node = 0

selected_nodes = connected_region(
    G,
    start_node,
    NUM_VISUAL_NODES
)

H = G.subgraph(selected_nodes).copy()

print()
print("Visualization region")
print("--------------------")
print("Nodes:", H.number_of_nodes())
print("Edges:", H.number_of_edges())


# ============================================================
# Load actual faults
# ============================================================

with open(FAULT_FILE, "r") as f:
    faults = json.load(f)

print()
print("Actual faults:", len(faults))


# ============================================================
# Extract fault nodes inside visualization region
# ============================================================

fault_nodes = set()

for fault in faults:

    # Support common JSON field names
    node_id = fault.get("gate_id")

    if node_id is None:
        node_id = fault.get("node_id")

    if node_id is None:
        continue

    node_id = int(node_id)

    if node_id in selected_nodes:
        fault_nodes.add(node_id)


print(
    "Fault sites in displayed region:",
    len(fault_nodes)
)


# ============================================================
# Identify PI / PO nodes
# ============================================================

primary_inputs = [
    n for n in H.nodes()
    if H.in_degree(n) == 0
]

primary_outputs = [
    n for n in H.nodes()
    if H.out_degree(n) == 0
]


# ============================================================
# Layout
# ============================================================

print()
print("Calculating graph layout...")

pos = nx.spring_layout(
    H,
    seed=SEED,
    k=1.5,
    iterations=100
)


# ============================================================
# Node categories
# ============================================================

normal_nodes = []

for n in H.nodes():

    if n in fault_nodes:
        continue

    if n in primary_inputs:
        continue

    if n in primary_outputs:
        continue

    normal_nodes.append(n)


# ============================================================
# Figure
# ============================================================

plt.figure(
    figsize=(20, 14)
)


# ============================================================
# Draw wires
# ============================================================

nx.draw_networkx_edges(
    H,
    pos,
    edge_color="gray",
    width=0.8,
    alpha=0.35,
    arrows=True,
    arrowsize=8
)


# ============================================================
# Draw normal gates
# ============================================================

nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=normal_nodes,
    node_color="lightblue",
    node_size=180,
    node_shape="o",
    edgecolors="black",
    linewidths=0.5
)


# ============================================================
# Draw primary inputs
# ============================================================

nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=primary_inputs,
    node_color="gold",
    node_size=300,
    node_shape="s",
    edgecolors="black",
    linewidths=1.0
)


# ============================================================
# Draw primary outputs
# ============================================================

nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=primary_outputs,
    node_color="limegreen",
    node_size=300,
    node_shape="D",
    edgecolors="black",
    linewidths=1.0
)


# ============================================================
# Draw fault sites
# ============================================================

if fault_nodes:

    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=list(fault_nodes),
        node_color="red",
        node_size=450,
        node_shape="X",
        edgecolors="black",
        linewidths=1.2
    )


# ============================================================
# Labels
# ============================================================

# Only label fault / PI / PO nodes.
important_nodes = set(fault_nodes)
important_nodes.update(primary_inputs)
important_nodes.update(primary_outputs)

labels = {
    n: str(n)
    for n in important_nodes
}


nx.draw_networkx_labels(
    H,
    pos,
    labels=labels,
    font_size=7,
    font_weight="bold"
)


# ============================================================
# Legend
# ============================================================

import matplotlib.lines as mlines

normal_legend = mlines.Line2D(
    [],
    [],
    marker="o",
    linestyle="None",
    markersize=8,
    markerfacecolor="lightblue",
    markeredgecolor="black",
    label="Gate"
)

pi_legend = mlines.Line2D(
    [],
    [],
    marker="s",
    linestyle="None",
    markersize=9,
    markerfacecolor="gold",
    markeredgecolor="black",
    label="Primary Input"
)

po_legend = mlines.Line2D(
    [],
    [],
    marker="D",
    linestyle="None",
    markersize=9,
    markerfacecolor="limegreen",
    markeredgecolor="black",
    label="Primary Output"
)

fault_legend = mlines.Line2D(
    [],
    [],
    marker="X",
    linestyle="None",
   