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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "picorv32_fault_nodes.png"

# Number of circuit nodes displayed
DISPLAY_NODES = 250

# Number of fault nodes highlighted
DISPLAY_FAULT_NODES = 25

SEED = 42

random.seed(SEED)


# ============================================================
# Load actual circuit graph
# ============================================================

print("=" * 60)
print("PicoRV32 Fault Node Visualization")
print("=" * 60)

data = torch.load(
    FEATURE_FILE,
    map_location="cpu",
    weights_only=False
)

edge_index = data.edge_index

num_nodes = data.x.shape[0]
num_edges = edge_index.shape[1]

print(f"Actual circuit nodes : {num_nodes}")
print(f"Actual circuit edges : {num_edges}")


# ============================================================
# Construct NetworkX graph from actual edge_index
# ============================================================

G = nx.DiGraph()

for node_id in range(num_nodes):
    G.add_node(node_id)

for i in range(num_edges):

    source = int(edge_index[0, i])
    target = int(edge_index[1, i])

    G.add_edge(
        source,
        target
    )


# ============================================================
# Find connected region
# ============================================================

def get_connected_region(graph, start, limit):

    visited = set()

    queue = [start]

    while queue and len(visited) < limit:

        current = queue.pop(0)

        if current in visited:
            continue

        visited.add(current)

        neighbours = set(
            graph.successors(current)
        )

        neighbours.update(
            graph.predecessors(current)
        )

        for node in neighbours:

            if node not in visited:

                queue.append(node)

    return visited


# Start from node 0
region_nodes = get_connected_region(
    G,
    start=0,
    limit=DISPLAY_NODES
)

H = G.subgraph(region_nodes).copy()

print(f"Displayed circuit nodes: {H.number_of_nodes()}")
print(f"Displayed circuit edges: {H.number_of_edges()}")


# ============================================================
# Load actual fault list
# ============================================================

with open(
    FAULT_FILE,
    "r"
) as f:

    faults = json.load(f)


print(f"Total actual faults: {len(faults)}")


# ============================================================
# Find actual fault sites in displayed region
# ============================================================

fault_information = {}

for fault in faults:

    gate_id = fault.get("gate_id")

    if gate_id is None:
        gate_id = fault.get("node_id")

    if gate_id is None:
        continue

    gate_id = int(gate_id)

    if gate_id not in region_nodes:
        continue

    fault_type = str(
        fault.get(
            "fault_type",
            fault.get(
                "type",
                "UNKNOWN"
            )
        )
    )

    if gate_id not in fault_information:

        fault_information[gate_id] = []

    fault_information[gate_id].append(
        fault_type
    )


# ============================================================
# Select fault nodes
# ============================================================

available_fault_nodes = list(
    fault_information.keys()
)

random.shuffle(
    available_fault_nodes
)

selected_fault_nodes = available_fault_nodes[
    :DISPLAY_FAULT_NODES
]

selected_fault_nodes = sorted(
    selected_fault_nodes
)

print(
    f"Fault nodes in displayed region: "
    f"{len(available_fault_nodes)}"
)

print(
    f"Highlighted fault nodes: "
    f"{len(selected_fault_nodes)}"
)


# ============================================================
# Identify primary inputs / outputs
# ============================================================

primary_inputs = [
    node
    for node in H.nodes()
    if H.in_degree(node) == 0
]

primary_outputs = [
    node
    for node in H.nodes()
    if H.out_degree(node) == 0
]


# ============================================================
# Layout
# ============================================================

print("Generating layout...")

pos = nx.spring_layout(
    H,
    seed=SEED,
    k=1.2,
    iterations=100
)


# ============================================================
# Create figure
# ============================================================

plt.figure(
    figsize=(20, 15)
)


# ============================================================
# Draw actual circuit wires
# ============================================================

nx.draw_networkx_edges(
    H,
    pos,
    edge_color="gray",
    width=0.7,
    alpha=0.35,
    arrows=True,
    arrowsize=7
)


# ============================================================
# Normal circuit nodes
# ============================================================

normal_nodes = [
    node
    for node in H.nodes()
    if node not in selected_fault_nodes
    and node not in primary_inputs
    and node not in primary_outputs
]


nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=normal_nodes,
    node_color="lightblue",
    node_size=130,
    node_shape="o",
    edgecolors="black",
    linewidths=0.4,
    alpha=0.85
)


# ============================================================
# Primary inputs
# ============================================================

nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=primary_inputs,
    node_color="gold",
    node_size=250,
    node_shape="s",
    edgecolors="black",
    linewidths=0.8
)


# ============================================================
# Primary outputs
# ============================================================

nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=primary_outputs,
    node_color="limegreen",
    node_size=280,
    node_shape="D",
    edgecolors="black",
    linewidths=0.8
)


# ============================================================
# Fault nodes
# ============================================================

if selected_fault_nodes:

    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=selected_fault_nodes,
        node_color="red",
        node_size=500,
        node_shape="o",
        edgecolors="black",
        linewidths=1.5
    )


# ============================================================
# Fault node labels
# ============================================================

fault_labels = {}

for node in selected_fault_nodes:

    fault_types = fault_information[node]

    label = (
        f"{node}\n"
        + "/".join(fault_types)
    )

    fault_labels[node] = label


nx.draw_networkx_labels(
    H,
    pos,
    labels=fault_labels,
    font_size=7,
    font_weight="bold"
)


# ============================================================
# Title
# ============================================================

plt.title(
    "PicoRV32 Gate-Level Circuit with Stuck-at Fault Sites",
    fontsize=18,
    fontweight="bold",
    pad=20
)


# ============================================================
# Legend
# ============================================================

from matplotlib.lines import Line2D

legend_elements = [

    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="lightblue",
        markeredgecolor="black",
        markersize=9,
        label="Circuit node"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=11,
        label="Stuck-at fault site"
    ),

    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="gold",
        markeredgecolor="black",
        markersize=9,
        label="Primary input"
    ),

    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor="limegreen",
        markeredgecolor="black",
        markersize=9,
        label="Primary output"
    ),

    Line2D(
        [0],
        [0],
        color="gray",
        linewidth=1,
        label="Circuit connection"
    )
]


plt.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=10
)


# ============================================================
# Additional information
# ============================================================

plt.text(
    0.01,
    0.01,
    (
        f"Full circuit: {num_nodes} nodes | "
        f"{num_edges} edges | "
        f"{len(faults)} stuck-at faults\n"
        f"Displayed: {H.number_of_nodes()} nodes | "
        f"{H.number_of_edges()} edges | "
        f"{len(selected_fault_nodes)} highlighted fault nodes"
    ),
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)


# ============================================================
# Save high-resolution PNG
# ============================================================

plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("=" * 60)
print("Visualization complete")
print("=" * 60)

print(
    f"Saved: {OUTPUT_FILE}"
)

print()
print("Highlighted fault nodes:")

for node in selected_fault_nodes:

    print(
        f"  Node {node}: "
        f"{', '.join(fault_information[node])}"
    )