from pathlib import Path
import json
import random

import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

DISPLAY_NODES = 250

# Number of nodes highlighted in each fault plot
MAX_FAULT_NODES = 25

SEED = 42

random.seed(SEED)


# ============================================================
# Load actual circuit
# ============================================================

print("=" * 70)
print("PicoRV32 Fault-Type Visualization")
print("=" * 70)

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
# Connected region
# ============================================================

def get_connected_region(graph, start_node, limit):

    visited = set()
    queue = [start_node]

    while queue and len(visited) < limit:

        node = queue.pop(0)

        if node in visited:
            continue

        visited.add(node)

        neighbours = set(
            graph.successors(node)
        )

        neighbours.update(
            graph.predecessors(node)
        )

        for neighbour in neighbours:

            if neighbour not in visited:

                queue.append(neighbour)

    return visited


region_nodes = get_connected_region(
    G,
    start_node=0,
    limit=DISPLAY_NODES
)

H = G.subgraph(region_nodes).copy()

print(f"Displayed nodes      : {H.number_of_nodes()}")
print(f"Displayed edges      : {H.number_of_edges()}")


# ============================================================
# Load fault list
# ============================================================

with open(
    FAULT_FILE,
    "r"
) as f:

    faults = json.load(f)

print(f"Total faults         : {len(faults)}")


# ============================================================
# Separate SA0 / SA1 faults
# ============================================================

sa0_nodes = set()
sa1_nodes = set()

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
            fault.get("type", "")
        )
    ).upper()

    if "SA0" in fault_type:

        sa0_nodes.add(gate_id)

    elif "SA1" in fault_type:

        sa1_nodes.add(gate_id)


print()
print("Fault sites in displayed region")
print("--------------------------------")
print("SA0 nodes:", len(sa0_nodes))
print("SA1 nodes:", len(sa1_nodes))


# ============================================================
# Layout
# ============================================================

print()
print("Calculating circuit layout...")

pos = nx.spring_layout(
    H,
    seed=SEED,
    k=1.2,
    iterations=100
)


# ============================================================
# PI / PO
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
# Generic plot function
# ============================================================

def plot_fault_type(
    fault_type,
    fault_nodes,
    output_file,
    title
):

    # --------------------------------------------------------
    # Limit number of displayed fault nodes
    # --------------------------------------------------------

    fault_nodes = list(fault_nodes)

    random.seed(SEED)

    if len(fault_nodes) > MAX_FAULT_NODES:

        fault_nodes = random.sample(
            fault_nodes,
            MAX_FAULT_NODES
        )

    fault_nodes = sorted(fault_nodes)

    # --------------------------------------------------------
    # Normal nodes
    # --------------------------------------------------------

    normal_nodes = [
        node
        for node in H.nodes()
        if node not in fault_nodes
        and node not in primary_inputs
        and node not in primary_outputs
    ]

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(20, 15)
    )

    # --------------------------------------------------------
    # Actual circuit wires
    # --------------------------------------------------------

    nx.draw_networkx_edges(
        H,
        pos,
        edge_color="gray",
        width=0.7,
        alpha=0.35,
        arrows=True,
        arrowsize=7
    )

    # --------------------------------------------------------
    # Normal circuit nodes
    # --------------------------------------------------------

    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=normal_nodes,
        node_color="lightblue",
        node_size=130,
        node_shape="o",
        edgecolors="black",
        linewidths=0.4
    )

    # --------------------------------------------------------
    # Primary inputs
    # --------------------------------------------------------

    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=primary_inputs,
        node_color="gold",
        node_size=260,
        node_shape="s",
        edgecolors="black",
        linewidths=0.8
    )

    # --------------------------------------------------------
    # Primary outputs
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fault nodes
    # --------------------------------------------------------

    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=fault_nodes,
        node_color="red",
        node_size=550,
        node_shape="X",
        edgecolors="black",
        linewidths=1.3
    )

    # --------------------------------------------------------
    # Labels ONLY on fault nodes
    # --------------------------------------------------------

    labels = {
        node: str(node)
        for node in fault_nodes
    }

    nx.draw_networkx_labels(
        H,
        pos,
        labels=labels,
        font_size=8,
        font_weight="bold"
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    plt.title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=20
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_elements = [

        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightblue",
            markeredgecolor="black",
            markersize=9,
            label="Normal circuit node"
        ),

        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=11,
            label=f"{fault_type} fault site"
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
            label="Circuit wire"
        )
    ]

    plt.legend(
        handles=legend_elements,
        loc="upper right"
    )

    # --------------------------------------------------------
    # Information box
    # --------------------------------------------------------

    plt.text(
        0.01,
        0.01,
        (
            f"Full PicoRV32: {num_nodes} nodes | "
            f"{num_edges} edges | "
            f"{len(faults)} faults\n"
            f"Displayed region: {H.number_of_nodes()} nodes | "
            f"{H.number_of_edges()} wires\n"
            f"{fault_type} fault nodes highlighted: "
            f"{len(fault_nodes)}"
        ),
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9
        )
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )

    print(
        f"{fault_type} nodes displayed:"
    )

    print(
        fault_nodes
    )


# ============================================================
# SA0 plot
# ============================================================

plot_fault_type(
    fault_type="SA0",
    fault_nodes=sa0_nodes,
    output_file=OUTPUT_DIR / "picorv32_SA0_faults.png",
    title="PicoRV32 Circuit — Stuck-at-0 Fault Sites"
)


# ============================================================
# SA1 plot
# ============================================================

plot_fault_type(
    fault_type="SA1",
    fault_nodes=sa1_nodes,
    output_file=OUTPUT_DIR / "picorv32_SA1_faults.png",
    title="PicoRV32 Circuit — Stuck-at-1 Fault Sites"
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    "Generated:"
)

print(
    "  data/evaluation_plots/picorv32_SA0_faults.png"
)

print(
    "  data/evaluation_plots/picorv32_SA1_faults.png"
)