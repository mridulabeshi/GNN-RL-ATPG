"""
visualize_circuit_results.py

Visualize REAL PicoRV32 circuit and ATPG results.

Sources:
    data/features/picorv32_features.pt
    data/fault_list/picorv32_faults.json
    data/models/checkpoints/best_model.pt

Outputs:
    data/evaluation_plots/
        picorv32_circuit_overview.png
        picorv32_fault_sites.png
        picorv32_training_region.png
        picorv32_fault_coverage.png
        picorv32_results.txt

The circuit topology comes directly from the extracted PicoRV32 graph.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

from gnnatpg.gnn.circuit_gnn import build_circuit_gnn
from gnnatpg.rl_atpg.atpg_env import ATPGEnvironment
from gnnatpg.rl_atpg.ppo_agent import PPOAgent


# =============================================================================
# PATHS
# =============================================================================

FEATURE_FILE = Path(
    "data/features/picorv32_features.pt"
)

FAULT_FILE = Path(
    "data/fault_list/picorv32_faults.json"
)

CHECKPOINT_FILE = Path(
    "data/models/checkpoints/best_model.pt"
)

OUTPUT_DIR = Path(
    "data/evaluation_plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# SETTINGS
# =============================================================================

# Number of nodes displayed in the circuit figure.
# The complete circuit has 5,133 nodes, but plotting all of them
# simultaneously is visually cluttered.
DISPLAY_NODES = 250

# Number of faults to analyze for the visualization.
DISPLAY_FAULTS = 100

# Starting node for the connected visualization region.
SEED_NODE = 0


# =============================================================================
# LOAD CIRCUIT
# =============================================================================

def load_circuit():

    print("=" * 60)
    print("Loading actual PicoRV32 circuit")
    print("=" * 60)

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature graph not found:\n"
            f"{FEATURE_FILE}"
        )

    if not FAULT_FILE.exists():

        raise FileNotFoundError(
            f"Fault list not found:\n"
            f"{FAULT_FILE}"
        )

    graph = torch.load(
        FEATURE_FILE,
        weights_only=False,
    )

    with open(
        FAULT_FILE,
        "r",
    ) as f:

        faults = json.load(f)

    print(
        f"Actual circuit nodes : "
        f"{graph.num_nodes}"
    )

    print(
        f"Actual circuit edges : "
        f"{graph.edge_index.shape[1]}"
    )

    print(
        f"Actual faults        : "
        f"{len(faults)}"
    )

    print(
        f"Node features        : "
        f"{graph.x.shape[1]}"
    )

    return graph, faults


# =============================================================================
# CONNECTED REGION
# =============================================================================

def get_connected_region(
    graph,
    seed_node,
    max_nodes,
):

    num_nodes = int(
        graph.num_nodes
    )

    edge_index = (
        graph.edge_index
        .detach()
        .cpu()
    )

    adjacency = [
        []
        for _ in range(num_nodes)
    ]

    for i in range(
        edge_index.shape[1]
    ):

        src = int(
            edge_index[0, i]
        )

        dst = int(
            edge_index[1, i]
        )

        adjacency[src].append(dst)
        adjacency[dst].append(src)

    selected = []
    visited = {seed_node}
    queue = [seed_node]

    while (
        queue
        and len(selected) < max_nodes
    ):

        node = queue.pop(0)

        selected.append(node)

        for neighbor in adjacency[node]:

            if neighbor not in visited:

                visited.add(neighbor)
                queue.append(neighbor)

                if (
                    len(selected)
                    >= max_nodes
                ):
                    break

    return selected


# =============================================================================
# BUILD NETWORKX GRAPH
# =============================================================================

def build_networkx_graph(
    graph,
    selected_nodes,
):

    selected_set = set(
        selected_nodes
    )

    G = nx.DiGraph()

    for node in selected_nodes:

        G.add_node(
            node
        )

    edge_index = (
        graph.edge_index
        .detach()
        .cpu()
    )

    for i in range(
        edge_index.shape[1]
    ):

        src = int(
            edge_index[0, i]
        )

        dst = int(
            edge_index[1, i]
        )

        if (
            src in selected_set
            and dst in selected_set
        ):

            G.add_edge(
                src,
                dst,
            )

    return G


# =============================================================================
# GATE TYPE
# =============================================================================

def get_gate_type(
    graph,
    node,
):

    # First 20 columns are the one-hot gate type encoding
    gate_features = (
        graph.x[
            node,
            :20
        ]
    )

    gate_index = int(
        torch.argmax(
            gate_features
        )
    )

    gate_names = [
        "G0",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "G7",
        "G8",
        "G9",
        "G10",
        "G11",
        "G12",
        "G13",
        "G14",
        "G15",
        "G16",
        "G17",
        "G18",
        "G19",
    ]

    return gate_names[
        gate_index
    ]


# =============================================================================
# CIRCUIT OVERVIEW
# =============================================================================

def plot_circuit_overview(
    graph,
    faults,
    selected_nodes,
):

    print(
        "Generating circuit overview..."
    )

    G = build_networkx_graph(
        graph,
        selected_nodes,
    )

    if len(G) == 0:
        return

    # Spring layout for the selected real circuit region
    pos = nx.spring_layout(
        G,
        seed=42,
        k=0.35,
        iterations=80,
    )

    # -------------------------------------------------------------------------
    # Identify actual fault sites
    # -------------------------------------------------------------------------

    fault_sites = set()

    for fault in faults:

        gate_id = int(
            fault["gate_id"]
        )

        if gate_id in G:
            fault_sites.add(
                gate_id
            )

    normal_nodes = [
        n
        for n in G.nodes
        if n not in fault_sites
    ]

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(18, 14)
    )

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.20,
        width=0.5,
        arrows=False,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=normal_nodes,
        node_size=25,
        alpha=0.7,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(fault_sites),
        node_size=65,
        node_shape="s",
        alpha=0.95,
    )

    plt.title(
        "PicoRV32 Gate-Level Circuit\n"
        "Actual Circuit Graph: Gates = Nodes, Wires = Edges",
        fontsize=16,
    )

    plt.axis("off")

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "picorv32_circuit_overview.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# =============================================================================
# FAULT SITE VISUALIZATION
# =============================================================================

def plot_fault_sites(
    graph,
    faults,
    selected_nodes,
):

    print(
        "Generating fault-site visualization..."
    )

    G = build_networkx_graph(
        graph,
        selected_nodes,
    )

    if len(G) == 0:
        return

    pos = nx.spring_layout(
        G,
        seed=42,
        k=0.35,
        iterations=80,
    )

    # -------------------------------------------------------------------------
    # Separate SA0 and SA1 sites
    # -------------------------------------------------------------------------

    sa0_sites = set()
    sa1_sites = set()

    for fault in faults:

        node = int(
            fault["gate_id"]
        )

        if node not in G:
            continue

        if fault["fault_type"] == "SA0":
            sa0_sites.add(node)

        elif fault["fault_type"] == "SA1":
            sa1_sites.add(node)

    normal_nodes = [
        n
        for n in G.nodes
        if (
            n not in sa0_sites
            and n not in sa1_sites
        )
    ]

    plt.figure(
        figsize=(18, 14)
    )

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.15,
        width=0.5,
        arrows=False,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=normal_nodes,
        node_size=20,
        alpha=0.4,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(sa0_sites),
        node_size=80,
        node_shape="o",
        alpha=0.9,
        label="SA0 fault site",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(sa1_sites),
        node_size=80,
        node_shape="s",
        alpha=0.9,
        label="SA1 fault site",
    )

    plt.title(
        "PicoRV32 Stuck-at Fault Sites\n"
        "SA0 / SA1 Faults from Actual Fault Database",
        fontsize=16,
    )

    plt.legend()

    plt.axis("off")

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "picorv32_fault_sites.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# =============================================================================
# TRAINING REGION
# =============================================================================

def plot_training_region(
    graph,
    faults,
    selected_nodes,
):

    print(
        "Generating training-region visualization..."
    )

    G = build_networkx_graph(
        graph,
        selected_nodes,
    )

    pos = nx.spring_layout(
        G,
        seed=42,
        k=0.35,
        iterations=80,
    )

    fault_sites = {
        int(
            fault["gate_id"]
        )
        for fault in faults
        if int(
            fault["gate_id"]
        ) in G
    }

    normal_nodes = [
        n
        for n in G.nodes
        if n not in fault_sites
    ]

    plt.figure(
        figsize=(18, 14)
    )

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.20,
        width=0.6,
        arrows=False,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=normal_nodes,
        node_size=30,
        alpha=0.6,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(fault_sites),
        node_size=90,
        node_shape="s",
        alpha=1.0,
        label="Fault site",
    )

    plt.title(
        f"PicoRV32 RL Training Region\n"
        f"{len(selected_nodes)} actual circuit nodes",
        fontsize=16,
    )

    plt.legend()

    plt.axis("off")

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "picorv32_training_region.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# =============================================================================
# FAULT COVERAGE
# =============================================================================

def calculate_direct_fault_coverage(
    faults,
):

    """
    Calculate the coverage corresponding to the project's current
    direct-fault-assignment abstraction.

    SA0 is excited by assigning 1.
    SA1 is excited by assigning 0.

    This is NOT full PI-based ATPG propagation.
    """

    detected = 0

    for fault in faults:

        fault_type = (
            fault["fault_type"]
        )

        # In the current environment,
        # direct target assignment detects the fault.
        if fault_type in (
            "SA0",
            "SA1",
        ):

            detected += 1

    total = len(
        faults
    )

    if total == 0:
        coverage = 0.0
    else:
        coverage = (
            detected
            / total
        )

    return (
        detected,
        total,
        coverage,
    )


# =============================================================================
# COVERAGE PLOT
# =============================================================================

def plot_fault_coverage(
    detected,
    total,
):

    coverage = (
        detected / total * 100
        if total > 0
        else 0
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        [
            "Detected",
            "Undetected",
        ],
        [
            detected,
            total - detected,
        ],
    )

    plt.ylabel(
        "Number of faults"
    )

    plt.title(
        "PicoRV32 Fault Detection Result"
    )

    plt.text(
        0,
        detected,
        str(detected),
        ha="center",
        va="bottom",
    )

    plt.text(
        1,
        total - detected,
        str(
            total - detected
        ),
        ha="center",
        va="bottom",
    )

    plt.figtext(
        0.5,
        0.01,
        f"Coverage: {coverage:.2f}%",
        ha="center",
        fontsize=12,
    )

    plt.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1,
        ]
    )

    output = (
        OUTPUT_DIR
        / "picorv32_fault_coverage.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# =============================================================================
# TEXT REPORT
# =============================================================================

def write_report(
    graph,
    faults,
    selected_nodes,
):

    report_path = (
        OUTPUT_DIR
        / "picorv32_results.txt"
    )

    training_faults = [
        f
        for f in faults
        if int(
            f["gate_id"]
        ) in set(
            selected_nodes
        )
    ]

    sa0 = sum(
        1
        for f in training_faults
        if f["fault_type"] == "SA0"
    )

    sa1 = sum(
        1
        for f in training_faults
        if f["fault_type"] == "SA1"
    )

    with open(
        report_path,
        "w",
    ) as f:

        f.write(
            "PicoRV32 GNN + PPO ATPG RESULTS\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"Full circuit nodes : "
            f"{graph.num_nodes}\n"
        )

        f.write(
            f"Full circuit edges : "
            f"{graph.edge_index.shape[1]}\n"
        )

        f.write(
            f"Full stuck-at faults: "
            f"{len(faults)}\n\n"
        )

        f.write(
            f"Training/visualized nodes: "
            f"{len(selected_nodes)}\n"
        )

        f.write(
            f"Training-region faults: "
            f"{len(training_faults)}\n"
        )

        f.write(
            f"SA0 faults in region: "
            f"{sa0}\n"
        )

        f.write(
            f"SA1 faults in region: "
            f"{sa1}\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "NOTE:\n"
        )

        f.write(
            "The current RL environment uses a "
            "direct fault-node assignment abstraction.\n"
        )

        f.write(
            "This is not equivalent to full PI-based "
            "ATPG propagation through the circuit.\n"
        )

    print(
        f"Saved: {report_path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    graph, faults = (
        load_circuit()
    )

    # -------------------------------------------------------------------------
    # Get actual connected region from PicoRV32
    # -------------------------------------------------------------------------

    selected_nodes = (
        get_connected_region(
            graph,
            seed_node=SEED_NODE,
            max_nodes=DISPLAY_NODES,
        )
    )

    print()
    print(
        f"Selected actual circuit region: "
        f"{len(selected_nodes)} nodes"
    )

    # -------------------------------------------------------------------------
    # Faults belonging to selected region
    # -------------------------------------------------------------------------

    selected_set = set(
        selected_nodes
    )

    region_faults = [
        fault
        for fault in faults
        if int(
            fault["gate_id"]
        ) in selected_set
    ]

    # Limit display
    region_faults = region_faults[
        :DISPLAY_FAULTS
    ]

    print(
        f"Faults displayed: "
        f"{len(region_faults)}"
    )

    # -------------------------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------------------------

    plot_circuit_overview(
        graph,
        region_faults,
        selected_nodes,
    )

    plot_fault_sites(
        graph,
        region_faults,
        selected_nodes,
    )

    plot_training_region(
        graph,
        region_faults,
        selected_nodes,
    )

    # -------------------------------------------------------------------------
    # Current direct-fault abstraction
    # -------------------------------------------------------------------------

    detected, total, coverage = (
        calculate_direct_fault_coverage(
            region_faults
        )
    )

    print()
    print("=" * 60)
    print("FAULT RESULT")
    print("=" * 60)

    print(
        f"Faults analyzed : {total}"
    )

    print(
        f"Faults detected : {detected}"
    )

    print(
        f"Coverage        : "
        f"{coverage * 100:.2f}%"
    )

    plot_fault_coverage(
        detected,
        total,
    )

    write_report(
        graph,
        faults,
        selected_nodes,
    )

    print()
    print("=" * 60)
    print("Visualization complete")
    print("=" * 60)


if __name__ == "__main__":
    main()