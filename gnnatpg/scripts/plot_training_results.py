"""
Review-2 Visualization Script
=============================

Uses only data that actually exists in the project.

Generates:

1. GNN node embedding (PCA 2D)
2. PPO checkpoint parameter-change graph
3. PicoRV32 training-region visualization

Does NOT fabricate missing reward/coverage history.
"""

from pathlib import Path
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.decomposition import PCA

from gnnatpg.gnn.circuit_gnn import CircuitGNN


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT
    / "data"
    / "features"
    / "picorv32_features.pt"
)

CHECKPOINT_DIR = (
    ROOT
    / "data"
    / "models"
    / "checkpoints"
)

BEST_MODEL = CHECKPOINT_DIR / "best_model.pt"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "evaluation_plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("review2_visualization")


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    logger.info(
        "Loading PicoRV32 feature graph..."
    )

    data = torch.load(
        FEATURE_FILE,
        map_location="cpu",
        weights_only=False
    )

    logger.info(
        "Nodes       : %d",
        data.x.shape[0]
    )

    logger.info(
        "Node features: %d",
        data.x.shape[1]
    )

    if hasattr(data, "edge_index"):

        logger.info(
            "Edges       : %d",
            data.edge_index.shape[1]
        )

    return data


# ============================================================
# GRAPH 1
# GNN NODE EMBEDDING — PCA
# ============================================================

def plot_gnn_embedding(data):

    logger.info(
        "============================================"
    )

    logger.info(
        "Generating GNN node embedding visualization..."
    )

    if not BEST_MODEL.exists():

        logger.warning(
            "Best model not found: %s",
            BEST_MODEL
        )

        return

    # --------------------------------------------------------
    # Create GNN using actual project constructor
    # --------------------------------------------------------

    gnn = CircuitGNN(
        in_dim=data.x.shape[1],
        hidden_dim=128,
        out_dim=64,
        num_layers=4,
        architecture="GCN",
        dropout=0.1,
        use_batch_norm=True,
        edge_dim=None
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        BEST_MODEL,
        map_location="cpu",
        weights_only=False
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    logger.info(
        "Loaded best_model.pt"
    )

    # --------------------------------------------------------
    # Extract GNN parameters
    # --------------------------------------------------------

    gnn_state = {}

    for key, value in state_dict.items():

        if key.startswith("gnn."):

            new_key = key[
                len("gnn.") :
            ]

            gnn_state[new_key] = value

        elif key.startswith(
            "actor_critic.gnn."
        ):

            new_key = key[
                len("actor_critic.gnn.") :
            ]

            gnn_state[new_key] = value

    if not gnn_state:

        logger.warning(
            "No GNN parameters found in checkpoint."
        )

        logger.info(
            "Available GNN-related keys:"
        )

        for key in state_dict.keys():

            if "gnn" in key.lower():

                logger.info(
                    "  %s",
                    key
                )

        return

    logger.info(
        "GNN parameter tensors found: %d",
        len(gnn_state)
    )

    # --------------------------------------------------------
    # Load GNN weights
    # --------------------------------------------------------

    missing, unexpected = gnn.load_state_dict(
        gnn_state,
        strict=False
    )

    logger.info(
        "Missing parameters  : %d",
        len(missing)
    )

    logger.info(
        "Unexpected parameters: %d",
        len(unexpected)
    )

    gnn.eval()

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    with torch.no_grad():

        embeddings = gnn(data)

    # Some GNN implementations may return
    # multiple outputs.

    if isinstance(
        embeddings,
        tuple
    ):

        embeddings = embeddings[0]

    embeddings = (
        embeddings
        .detach()
        .cpu()
        .numpy()
    )

    logger.info(
        "Embedding shape: %s",
        embeddings.shape
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=2
    )

    embedding_2d = pca.fit_transform(
        embeddings
    )

    variance = (
        pca.explained_variance_ratio_
        * 100
    )

    logger.info(
        "PC1 variance: %.2f%%",
        variance[0]
    )

    logger.info(
        "PC2 variance: %.2f%%",
        variance[1]
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        s=8,
        alpha=0.60
    )

    plt.xlabel(
        "Principal Component 1"
    )

    plt.ylabel(
        "Principal Component 2"
    )

    plt.title(
        "PicoRV32 GNN Node Embedding"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "gnn_embedding_2d.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(
        "Saved: %s",
        output
    )


# ============================================================
# GRAPH 2
# PPO CHECKPOINT PARAMETER CHANGE
# ============================================================

def get_checkpoint_episodes():

    checkpoints = []

    for file in CHECKPOINT_DIR.glob(
        "episode_*.pt"
    ):

        try:

            episode = int(
                file.stem.split("_")[1]
            )

            checkpoints.append(
                (episode, file)
            )

        except (
            ValueError,
            IndexError
        ):

            continue

    checkpoints.sort(
        key=lambda x: x[0]
    )

    return checkpoints


def get_model_distance(
    reference_state,
    current_state
):

    total_squared = 0.0
    count = 0

    for key in reference_state:

        if key not in current_state:
            continue

        a = reference_state[key]

        b = current_state[key]

        if not (
            torch.is_tensor(a)
            and torch.is_tensor(b)
        ):
            continue

        if a.shape != b.shape:
            continue

        difference = (
            a.float() - b.float()
        )

        total_squared += (
            difference.pow(2)
            .sum()
            .item()
        )

        count += (
            difference.numel()
        )

    if count == 0:
        return 0.0

    return (
        total_squared
        / count
    ) ** 0.5


def plot_checkpoint_progress():

    logger.info(
        "============================================"
    )

    logger.info(
        "Generating PPO checkpoint progression..."
    )

    checkpoints = (
        get_checkpoint_episodes()
    )

    if len(checkpoints) < 2:

        logger.warning(
            "Not enough checkpoints."
        )

        return

    logger.info(
        "Found %d checkpoints.",
        len(checkpoints)
    )

    # --------------------------------------------------------
    # Load episode 0 as reference
    # --------------------------------------------------------

    first_episode, first_file = (
        checkpoints[0]
    )

    first_checkpoint = torch.load(
        first_file,
        map_location="cpu",
        weights_only=False
    )

    reference_state = (
        first_checkpoint[
            "model_state_dict"
        ]
    )

    episodes = []
    distances = []

    # --------------------------------------------------------
    # Compare every checkpoint with episode 0
    # --------------------------------------------------------

    for episode, file in checkpoints:

        checkpoint = torch.load(
            file,
            map_location="cpu",
            weights_only=False
        )

        state = checkpoint[
            "model_state_dict"
        ]

        distance = get_model_distance(
            reference_state,
            state
        )

        episodes.append(
            episode
        )

        distances.append(
            distance
        )

        logger.info(
            "Episode %d -> parameter distance %.6f",
            episode,
            distance
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        episodes,
        distances,
        marker="o",
        linewidth=2,
        markersize=4
    )

    plt.xlabel(
        "Training Episode Checkpoint"
    )

    plt.ylabel(
        "Model Parameter Distance from Episode 0"
    )

    plt.title(
        "PPO Model Parameter Evolution During Training"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "ppo_checkpoint_progress.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(
        "Saved: %s",
        output
    )


# ============================================================
# GRAPH 3
# PICORV32 TRAINING REGION
# ============================================================

def build_training_region(
    data,
    num_nodes=2000,
    seed_node=0
):

    logger.info(
        "Building training-region visualization..."
    )

    edge_index = (
        data.edge_index
        .detach()
        .cpu()
        .numpy()
    )

    graph = nx.Graph()

    # --------------------------------------------------------
    # Add all nodes
    # --------------------------------------------------------

    graph.add_nodes_from(
        range(data.x.shape[0])
    )

    # --------------------------------------------------------
    # Add edges
    # --------------------------------------------------------

    edges = edge_index.T.tolist()

    graph.add_edges_from(
        edges
    )

    # --------------------------------------------------------
    # BFS training region
    # --------------------------------------------------------

    visited = set(
        [seed_node]
    )

    queue = [
        seed_node
    ]

    while queue and len(
        visited
    ) < num_nodes:

        current = queue.pop(
            0
        )

        for neighbor in graph.neighbors(
            current
        ):

            if neighbor not in visited:

                visited.add(
                    neighbor
                )

                queue.append(
                    neighbor
                )

                if len(
                    visited
                ) >= num_nodes:

                    break

    return graph, visited


def plot_training_region(data):

    logger.info(
        "============================================"
    )

    logger.info(
        "Generating PicoRV32 training-region graph..."
    )

    graph, training_nodes = (
        build_training_region(
            data,
            num_nodes=2000,
            seed_node=0
        )
    )

    logger.info(
        "Training-region nodes: %d",
        len(training_nodes)
    )

    # --------------------------------------------------------
    # For readability, plot a reduced number of edges
    # while keeping the complete selected node region.
    # --------------------------------------------------------

    selected_edges = []

    for u, v in graph.edges():

        if (
            u in training_nodes
            and v in training_nodes
        ):

            selected_edges.append(
                (u, v)
            )

    logger.info(
        "Training-region edges: %d",
        len(selected_edges)
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    subgraph = graph.subgraph(
        training_nodes
    ).copy()

    logger.info(
        "Calculating graph layout..."
    )

    # spring layout can be expensive for 2000 nodes,
    # therefore use a fixed seed.

    position = nx.spring_layout(
        subgraph,
        seed=42,
        iterations=30
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 10)
    )

    nx.draw_networkx_edges(
        subgraph,
        position,
        alpha=0.08,
        width=0.3
    )

    nx.draw_networkx_nodes(
        subgraph,
        position,
        node_size=5,
        alpha=0.75
    )

    plt.title(
        "PicoRV32 PPO Training Region\n"
        "2000 Nodes from the 5133-Node Circuit"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "picorv32_training_region.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(
        "Saved: %s",
        output
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print("GNN + PPO REVIEW-2 VISUALIZATION")
    print("=" * 65)
    print()

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    data = load_features()

    # --------------------------------------------------------
    # Graph 1
    # --------------------------------------------------------

    plot_gnn_embedding(
        data
    )

    # --------------------------------------------------------
    # Graph 2
    # --------------------------------------------------------

    plot_checkpoint_progress()

    # --------------------------------------------------------
    # Graph 3
    # --------------------------------------------------------

    plot_training_region(
        data
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("GENERATED FILES")
    print("=" * 65)

    print(
        OUTPUT_DIR
        / "gnn_embedding_2d.png"
    )

    print(
        OUTPUT_DIR
        / "ppo_checkpoint_progress.png"
    )

    print(
        OUTPUT_DIR
        / "picorv32_training_region.png"
    )

    print("=" * 65)
    print()


if __name__ == "__main__":
    main()