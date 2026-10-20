"""
circuit_gnn.py - Graph Neural Network for Circuit Representation

Implements multiple GNN architectures (GCN, GAT, GraphSAGE) for learning
structural representations of digital circuits. These representations are
used by the RL agent to guide ATPG decisions.

Based on:
    - SmartATPG (DAC 2024): GCN + RL for ATPG
    - FPGNN-ATPG (GLSVLSI 2023): Fault-parallel GNN ATPG
    - InF-ATPG (2025): FFR-driven GNN-RL ATPG
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)

logger = logging.getLogger(__name__)


class CircuitGNN(nn.Module):
    """
    GNN encoder for circuit graphs.

    Supports:
        - GCN (Graph Convolutional Network)
        - GAT (Graph Attention Network)
        - GraphSAGE

    The encoder produces:
        - Node embeddings for each gate
        - Graph-level embedding for the entire circuit
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 4,
        architecture: str = "GCN",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.architecture = architecture.upper()
        self.dropout = dropout

        # ---------------------------------------------------------
        # Build GNN layers
        # ---------------------------------------------------------
        self.convs = nn.ModuleList()

        self.norms = (
            nn.ModuleList()
            if use_batch_norm
            else None
        )

        for i in range(num_layers):

            layer_in = (
                in_dim
                if i == 0
                else hidden_dim
            )

            layer_out = (
                out_dim
                if i == num_layers - 1
                else hidden_dim
            )

            # -----------------------------------------------------
            # GCN
            # -----------------------------------------------------
            if self.architecture == "GCN":

                conv = GCNConv(
                    layer_in,
                    layer_out,
                )

            # -----------------------------------------------------
            # GAT
            # -----------------------------------------------------
            elif self.architecture == "GAT":

                heads = (
                    4
                    if i < num_layers - 1
                    else 1
                )

                conv = GATConv(
                    layer_in,
                    layer_out // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )

            # -----------------------------------------------------
            # GraphSAGE
            # -----------------------------------------------------
            elif self.architecture == "GRAPHSAGE":

                conv = SAGEConv(
                    layer_in,
                    layer_out,
                    normalize=True,
                )

            else:

                raise ValueError(
                    f"Unknown architecture: "
                    f"{self.architecture}"
                )

            self.convs.append(conv)

            # -----------------------------------------------------
            # Batch normalization
            # -----------------------------------------------------
            if use_batch_norm:

                self.norms.append(
                    nn.BatchNorm1d(layer_out)
                )

        # ---------------------------------------------------------
        # Node-level MLP
        # ---------------------------------------------------------
        # Refines the final GNN node embeddings.
        # Output dimension remains out_dim.
        # ---------------------------------------------------------
        self.node_mlp = nn.Sequential(

            nn.Linear(
                out_dim,
                out_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                out_dim,
                out_dim,
            ),
        )

        # ---------------------------------------------------------
        # Graph-level readout
        # ---------------------------------------------------------
        # Uses:
        #   1. Mean pooling
        #   2. Max pooling
        #   3. Add/Sum pooling
        #
        # Each produces out_dim features.
        # Concatenated dimension = out_dim * 3.
        # ---------------------------------------------------------
        self.graph_readout = nn.Sequential(

            nn.Linear(
                out_dim * 3,
                out_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                out_dim,
                out_dim,
            ),
        )

        logger.info(
            f"CircuitGNN: {self.architecture}, "
            f"{num_layers} layers, "
            f"in={in_dim}, "
            f"hidden={hidden_dim}, "
            f"out={out_dim}"
        )

    def forward(
        self,
        data: Data,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            data:
                PyG Data or Batch containing:

                x:
                    Node features

                edge_index:
                    Graph connectivity

                edge_attr:
                    Optional edge features

                batch:
                    Optional graph assignment tensor

        Returns:
            node_embeddings:
                [num_nodes, out_dim]

            graph_embedding:
                [batch_size, out_dim]
        """

        # ---------------------------------------------------------
        # Input data
        # ---------------------------------------------------------
        x = data.x

        edge_index = data.edge_index

        # Edge attributes are used by GAT.
        edge_attr = (
            data.edge_attr
            if hasattr(data, "edge_attr")
            else None
        )

        # ---------------------------------------------------------
        # Batch vector
        # ---------------------------------------------------------
        # For a single graph, create a batch vector containing
        # only zeros.
        # ---------------------------------------------------------
        if hasattr(data, "batch"):

            batch = data.batch

        else:

            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device,
            )

        # ---------------------------------------------------------
        # GNN message passing
        # ---------------------------------------------------------
        for i, conv in enumerate(self.convs):

            # -----------------------------------------------------
            # GCN
            # -----------------------------------------------------
            if self.architecture == "GCN":

                x_new = conv(
                    x,
                    edge_index,
                )

            # -----------------------------------------------------
            # GAT
            # -----------------------------------------------------
            elif self.architecture == "GAT":

                x_new = conv(
                    x,
                    edge_index,
                    edge_attr=edge_attr,
                )

            # -----------------------------------------------------
            # GraphSAGE
            # -----------------------------------------------------
            elif self.architecture == "GRAPHSAGE":

                x_new = conv(
                    x,
                    edge_index,
                )

            else:

                x_new = conv(
                    x,
                    edge_index,
                )

            # -----------------------------------------------------
            # Batch normalization
            # -----------------------------------------------------
            if self.norms is not None:

                x_new = self.norms[i](
                    x_new
                )

            # -----------------------------------------------------
            # Activation + dropout
            # -----------------------------------------------------
            # The final GNN layer does not receive ReLU/dropout.
            # -----------------------------------------------------
            if i < self.num_layers - 1:

                x_new = F.relu(
                    x_new
                )

                x_new = F.dropout(
                    x_new,
                    p=self.dropout,
                    training=self.training,
                )

            x = x_new

        # ---------------------------------------------------------
        # Node embeddings
        # ---------------------------------------------------------
        node_embeddings = self.node_mlp(
            x
        )

        # ---------------------------------------------------------
        # Graph-level pooling
        # ---------------------------------------------------------
        graph_emb_mean = global_mean_pool(
            node_embeddings,
            batch,
        )

        graph_emb_max = global_max_pool(
            node_embeddings,
            batch,
        )

        graph_emb_add = global_add_pool(
            node_embeddings,
            batch,
        )

        # ---------------------------------------------------------
        # Concatenate pooling results
        # ---------------------------------------------------------
        graph_emb = torch.cat(
            [
                graph_emb_mean,
                graph_emb_max,
                graph_emb_add,
            ],
            dim=-1,
        )

        # ---------------------------------------------------------
        # Graph embedding
        # ---------------------------------------------------------
        graph_embedding = self.graph_readout(
            graph_emb
        )

        return (
            node_embeddings,
            graph_embedding,
        )


class FaultAwareCircuitGNN(nn.Module):
    """
    Extended GNN that incorporates fault information.

    Takes the current ATPG state (fault site, current values) as
    additional input alongside circuit features.

    Used by the RL agent to make fault-specific decisions.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 4,
        architecture: str = "GCN",
        fault_embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fault_embed_dim = fault_embed_dim

        # ---------------------------------------------------------
        # Circuit GNN backbone
        # ---------------------------------------------------------
        self.circuit_gnn = CircuitGNN(
            in_dim=in_dim + fault_embed_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            architecture=architecture,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Fault embedding
        # ---------------------------------------------------------
        # Input:
        #   0 -> fault type
        #   1 -> fault location normalized
        #   2 -> whether fault is detected
        #
        # Shape:
        #   [num_nodes, 3]
        #
        # Output:
        #   [num_nodes, fault_embed_dim]
        # ---------------------------------------------------------
        self.fault_embedding = nn.Sequential(

            nn.Linear(
                3,
                fault_embed_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                fault_embed_dim,
                fault_embed_dim,
            ),
        )

    def forward(
        self,
        data: Data,
        fault_info: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with fault awareness.

        Args:
            data:
                Circuit graph.

            fault_info:
                [num_nodes, 3] tensor containing:

                    fault_type
                    fault_location
                    is_detected

        Returns:
            node_embeddings:
                [num_nodes, out_dim]

            graph_embedding:
                [batch_size, out_dim]
        """

        # ---------------------------------------------------------
        # Embed fault information
        # ---------------------------------------------------------
        fault_emb = self.fault_embedding(
            fault_info
        )

        # ---------------------------------------------------------
        # Copy graph so original graph isn't modified
        # ---------------------------------------------------------
        data = data.clone()

        # ---------------------------------------------------------
        # Concatenate circuit features and fault features
        # ---------------------------------------------------------
        data.x = torch.cat(
            [
                data.x,
                fault_emb,
            ],
            dim=-1,
        )

        # ---------------------------------------------------------
        # Run circuit GNN
        # ---------------------------------------------------------
        return self.circuit_gnn(
            data
        )


class CircuitGNNDecoder(nn.Module):
    """
    Decoder that uses circuit embeddings to predict ATPG decisions.

    Takes node embeddings and produces action probabilities/logits
    for the RL agent.

    Example actions could include:
        - Select a gate
        - Set a gate to 0
        - Set a gate to 1
    """

    def __init__(
        self,
        embedding_dim: int,
        num_actions: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.decoder = nn.Sequential(

            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                num_actions,
            ),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict action logits from node embeddings.

        Args:
            node_embeddings:
                [num_nodes, embedding_dim]

        Returns:
            action_logits:
                [num_nodes, num_actions]
        """

        return self.decoder(
            node_embeddings
        )


def build_circuit_gnn(
    config: dict,
) -> CircuitGNN:
    """
    Build CircuitGNN from config dictionary.
    """

    gnn_config = config.get(
        "gnn",
        {},
    )

    # Total node feature dimension.
    from gnnatpg.features.extract import (
        TOTAL_FEATURE_DIM,
    )

    return CircuitGNN(

        in_dim=TOTAL_FEATURE_DIM,

        hidden_dim=gnn_config.get(
            "hidden_dim",
            128,
        ),

        out_dim=gnn_config.get(
            "embedding_dim",
            64,
        ),

        num_layers=gnn_config.get(
            "num_layers",
            4,
        ),

        architecture=gnn_config.get(
            "architecture",
            "GCN",
        ),

        dropout=gnn_config.get(
            "dropout",
            0.1,
        ),

        use_batch_norm=gnn_config.get(
            "batch_norm",
            True,
        ),

        edge_dim=(
            gnn_config.get(
                "edge_feature_dim",
                3,
            )
            if gnn_config.get(
                "use_edge_features"
            )
            else None
        ),
    )


def build_fault_aware_gnn(
    config: dict,
) -> FaultAwareCircuitGNN:
    """
    Build FaultAwareCircuitGNN from config dictionary.
    """

    gnn_config = config.get(
        "gnn",
        {},
    )

    from gnnatpg.features.extract import (
        TOTAL_FEATURE_DIM,
    )

    return FaultAwareCircuitGNN(

        in_dim=TOTAL_FEATURE_DIM,

        hidden_dim=gnn_config.get(
            "hidden_dim",
            128,
        ),

        out_dim=gnn_config.get(
            "embedding_dim",
            64,
        ),

        num_layers=gnn_config.get(
            "num_layers",
            4,
        ),

        architecture=gnn_config.get(
            "architecture",
            "GCN",
        ),

        dropout=gnn_config.get(
            "dropout",
            0.1,
        ),
    )