"""
circuit_graph.py - Circuit Graph Construction

Converts parsed netlists into PyTorch Geometric Data objects for GNN processing.

Graph representation:
    - Nodes: Logic gates (AND, OR, NOT, DFF, etc.)
    - Edges: Wire connections between gates
    - Node features: Gate type, fanout, depth, SCOAP, logic state, fault info
    - Edge features: Wire type, connectivity
"""

import logging
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_networkx

from gnnatpg.graph_construction.netlist_parser import (
    NetlistParser,
    Netlist,
    Gate,
    GateType,
    Wire,
)

logger = logging.getLogger(__name__)


# Gate type to integer mapping for one-hot encoding
GATE_TYPE_MAP = {
    GateType.AND2: 0,
    GateType.AND3: 1,
    GateType.AND4: 2,
    GateType.OR2: 3,
    GateType.OR3: 4,
    GateType.OR4: 5,
    GateType.NAND2: 6,
    GateType.NAND3: 7,
    GateType.NAND4: 8,
    GateType.NOR2: 9,
    GateType.NOR3: 10,
    GateType.NOR4: 11,
    GateType.XOR2: 12,
    GateType.XNOR2: 13,
    GateType.NOT: 14,
    GateType.BUF: 15,
    GateType.MUX2: 16,
    GateType.DFF: 17,
    GateType.DFFE: 18,
    GateType.UNKNOWN: 19,
}

NUM_GATE_TYPES = len(GATE_TYPE_MAP)


class CircuitGraphBuilder:
    """
    Builds PyTorch Geometric graph from a parsed netlist.

    Each gate becomes a node. Wire connections become directed edges
    from the driving gate's output to the driven gate's inputs.
    """

    def __init__(self):
        self.parser = NetlistParser()

    def build_from_file(self, netlist_path: str | Path) -> Data:
        """Build a PyG Data object from a netlist file."""
        netlist = self.parser.parse_file(netlist_path)
        return self.build_graph(netlist)

    def build_from_netlist(self, netlist: Netlist) -> Data:
        """Build a PyG Data object from a parsed Netlist."""
        return self.build_graph(netlist)

    def build_graph(self, netlist: Netlist) -> Data:
        """
        Build PyG Data from parsed netlist.

        Returns:
            torch_geometric.data.Data with:
                - x: Node feature matrix [num_nodes, num_features]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge feature matrix [num_edges, edge_features]
                - y: Target labels (for fault simulation, optional)
                - gate_ids: Mapping from node index to gate_id
                - gate_names: Mapping from node index to instance name
        """

        # Reset caches for each new netlist.
        self._depth_cache = {}
        self._gate_lookup = {
            gate.gate_id: gate
            for gate in netlist.gates
        }
        self._wire_driver_lookup = {
            wire.name: wire.driven_by
            for wire in netlist.wires.values()
        }

        # Create gate_id to node_index mapping
        gate_to_idx = {}
        node_features = []
        node_gate_ids = []
        node_gate_names = []

        for i, gate in enumerate(netlist.gates):
            gate_to_idx[gate.gate_id] = i
            node_gate_ids.append(gate.gate_id)
            node_gate_names.append(gate.instance_name)

            # Node feature vector
            features = self._extract_node_features(
                gate,
                netlist,
                i,
            )
            node_features.append(features)

        # Build edge index
        edge_src = []
        edge_dst = []
        edge_features = []

        for gate in netlist.gates:
            gate_idx = gate_to_idx.get(gate.gate_id)

            if gate_idx is None:
                continue

            # For each input net, find the driving gate
            for input_net in gate.input_nets:
                wire = netlist.wires.get(input_net)

                if wire and wire.driven_by is not None:
                    driver_idx = gate_to_idx.get(wire.driven_by)

                    if driver_idx is not None:
                        edge_src.append(driver_idx)
                        edge_dst.append(gate_idx)

                        # Edge features: [is_bus, is_clock, weight]
                        edge_feat = self._extract_edge_features(
                            wire,
                            netlist,
                        )
                        edge_features.append(edge_feat)

        # Convert to tensors
        x = torch.tensor(
            node_features,
            dtype=torch.float,
        )

        if edge_src:
            edge_index = torch.tensor(
                [edge_src, edge_dst],
                dtype=torch.long,
            )

            edge_attr = torch.tensor(
                edge_features,
                dtype=torch.float,
            )
        else:
            edge_index = torch.zeros(
                (2, 0),
                dtype=torch.long,
            )

            edge_attr = torch.zeros(
                (0, 3),
                dtype=torch.float,
            )

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Store metadata
        data.num_nodes = len(netlist.gates)
        data.gate_ids = node_gate_ids
        data.gate_names = node_gate_names
        data.netlist_stats = netlist.stats()

        logger.info(
            f"Built graph: {data.num_nodes} nodes, "
            f"{edge_index.shape[1]} edges, "
            f"{x.shape[1]} features per node"
        )

        return data

    def _extract_node_features(
        self,
        gate: Gate,
        netlist: Netlist,
        node_idx: int,
    ) -> list[float]:
        """
        Extract feature vector for a gate node.

        Features:
            - gate_type: One-hot (20 values)
            - fanout: Integer count
            - depth: Topological depth
            - is_dff: Binary (is flip-flop)
            - is_primary_input: Binary
            - is_primary_output: Binary
            - num_inputs: Number of input pins
            - (SCOAP features will be added by the feature extraction module)
        """

        features = []

        # 1. Gate type one-hot encoding
        gate_type_idx = GATE_TYPE_MAP.get(
            gate.gate_type,
            GATE_TYPE_MAP[GateType.UNKNOWN],
        )

        one_hot = [0.0] * NUM_GATE_TYPES
        one_hot[gate_type_idx] = 1.0
        features.extend(one_hot)

        # 2. Fanout count
        wire = netlist.wires.get(gate.output_net)
        fanout = len(wire.fanout_gate_ids) if wire else 0
        features.append(float(fanout))

        # 3. Depth
        depth = self._compute_depth(
            gate,
            netlist,
            node_idx,
        )
        features.append(float(depth))

        # 4. Is flip-flop
        features.append(
            1.0
            if gate.gate_type in (GateType.DFF, GateType.DFFE)
            else 0.0
        )

        # 5. Is primary input
        is_pi = any(
            net in netlist.primary_inputs
            for net in gate.input_nets
        )
        features.append(1.0 if is_pi else 0.0)

        # 6. Is primary output
        is_po = gate.output_net in netlist.primary_outputs
        features.append(1.0 if is_po else 0.0)

        # 7. Number of input pins
        features.append(float(len(gate.input_nets)))

        # 8. Placeholder for SCOAP controllability-0
        features.append(0.0)

        # 9. Placeholder for SCOAP controllability-1
        features.append(0.0)

        # 10. Placeholder for SCOAP observability
        features.append(0.0)

        return features

    def _extract_edge_features(
        self,
        wire: Wire,
        netlist: Netlist,
    ) -> list[float]:
        """
        Extract edge features.

        Features:
            - is_bus: Binary (is multi-bit wire)
            - fanout_weight: Normalized fanout
            - is_feedback: Binary (feedback loop)
        """

        is_bus = 1.0 if wire.width > 1 else 0.0
        fanout_weight = min(
            len(wire.fanout_gate_ids) / 10.0,
            1.0,
        )

        # Will be computed properly later
        is_feedback = 0.0

        return [
            is_bus,
            fanout_weight,
            is_feedback,
        ]

    def _compute_depth(
        self,
        gate,
        netlist,
        node_idx,
    ):
        """
        Compute approximate logic depth of a gate.

        Depth is calculated by walking backwards through the circuit:
        primary-input driven gates have depth 0, and each additional
        logic level increases depth by 1.

        A visited set prevents infinite traversal through feedback loops.
        """

        # Cache depths so the same gate does not get recomputed repeatedly.
        if gate.gate_id in self._depth_cache:
            return self._depth_cache[gate.gate_id]

        max_depth = 0

        # (gate, current_depth)
        stack = [(gate, 0)]

        # Prevent cycles/feedback from causing infinite traversal.
        visited = set()

        while stack:
            current_gate, current_depth = stack.pop()

            if current_gate.gate_id in visited:
                continue

            visited.add(current_gate.gate_id)

            # Depth limit prevents pathological feedback circuits.
            if current_depth >= 50:
                max_depth = max(
                    max_depth,
                    current_depth,
                )
                continue

            for input_net in current_gate.input_nets:
                driver_id = self._wire_driver_lookup.get(input_net)

                # No driver means this is a primary input / undriven net.
                if driver_id is None:
                    continue

                driver_gate = self._gate_lookup.get(driver_id)

                if driver_gate is None:
                    continue

                next_depth = current_depth + 1

                max_depth = max(
                    max_depth,
                    next_depth,
                )

                if driver_gate.gate_id not in visited:
                    stack.append(
                        (driver_gate, next_depth)
                    )

        self._depth_cache[gate.gate_id] = max_depth

        return max_depth


class BatchGraphBuilder:
    """Builds batched graphs for multiple netlists."""

    def __init__(self):
        self.builder = CircuitGraphBuilder()

    def build_batch(
        self,
        netlist_paths: list[str | Path],
    ) -> list[Data]:
        """Build a list of graphs from multiple netlists."""

        graphs = []

        for path in netlist_paths:
            try:
                graph = self.builder.build_from_file(path)
                graphs.append(graph)

            except Exception as e:
                logger.error(
                    f"Failed to build graph from {path}: {e}"
                )

        return graphs

    @staticmethod
    def collate(graphs: list[Data]) -> Data:
        """Collate multiple PyG Data objects into a batch."""

        from torch_geometric.data import Batch

        return Batch.from_data_list(graphs)


def save_graph(
    data: Data,
    path: str | Path,
):
    """Save a PyG Data object to disk."""

    torch.save(data, path)
    logger.info(f"Saved graph to {path}")


def load_graph(
    path: str | Path,
) -> Data:
    """Load a PyG Data object from disk."""

    data = torch.load(
        path,
        weights_only=False,
    )

    logger.info(f"Loaded graph from {path}")

    return data