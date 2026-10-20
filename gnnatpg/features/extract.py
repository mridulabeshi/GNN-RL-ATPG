"""
extract.py - Feature Extraction Pipeline

Combines gate-level features (from netlist parsing) with SCOAP
controllability/observability measures to create comprehensive
node feature vectors for GNN input.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from gnnatpg.graph_construction.circuit_graph import CircuitGraphBuilder, NUM_GATE_TYPES
from gnnatpg.features.scoap import SCOAPAnalyzer
from gnnatpg.graph_construction.netlist_parser import NetlistParser, Netlist

logger = logging.getLogger(__name__)


# Feature dimensions
FEATURE_DIMENSIONS = {
    "gate_type_onehot": NUM_GATE_TYPES,  # 20
    "fanout": 1,
    "depth": 1,
    "is_dff": 1,
    "is_pi": 1,
    "is_po": 1,
    "num_inputs": 1,
    "scoap_cc0": 1,
    "scoap_cc1": 1,
    "scoap_co": 1,
}
TOTAL_FEATURE_DIM = sum(FEATURE_DIMENSIONS.values())


class FeatureExtractor:
    """
    Full feature extraction pipeline:
        1. Parse netlist
        2. Run SCOAP analysis
        3. Build circuit graph with enriched features
        4. Return PyG Data object
    """

    def __init__(self):
        self.parser = NetlistParser()
        self.graph_builder = CircuitGraphBuilder()
        self.scoap = SCOAPAnalyzer()

    def extract(
        self,
        netlist_path: str | Path,
        scoap_results: Optional[dict] = None,
    ) -> Data:
        """
        Extract features from a netlist file.
        
        Args:
            netlist_path: Path to gate-level Verilog netlist
            scoap_results: Pre-computed SCOAP results (optional)
            
        Returns:
            PyG Data object with enriched node features
        """
        # 1. Parse netlist
        netlist = self.parser.parse_file(netlist_path)

        # 2. Run SCOAP if not provided
        if scoap_results is None:
            scoap_results = self.scoap.analyze(netlist)

        # 3. Build graph
        graph = self.graph_builder.build_from_netlist(netlist)

        # 4. Enrich features with SCOAP values
        graph = self._enrich_with_scoap(graph, netlist, scoap_results)

        # 5. Normalize features
        graph = self._normalize_features(graph)

        logger.info(
            f"Feature extraction complete: {graph.x.shape[0]} nodes, "
            f"{graph.x.shape[1]} features"
        )
        return graph

    def extract_from_netlist(
        self, netlist: Netlist
    ) -> Data:
        """Extract features from a pre-parsed Netlist object."""
        scoap_results = self.scoap.analyze(netlist)
        graph = self.graph_builder.build_from_netlist(netlist)
        graph = self._enrich_with_scoap(graph, netlist, scoap_results)
        graph = self._normalize_features(graph)
        return graph

    def _enrich_with_scoap(
        self,
        graph: Data,
        netlist: Netlist,
        scoap_results: dict,
    ) -> Data:
        """Replace SCOAP placeholder features with actual values."""
        if graph.x.shape[0] == 0:
            return graph

        # SCOAP features are at indices 20+1+1+1+1+1+1 = 26, 27, 28
        # (after gate_type_onehot(20) + fanout(1) + depth(1) +
        #  is_dff(1) + is_pi(1) + is_po(1) + num_inputs(1))
        scoap_cc0_idx = 26
        scoap_cc1_idx = 27
        scoap_co_idx = 28

        features = graph.x.clone()

        for i, gate_id in enumerate(graph.gate_ids):
            if gate_id in scoap_results:
                results = scoap_results[gate_id]
                # Store raw SCOAP values (will be normalized later)
                features[i, scoap_cc0_idx] = results["cc0"]
                features[i, scoap_cc1_idx] = results["cc1"]
                features[i, scoap_co_idx] = results["co"]

        graph.x = features
        return graph

    def _normalize_features(self, graph: Data) -> Data:
        """Normalize continuous features to [0, 1] range."""
        if graph.x.shape[0] == 0:
            return graph

        features = graph.x.clone()

        # Normalize fanout (index 20)
        fanout = features[:, 20]
        max_fanout = fanout.max()
        if max_fanout > 0:
            features[:, 20] = fanout / max_fanout

        # Normalize depth (index 21)
        depth = features[:, 21]
        max_depth = depth.max()
        if max_depth > 0:
            features[:, 21] = depth / max_depth

        # Normalize num_inputs (index 25)
        num_inputs = features[:, 25]
        max_inputs = num_inputs.max()
        if max_inputs > 0:
            features[:, 25] = num_inputs / max_inputs

        # Normalize SCOAP features (indices 26, 27, 28)
        for idx in [26, 27, 28]:
            col = features[:, idx]
            max_val = col.max()
            if max_val > 0:
                features[:, idx] = col / max_val

        graph.x = features
        return graph


def extract_all_features(
    netlist_paths: dict[str, str | Path],
    output_dir: str | Path = "data/features",
) -> dict[str, Data]:
    """
    Extract features for multiple netlists.
    
    Args:
        netlist_paths: Dict mapping core_name -> netlist_path
        output_dir: Directory to save extracted features
        
    Returns:
        Dict mapping core_name -> PyG Data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor()
    results = {}

    for core_name, netlist_path in netlist_paths.items():
        try:
            graph = extractor.extract(netlist_path)
            results[core_name] = graph

            # Save
            save_path = output_dir / f"{core_name}_features.pt"
            torch.save(graph, save_path)
            logger.info(f"Saved features for {core_name} to {save_path}")

        except Exception as e:
            logger.error(f"Feature extraction failed for {core_name}: {e}")

    return results
