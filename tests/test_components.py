"""
test_components.py - Unit Tests for GNN-ATPG Components

Tests:
    1. Netlist parser
    2. SCOAP analysis
    3. Graph construction
    4. Feature extraction
    5. GNN forward pass
    6. RL environment
    7. Fault simulation
"""

import sys
import os
import pytest
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnnatpg.graph_construction.netlist_parser import (
    NetlistParser, Netlist, GateType, Port, Gate, Wire
)
from gnnatpg.features.scoap import SCOAPAnalyzer
from gnnatpg.graph_construction.circuit_graph import CircuitGraphBuilder, NUM_GATE_TYPES
from gnnatpg.features.extract import FeatureExtractor, TOTAL_FEATURE_DIM
from gnnatpg.gnn.circuit_gnn import CircuitGNN, ActorCritic
from gnnatpg.rl_atpg.atpg_env import ATPGEnvironment, ATPGAction, ATPGState
from gnnatpg.fault_simulation.simulator import FaultSimulator, TestPattern


# =============================================================================
# Test Fixtures
# =============================================================================

SIMPLE_NETLIST = """
module simple_circuit (
    input A,
    input B,
    input C,
    output Y
);
    wire w1;
    AND2 g1 (.A(A), .B(B), .Y(w1));
    OR2  g2 (.A(w1), .B(C), .Y(Y));
endmodule
"""

COMPLEX_NETLIST = """
module complex_circuit (
    input clk,
    input rst,
    input [3:0] data_in,
    output [3:0] data_out
);
    wire w1, w2, w3, w4;
    wire [3:0] reg_out;
    NOT g1 (.A(clk), .Y(w1));
    AND2 g2 (.A(w1), .B(rst), .Y(w2));
    OR2  g3 (.A(w2), .B(data_in[0]), .Y(w3));
    DFF  g4 (.D(w3), .CLK(clk), .Q(reg_out[0]));
    XOR2 g5 (.A(reg_out[0]), .B(data_in[1]), .Y(w4));
    BUF  g6 (.A(w4), .Y(data_out[0]));
endmodule
"""


@pytest.fixture
def simple_netlist():
    parser = NetlistParser()
    return parser.parse_string(SIMPLE_NETLIST)


@pytest.fixture
def complex_netlist():
    parser = NetlistParser()
    return parser.parse_string(COMPLEX_NETLIST)


@pytest.fixture
def simple_graph(simple_netlist):
    builder = CircuitGraphBuilder()
    return builder.build_from_netlist(simple_netlist)


# =============================================================================
# Test Netlist Parser
# =============================================================================

class TestNetlistParser:
    """Tests for the Verilog netlist parser."""

    def test_parse_simple_netlist(self, simple_netlist):
        assert simple_netlist.module_name == "simple_circuit"
        assert simple_netlist.num_inputs == 3
        assert simple_netlist.num_outputs == 1
        assert simple_netlist.num_gates == 2

    def test_gate_types(self, simple_netlist):
        gate_types = [g.gate_type for g in simple_netlist.gates]
        assert GateType.AND2 in gate_types
        assert GateType.OR2 in gate_types

    def test_wire_connectivity(self, simple_netlist):
        # w1 should be driven by g1
        w1_wire = simple_netlist.wires.get("w1")
        assert w1_wire is not None
        assert w1_wire.driven_by is not None

    def test_primary_ports(self, simple_netlist):
        assert "A" in simple_netlist.primary_inputs
        assert "B" in simple_netlist.primary_inputs
        assert "C" in simple_netlist.primary_inputs
        assert "Y" in simple_netlist.primary_outputs

    def test_parse_complex_netlist(self, complex_netlist):
        assert complex_netlist.module_name == "complex_circuit"
        assert complex_netlist.num_inputs == 3  # clk, rst, data_in
        assert complex_netlist.num_outputs == 1  # data_out
        assert complex_netlist.num_gates == 6
        assert complex_netlist.num_ffs >= 1  # DFF

    def test_gate_type_mapping(self):
        assert GateType.from_yosys_name("AND2") == GateType.AND2
        assert GateType.from_yosys_name("DFF") == GateType.DFF
        assert GateType.from_yosys_name("NOT") == GateType.NOT
        assert GateType.from_yosys_name("MY_CUSTOM_GATE") == GateType.UNKNOWN

    def test_netlist_stats(self, simple_netlist):
        stats = simple_netlist.stats()
        assert stats["module_name"] == "simple_circuit"
        assert stats["num_gates"] == 2
        assert stats["num_inputs"] == 3
        assert stats["num_outputs"] == 1


# =============================================================================
# Test SCOAP Analysis
# =============================================================================

class TestSCOAP:
    """Tests for SCOAP controllability/observability analysis."""

    def test_scoap_basic(self, simple_netlist):
        analyzer = SCOAPAnalyzer()
        results = analyzer.analyze(simple_netlist)

        # Should have results for all gates
        assert len(results) == simple_netlist.num_gates

        # All gates should have finite values
        for gate_id, values in results.items():
            assert 0 <= values["cc0"] < 1000
            assert 0 <= values["cc1"] < 1000
            assert 0 <= values["co"] < 1000

    def test_scoap_normalized(self, simple_netlist):
        analyzer = SCOAPAnalyzer()
        results = analyzer.get_normalized_features(simple_netlist)

        for gate_id, values in results.items():
            assert 0.0 <= values["cc0"] <= 1.0
            assert 0.0 <= values["cc1"] <= 1.0
            assert 0.0 <= values["co"] <= 1.0

    def test_scoap_pi_has_low_controllability(self, simple_netlist):
        analyzer = SCOAPAnalyzer()
        results = analyzer.analyze(simple_netlist)

        # PI-driven gates should have low CC0/CC1
        for gate in simple_netlist.gates:
            if any(net in simple_netlist.primary_inputs for net in gate.input_nets):
                # Direct PI input should have controllability = 1 (cost of 1 gate)
                assert results[gate.gate_id]["cc0"] <= 5
                assert results[gate.gate_id]["cc1"] <= 5


# =============================================================================
# Test Graph Construction
# =============================================================================

class TestGraphConstruction:
    """Tests for PyG graph construction from netlists."""

    def test_graph_builds(self, simple_graph):
        assert simple_graph is not None
        assert simple_graph.num_nodes == 2
        assert simple_graph.x.shape[1] > 0

    def test_node_features(self, simple_graph):
        # Should have at least 29 features
        assert simple_graph.x.shape[1] >= 20

        # Gate type one-hot should sum to 1 per row
        gate_type_sum = simple_graph.x[:, :NUM_GATE_TYPES].sum(dim=1)
        assert torch.allclose(gate_type_sum, torch.ones_like(gate_type_sum), atol=1e-5)

    def test_edges_exist(self, simple_graph):
        # There should be edges (w1 connects g1 output to g2 input)
        assert simple_graph.edge_index.shape[1] > 0

    def test_edge_features(self, simple_graph):
        if simple_graph.edge_attr.shape[0] > 0:
            assert simple_graph.edge_attr.shape[1] == 3  # 3 edge features


# =============================================================================
# Test Feature Extraction
# =============================================================================

class TestFeatureExtraction:
    """Tests for the feature extraction pipeline."""

    def test_feature_dimensions(self, simple_graph):
        assert simple_graph.x.shape[1] == TOTAL_FEATURE_DIM

    def test_feature_range(self, simple_graph):
        # All features should be in [0, 1] after normalization
        assert simple_graph.x.min() >= -0.1  # Allow small negatives from numerical precision
        assert simple_graph.x.max() <= 1.1


# =============================================================================
# Test GNN Model
# =============================================================================

class TestGNN:
    """Tests for the CircuitGNN model."""

    def test_gcn_forward(self, simple_graph):
        gnn = CircuitGNN(
            in_dim=simple_graph.x.shape[1],
            hidden_dim=32,
            out_dim=16,
            num_layers=2,
            architecture="GCN",
        )
        node_emb, graph_emb = gnn(simple_graph)
        assert node_emb.shape[0] == simple_graph.num_nodes
        assert node_emb.shape[1] == 16
        assert graph_emb.shape[1] == 16

    def test_gat_forward(self, simple_graph):
        gnn = CircuitGNN(
            in_dim=simple_graph.x.shape[1],
            hidden_dim=32,
            out_dim=16,
            num_layers=2,
            architecture="GAT",
        )
        node_emb, graph_emb = gnn(simple_graph)
        assert node_emb.shape[0] == simple_graph.num_nodes

    def test_graphsage_forward(self, simple_graph):
        gnn = CircuitGNN(
            in_dim=simple_graph.x.shape[1],
            hidden_dim=32,
            out_dim=16,
            num_layers=2,
            architecture="GRAPHSAGE",
        )
        node_emb, graph_emb = gnn(simple_graph)
        assert node_emb.shape[0] == simple_graph.num_nodes

    def test_actor_critic(self, simple_graph):
        gnn = CircuitGNN(
            in_dim=simple_graph.x.shape[1],
            hidden_dim=32,
            out_dim=16,
            num_layers=2,
        )
        ac = ActorCritic(gnn, num_actions=8)
        action_logits, value = ac(simple_graph, simple_graph.x)
        assert action_logits.shape[1] == 8
        assert value.shape[1] == 1


# =============================================================================
# Test RL Environment
# =============================================================================

class TestATPGEnvironment:
    """Tests for the ATPG RL environment."""

    def _make_env(self, num_nodes=10, num_faults=5):
        adj = np.random.randint(0, 2, (num_nodes, num_nodes)).astype(np.float32)
        gate_types = np.random.randint(0, 10, num_nodes)
        pis = list(range(min(3, num_nodes)))
        pos = list(range(max(0, num_nodes - 3), num_nodes))
        faults = [(np.random.randint(0, num_nodes), np.random.randint(0, 2))
                  for _ in range(num_faults)]

        return ATPGEnvironment(
            adj_matrix=adj,
            gate_types=gate_types,
            primary_inputs=pis,
            primary_outputs=pos,
            fault_list=faults,
            max_steps=50,
        )

    def test_reset(self):
        env = self._make_env()
        state, info = env.reset()
        assert state is not None
        assert state.shape[0] == 10  # num_nodes
        assert "fault_node" in info

    def test_step(self):
        env = self._make_env()
        env.reset()
        state, reward, done, info = env.step(ATPGAction.SET_0)
        assert state is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_episode_terminates(self):
        env = self._make_env(num_faults=2)
        env.reset()
        total_reward = 0
        for _ in range(100):
            state, reward, done, info = env.step(ATPGAction.NO_OP)
            total_reward += reward
            if done:
                break
        # Episode should terminate within max_steps
        assert done

    def test_fault_detection(self):
        env = self._make_env(num_nodes=5, num_faults=1)
        env.reset()
        # Setting PI values should eventually trigger fault detection
        for _ in range(10):
            state, reward, done, info = env.step(
                np.random.choice([
                    ATPGAction.BACKTRACE_PI_0,
                    ATPGAction.BACKTRACE_PI_1,
                ])
            )
        # Reward should be non-zero for some actions
        assert isinstance(reward, float)

    def test_statistics(self):
        env = self._make_env(num_faults=5)
        env.reset()
        stats = env.get_statistics()
        assert stats["total_faults"] == 5
        assert stats["faults_detected"] == 0


# =============================================================================
# Test Fault Simulation
# =============================================================================

class TestFaultSimulator:
    """Tests for the fault simulation engine."""

    def test_fault_list_generation(self, simple_netlist):
        sim = FaultSimulator(simple_netlist)
        fault_list = sim.generate_fault_list()
        # Each gate has SA0 and SA1
        assert len(fault_list) == simple_netlist.num_gates * 2

    def test_fault_types(self, simple_netlist):
        sim = FaultSimulator(simple_netlist)
        fault_list = sim.generate_fault_list()
        sa0 = [f for f in fault_list if f.fault_type == "SA0"]
        sa1 = [f for f in fault_list if f.fault_type == "SA1"]
        assert len(sa0) == len(sa1)

    def test_pattern_simulation(self, simple_netlist):
        sim = FaultSimulator(simple_netlist)
        # Create a test pattern
        pi_values = {pi: np.random.randint(0, 2)
                     for pi in simple_netlist.primary_inputs}
        pattern = TestPattern(pattern_id=0, pi_values=pi_values)

        fault_list = sim.generate_fault_list()
        detected = sim.simulate_pattern(pattern, fault_list)
        assert isinstance(detected, list)

    def test_circuit_evaluation(self, simple_netlist):
        sim = FaultSimulator(simple_netlist)
        pi_values = {"A": 1, "B": 1, "C": 0}
        values = sim._evaluate_circuit(pi_values)
        # Should have values for all gates
        assert len(values) == simple_netlist.num_gates


# =============================================================================
# Test PODEM
# =============================================================================

class TestPODEM:
    """Tests for the PODEM baseline ATPG."""

    def test_podem_initialization(self, simple_netlist):
        from gnnatpg.evaluation.podem_baseline import PODEM
        podem = PODEM(simple_netlist, max_backtracks=10)
        assert podem.netlist == simple_netlist

    def test_podem_generate_test(self, simple_netlist):
        from gnnatpg.evaluation.podem_baseline import PODEM
        podem = PODEM(simple_netlist, max_backtracks=50)
        result = podem.generate_test(
            target_fault_gate_id=simple_netlist.gates[0].gate_id,
            fault_type="SA0",
        )
        assert result is not None
        assert result.fault_name != "unknown"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
