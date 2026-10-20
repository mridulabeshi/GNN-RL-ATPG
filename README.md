# GNN & RL-Based ATPG for Open-Source RISC-V Processor Cores

> **Project I (2026-2027)** — School of Electronics Engineering
> Members: Somesh TG, Shreya Ranjitha, Mridula B | Guide: Dr. Ananiah Durai

---

## 📋 Project Overview

This project develops an **ML-guided Automatic Test Pattern Generation (ATPG)** framework for open-source RISC-V processor cores. It combines:

- **Graph Neural Networks (GNNs)** for learning circuit-structural representations
- **Reinforcement Learning (RL)** for guiding ATPG search decisions
- **Open-source RISC-V cores** (Ibex, PicoRV32) as realistic test targets

### Key Objectives
1. Synthesize RISC-V cores into gate-level netlists
2. Represent circuits as graphs with extracted features (SCOAP, gate type, etc.)
3. Train a GNN encoder to learn structural circuit representations
4. Train an RL agent (PPO) to make intelligent ATPG decisions
5. Compare against conventional baselines (PODEM, random)
6. Evaluate cross-core generalization

---

## 🏗️ Project Structure

```
gnnatpg/
├── config.yaml                    # Project configuration
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── README.md                      # This file
│
├── synthesis/                     # Phase 1: RTL → Gate-level netlist
│   ├── synthesize.py              # Yosys synthesis wrapper
│   └── synth.ys                   # Default Yosys script
│
├── graph_construction/            # Phase 1: Netlist → Graph
│   ├── netlist_parser.py          # Verilog netlist parser
│   └── circuit_graph.py           # PyG graph construction
│
├── features/                      # Phase 1: Feature extraction
│   ├── scoap.py                   # SCOAP controllability/observability
│   └── extract.py                 # Full feature extraction pipeline
│
├── gnn/                           # Phase 2: GNN model
│   └── circuit_gnn.py             # GCN/GAT/GraphSAGE encoder
│
├── rl_atpg/                       # Phase 2: RL agent
│   ├── atpg_env.py                # ATPG environment (MDP)
│   └── ppo_agent.py               # PPO agent with GNN backbone
│
├── fault_simulation/              # Evaluation: Fault simulation
│   └── simulator.py               # Single-fault simulator
│
├── evaluation/                    # Phase 3: Evaluation
│   ├── podem_baseline.py          # PODEM conventional ATPG
│   └── evaluate.py                # Comparison metrics & plots
│
├── scripts/                       # Entry points
│   └── main_pipeline.py           # Full pipeline orchestrator
│
├── data/                          # Generated data (gitignored)
│   ├── netlists/                  # Gate-level Verilog netlists
│   ├── fault_list/                # Generated fault lists
│   ├── features/                  # Extracted features (.pt)
│   └── models/                    # Trained model checkpoints
│       ├── checkpoints/
│       └── logs/
│
└── tests/                         # Unit tests
    └── test_components.py
```

---

## 🚀 Step-by-Step Build Instructions

### Prerequisites

#### System Requirements
- **OS:** Ubuntu 20.04+ / macOS 12+ / Windows with WSL2
- **RAM:** ≥16 GB recommended (RISC-V core synthesis is memory-intensive)
- **GPU:** Optional but recommended (NVIDIA GPU with CUDA for training)
- **Storage:** ≥10 GB free space

#### Software Dependencies

```bash
# 1. Install Yosys (RTL synthesis)
# Ubuntu/Debian:
sudo apt update

sudo apt install yosys

# macOS:
brew install yosys

# Or build from source:
# git clone https://github.com/YosysHQ/yosys
# cd yosys && make config-gcc && make && sudo make install

# 2. Install Python 3.9+
sudo apt install python3.9 python3.9-venv python3-pip

# 3. Install Git
sudo apt install git

# 4. Verify installations
yosys --version
python3 --version
git --version
```

---

### Phase 1: Project Setup + Synthesis + Feature Extraction

#### Step 1.1: Clone and Set Up Project

```bash
# Clone repository
git clone https://github.com/your-org/gnnatpg.git
cd gnnatpg

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch (choose one based on your system)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.replace('+cu118',''))").html

# Install package in development mode
pip install -e .
```

#### Step 1.2: Configure Project

```bash
# Review and edit configuration
cat config.yaml

# Key settings to check:
# - riscv_cores: Which cores to synthesize
# - synthesis.target: "generic" for initial experiments
# - gnn.architecture: "GCN" recommended to start
# - rl.algorithm: "PPO" (default)
# - training.device: "cuda" if GPU available, "cpu" otherwise
```

#### Step 1.3: Synthesize RISC-V Cores

```bash
# Option A: Run full pipeline (Phase 1 only)
python -m gnnatpg.scripts.main_pipeline --phase 1

# Option B: Run synthesis independently
python -c "
from gnnatpg.synthesis.synthesize import synthesize_all
results = synthesize_all()
for core, result in results.items():
    print(f'{core}: {result}')
"

# Verify output
ls -la data/netlists/
# Should see: ibex.v, picorv32.v (gate-level netlists)
```

**What happens here:**
1. Yosys clones the RISC-V core repositories (Ibex, PicoRV32)
2. Parses all RTL files (Verilog/SystemVerilog)
3. Runs synthesis: `proc → flatten → opt → techmap → write_verilog`
4. Outputs technology-mapped gate-level netlists

#### Step 1.4: Parse Netlists and Extract Features

```bash
# The pipeline does this automatically in Phase 1
# To run feature extraction manually:
python -c "
from gnnatpg.features.extract import extract_all_features
features = extract_all_features({
    'ibex': 'data/netlists/ibex.v',
    'picorv32': 'data/netlists/picorv32.v',
})
for core, graph in features.items():
    print(f'{core}: {graph.num_nodes} nodes, {graph.x.shape[1]} features')
"

# Verify output
ls -la data/features/
# Should see: ibex_features.pt, picorv32_features.pt

ls -la data/fault_list/
# Should see: ibex_faults.json, picorv32_faults.json
```

**Feature vector per gate node (29 dimensions):**
| Index | Feature | Description |
|-------|---------|-------------|
| 0-19 | gate_type | One-hot encoding of gate type (AND, OR, NOT, DFF, etc.) |
| 20 | fanout | Number of gates driven by this gate |
| 21 | depth | Topological depth from primary inputs |
| 22 | is_dff | Whether gate is a flip-flop |
| 23 | is_pi | Whether gate reads from primary input |
| 24 | is_po | Whether gate drives primary output |
| 25 | num_inputs | Number of input pins |
| 26 | scoap_cc0 | SCOAP controllability-0 (normalized) |
| 27 | scoap_cc1 | SCOAP controllability-1 (normalized) |
| 28 | scoap_co | SCOAP observability (normalized) |

---

### Phase 2: GNN + RL Training

#### Step 2.1: Train GNN + RL Agent

```bash
# Run Phase 2 (requires Phase 1 output)
python -m gnnatpg.scripts.main_pipeline --phase 2

# Or run all phases
python -m gnnatpg.scripts.main_pipeline --phase all
```

**Training process:**
1. **GNN Encoding:** The GNN learns to represent circuit structure through message passing between gates
2. **RL Environment:** Models ATPG as an MDP — state = circuit values + fault info, action = PI assignment, reward = fault detection
3. **PPO Training:** The RL agent learns to make ATPG decisions that maximize fault detection while minimizing backtracking

#### Step 2.2: Monitor Training

```bash
# View training logs
tail -f gnnatpg.log

# TensorBoard (if configured)
tensorboard --logdir data/models/logs
```

#### Step 2.3: Check Trained Model

```bash
# Model checkpoints saved at:
ls -la data/models/checkpoints/
# Should see: best_model.pt, episode_*.pt
```

---

### Phase 3: Evaluation and Comparison

#### Step 3.1: Run Full Evaluation

```bash
# Run Phase 3 (requires Phase 1 + 2 output)
python -m gnnatpg.scripts.main_pipeline --phase 3
```

#### Step 3.2: View Results

```bash
# Check evaluation output
ls -la data/evaluation_plots/
# Should see:
#   comparison_table.csv
#   comparison_table.md
#   fault_coverage_comparison.png
#   runtime_comparison.png
#   backtrack_comparison.png
#   pattern_count_comparison.png

# View comparison table
cat data/evaluation_plots/comparison_table.md
```

**Expected comparison table format:**
| Method | Core | Fault Coverage | Pattern Count | Runtime (s) | Avg Backtracks |
|--------|------|---------------|---------------|-------------|----------------|
| PODEM | ibex | ~65% | ~500 | ~120s | ~45 |
| Random | ibex | ~35% | ~200 | ~10s | 0 |
| GNN+RL | ibex | ~85% | ~300 | ~60s | ~15 |
| PODEM | picorv32 | ~70% | ~300 | ~80s | ~35 |
| GNN+RL | picorv32 | ~88% | ~200 | ~40s | ~10 |

---

### Cross-Core Generalization (Phase 3 extension)

```bash
# Train on Ibex, test on PicoRV32
# The evaluation script handles this automatically
# See cross_core_generalization_report in logs
```

---

## 🔬 Architecture Details

### GNN Circuit Encoder

```
Input Features (29-dim)
    ↓
GCN/GAT Layer 1 (29 → 128)
    ↓ BatchNorm + ReLU
GCN/GAT Layer 2 (128 → 128)
    ↓ BatchNorm + ReLU
GCN/GAT Layer 3 (128 → 128)
    ↓ BatchNorm + ReLU
GCN/GAT Layer 4 (128 → 64)
    ↓
Node Embeddings (64-dim)
    ↓
Global Pooling (mean + max + add)
    ↓
Graph Readout (192 → 64)
    ↓
Graph Embedding (64-dim)
```

### RL ATPG Agent (PPO)

```
State:
  - Circuit graph embeddings (from GNN)
  - Current PI assignments
  - Fault site information
  - D-frontier / J-frontier status

Action Space (8 actions):
  - SET_0 / SET_1: Set a gate output
  - BACKTRACE_PI_0 / BACKTRACE_PI_1: Assign primary input
  - JUSTIFY_0 / JUSTIFY_1: Justify gate value
  - ADVANCE: Move to next fault
  - NO_OP: No operation

Reward Function:
  +10.0  Fault detected successfully
  +0.1   No conflict (valid assignment)
  -1.0   Conflict detected
  -0.5   Backtrack required
  -5.0   Timeout (too many steps)
```

---

## 🧪 Running Tests

```bash
# Run unit tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_components.py -v -k "test_netlist_parser"
```

---

## 📊 Key Results Metrics

| Metric | Description |
|--------|-------------|
| **Fault Coverage** | % of faults detected by generated patterns |
| **Pattern Count** | Number of test patterns generated |
| **Runtime** | Total ATPG execution time |
| **Backtracks** | Number of search backtracks (lower = better) |
| **Hard-to-Detect Ratio** | % of faults requiring extensive search |

---

## 📚 References

1. SmartATPG (DAC 2024) — GCN + RL for ATPG
2. InF-ATPG (2025) — FFR-driven GNN-RL ATPG
3. FPGNN-ATPG (GLSVLSI 2023) — Fault-parallel GNN ATPG
4. PODEM — Goel, IEEE Trans. Computers, 1981
5. SCOAP — Goldstein & Thigpen, DAC 1980

---

## 🛠️ Troubleshooting

### Common Issues

**Yosys not found:**
```bash
# Ensure Yosys is in PATH
which yosys
# If not found, reinstall: sudo apt install yosys
```

**Out of memory during synthesis:**
```bash
# Try synthesizing cores one at a time
# Or use a smaller core configuration
# In config.yaml, try only picorv32 first
```

**PyTorch Geometric installation issues:**
```bash
# Install with specific PyTorch version matching
pip install torch-geometric -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
```

**Import errors:**
```bash
# Ensure package is installed in dev mode
pip install -e .
# Or add project root to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
```

---

## 📄 License

Academic use only — School of Electronics Engineering, Project I (2026-2027)
