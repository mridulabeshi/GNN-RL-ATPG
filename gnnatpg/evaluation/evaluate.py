"""
evaluate.py - Evaluation and Comparison Module

Compares ML-guided (GNN+RL) ATPG against conventional baselines (PODEM, random)
across multiple metrics:
    - Fault coverage
    - Test pattern count
    - Runtime
    - Backtracking count
    - Hard-to-detect fault ratio
    - Cross-core generalization
"""

import logging
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for ATPG evaluation metrics."""
    method: str
    core_name: str
    fault_coverage: float = 0.0
    pattern_count: int = 0
    total_runtime: float = 0.0
    avg_runtime_per_fault: float = 0.0
    total_backtracks: int = 0
    avg_backtracks: float = 0.0
    hard_to_detect_count: int = 0
    hard_to_detect_ratio: float = 0.0
    detected_faults: int = 0
    total_faults: int = 0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "core": self.core_name,
            "fault_coverage": f"{self.fault_coverage:.4f}",
            "pattern_count": self.pattern_count,
            "total_runtime_s": f"{self.total_runtime:.2f}",
            "avg_runtime_s": f"{self.avg_runtime_per_fault:.4f}",
            "total_backtracks": self.total_backtracks,
            "avg_backtracks": f"{self.avg_backtracks:.1f}",
            "hard_to_detect": self.hard_to_detect_count,
            "hard_to_detect_ratio": f"{self.hard_to_detect_ratio:.4f}",
        }


class Evaluator:
    """
    Comprehensive evaluator for ATPG methods.
    
    Compares:
        1. GNN+RL ATPG (proposed method)
        2. PODEM (conventional baseline)
        3. Random ATPG (simple baseline)
    """

    def __init__(self, config: dict):
        self.config = config
        self.eval_config = config.get("evaluation", {})
        self.results: list[EvaluationMetrics] = []

    def evaluate_method(
        self,
        method_name: str,
        core_name: str,
        detected_faults: int,
        total_faults: int,
        pattern_count: int,
        runtime: float,
        backtracks: int,
    ) -> EvaluationMetrics:
        """
        Record evaluation metrics for a method/core combination.
        """
        fc = detected_faults / total_faults if total_faults > 0 else 0.0
        avg_rt = runtime / max(total_faults, 1)
        avg_bt = backtracks / max(total_faults, 1)

        # Hard-to-detect: faults requiring >100 backtracks (simplified)
        htd = int(total_faults * (1 - fc) * 0.5)  # Estimate
        htd_ratio = htd / total_faults if total_faults > 0 else 0.0

        metrics = EvaluationMetrics(
            method=method_name,
            core_name=core_name,
            fault_coverage=fc,
            pattern_count=pattern_count,
            total_runtime=runtime,
            avg_runtime_per_fault=avg_rt,
            total_backtracks=backtracks,
            avg_backtracks=avg_bt,
            hard_to_detect_count=htd,
            hard_to_detect_ratio=htd_ratio,
            detected_faults=detected_faults,
            total_faults=total_faults,
        )

        self.results.append(metrics)
        logger.info(
            f"[{method_name}] on {core_name}: "
            f"FC={fc:.2%}, patterns={pattern_count}, "
            f"runtime={runtime:.2f}s, backtracks={backtracks}"
        )
        return metrics

    def compare_methods(self) -> pd.DataFrame:
        """
        Generate comparison table of all evaluated methods.
        
        Returns:
            DataFrame with method comparison
        """
        if not self.results:
            logger.warning("No evaluation results to compare")
            return pd.DataFrame()

        records = [r.to_dict() for r in self.results]
        df = pd.DataFrame(records)

        # Pivot for easier comparison
        logger.info("=== Method Comparison ===")
        logger.info(f"\n{df.to_string(index=False)}")

        return df

    def generate_report(self, output_dir: str = "data/evaluation_plots"):
        """Generate evaluation report with plots and tables."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.results:
            logger.warning("No results to report")
            return

        df = self.compare_methods()

        # Save table
        df.to_csv(output_dir / "comparison_table.csv", index=False)
        df.to_markdown(output_dir / "comparison_table.md", index=False)

        # Generate plots
        self._plot_fault_coverage(df, output_dir)
        self._plot_runtime_comparison(df, output_dir)
        self._plot_backtrack_comparison(df, output_dir)
        self._plot_pattern_count(df, output_dir)

        logger.info(f"Report generated in {output_dir}")

    def _plot_fault_coverage(self, df: pd.DataFrame, output_dir: Path):
        """Plot fault coverage comparison bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        cores = df["core"].unique()
        methods = df["method"].unique()
        x = np.arange(len(cores))
        width = 0.25

        for i, method in enumerate(methods):
            method_data = df[df["method"] == method]
            fc_values = []
            for core in cores:
                core_data = method_data[method_data["core"] == core]
                fc = float(core_data["fault_coverage"].iloc[0]) if len(core_data) > 0 else 0
                fc_values.append(fc)

            bars = ax.bar(x + i * width, fc_values, width, label=method)
            # Add value labels
            for bar, val in zip(bars, fc_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xlabel("RISC-V Core")
        ax.set_ylabel("Fault Coverage")
        ax.set_title("Fault Coverage Comparison Across Methods and Cores")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(cores)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "fault_coverage_comparison.png", dpi=150)
        plt.close()

    def _plot_runtime_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Plot runtime comparison bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        cores = df["core"].unique()
        methods = df["method"].unique()
        x = np.arange(len(cores))
        width = 0.25

        for i, method in enumerate(methods):
            method_data = df[df["method"] == method]
            rt_values = []
            for core in cores:
                core_data = method_data[method_data["core"] == core]
                rt = float(core_data["total_runtime_s"].iloc[0]) if len(core_data) > 0 else 0
                rt_values.append(rt)

            ax.bar(x + i * width, rt_values, width, label=method)

        ax.set_xlabel("RISC-V Core")
        ax.set_ylabel("Total Runtime (seconds)")
        ax.set_title("Runtime Comparison Across Methods")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(cores)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "runtime_comparison.png", dpi=150)
        plt.close()

    def _plot_backtrack_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Plot average backtrack comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        cores = df["core"].unique()
        methods = df["method"].unique()
        x = np.arange(len(cores))
        width = 0.25

        for i, method in enumerate(methods):
            method_data = df[df["method"] == method]
            bt_values = []
            for core in cores:
                core_data = method_data[method_data["core"] == core]
                bt = float(core_data["avg_backtracks"].iloc[0]) if len(core_data) > 0 else 0
                bt_values.append(bt)

            ax.bar(x + i * width, bt_values, width, label=method)

        ax.set_xlabel("RISC-V Core")
        ax.set_ylabel("Average Backtracks per Fault")
        ax.set_title("Backtracking Comparison (Lower is Better)")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(cores)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "backtrack_comparison.png", dpi=150)
        plt.close()

    def _plot_pattern_count(self, df: pd.DataFrame, output_dir: Path):
        """Plot test pattern count comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        cores = df["core"].unique()
        methods = df["method"].unique()
        x = np.arange(len(cores))
        width = 0.25

        for i, method in enumerate(methods):
            method_data = df[df["method"] == method]
            pc_values = []
            for core in cores:
                core_data = method_data[method_data["core"] == core]
                pc = int(core_data["pattern_count"].iloc[0]) if len(core_data) > 0 else 0
                pc_values.append(pc)

            ax.bar(x + i * width, pc_values, width, label=method)

        ax.set_xlabel("RISC-V Core")
        ax.set_ylabel("Test Pattern Count")
        ax.set_title("Pattern Count Comparison (Lower is Better)")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(cores)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "pattern_count_comparison.png", dpi=150)
        plt.close()

    def cross_core_generalization_report(
        self,
        train_core: str,
        test_cores: dict[str, float],
    ) -> str:
        """
        Report cross-core generalization results.
        
        Args:
            train_core: Core used for training
            test_cores: Dict mapping test_core -> fault_coverage
        """
        lines = [
            f"=== Cross-Core Generalization Report ===",
            f"Training Core: {train_core}",
            "",
        ]
        for core, coverage in test_cores.items():
            lines.append(f"  Tested on {core}: {coverage:.2%} fault coverage")

        report = "\n".join(lines)
        logger.info(report)
        return report
