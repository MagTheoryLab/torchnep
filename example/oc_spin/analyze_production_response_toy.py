"""Aggregate and plot the sealed production response-toy A/B."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("generator_nrmse", "j_nrmse", "d_nrmse")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "production_response_toy.png"
    variants = tuple(
        variant for variant in ("absolute", "curve", "joint")
        if (args.root / variant).exists()
    )
    runs = {variant: [] for variant in variants}
    for variant in runs:
        for path in sorted((args.root / variant).glob("seed*/result.json")):
            runs[variant].append(json.loads(path.read_text()))
        if not runs[variant]:
            raise FileNotFoundError(f"no completed {variant} runs under {args.root}")

    summary = {}
    for variant, entries in runs.items():
        summary[variant] = {
            metric: {
                "mean": float(np.mean([e["metrics"][metric] for e in entries])),
                "std": float(np.std([e["metrics"][metric] for e in entries])),
                "values": [e["metrics"][metric] for e in entries],
            } for metric in METRICS
        }
    for variant in variants[1:]:
        summary[f"paired_{variant}_over_absolute"] = {
            metric: float(np.mean([
                candidate["metrics"][metric] / absolute["metrics"][metric]
                for absolute, candidate in zip(
                    runs["absolute"], runs[variant])
            ])) for metric in METRICS
        }
    health_entries = [
        entry["health"] for entries in runs.values() for entry in entries
        if "health" in entry
    ]
    if health_entries:
        derivative_names = ("force", "mforce", "isotropic_strain")
        health_summary = {
            "runs": len(health_entries),
            "descriptor_scaled_full_rank_all": all(
                health["descriptor"]["training"]["scaled"]
                ["full_column_rank"]
                for health in health_entries
            ),
            "descriptor_scaled_condition": [
                health["descriptor"]["training"]["scaled"]
                ["condition_number"]
                for health in health_entries
            ],
            "blind_descriptor_scaled_full_rank_all": all(
                health["descriptor"]["blind"]["scaled"]["full_column_rank"]
                for health in health_entries
            ),
            "position_jacobian_full_rank_all": all(
                health["derivatives"]["descriptor_jacobian"]["position"]
                ["full_column_rank"] for health in health_entries
            ),
            "spin_jacobian_full_rank_all": all(
                health["derivatives"]["descriptor_jacobian"]["spin"]
                ["full_column_rank"] for health in health_entries
            ),
            "zero_spin_exact_all": all(
                health["derivatives"]["spin_limits"]["zero"]["exact_zero"]
                for health in health_entries
            ),
            "near_zero_spin_finite_all": all(
                health["derivatives"]["spin_limits"]["near_zero"]["finite"]
                for health in health_entries
            ),
            "training_log_finite_all": all(
                health["training_log"]["all_observed_finite"]
                for health in health_entries
            ),
        }
        health_summary["worst_relative_derivative_error"] = {
            name: max(
                health["derivatives"]["energy_derivative_closure"][name]
                ["relative_rms"] for health in health_entries
            ) for name in derivative_names
        }
        health_summary["minimum_sector_derivative_rms"] = {
            f"{order}_{variable}": min(
                health["derivatives"]["descriptor_jacobian"]["sector_rms"]
                [f"O{order}_increment"][f"{variable}_rms"]
                for health in health_entries
            )
            for order in (1, 2, 3)
            for variable in ("position", "spin", "isotropic_strain")
        }
        summary["health"] = health_summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    colors = {"absolute": "#777777", "curve": "#1769aa",
              "joint": "#d97706"}
    x = np.arange(len(METRICS))
    width = 0.24 if len(runs) == 3 else 0.34
    offsets = (np.arange(len(runs)) - (len(runs) - 1) / 2) * width
    for offset, variant in zip(offsets, runs):
        means = [summary[variant][m]["mean"] for m in METRICS]
        stds = [summary[variant][m]["std"] for m in METRICS]
        axes[0, 0].bar(x + offset, means, width, yerr=stds,
                       label=variant, color=colors[variant], capsize=3)
    axes[0, 0].set_xticks(x, ["generator", "J-like", "D-like"])
    axes[0, 0].set_ylabel("blind NRMSE")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("Sealed blind response")

    example = {variant: runs[variant][0]["metrics"] for variant in runs}
    for variant in runs:
        curve = example[variant]["curves"][0]
        axes[0, 1].plot(curve["theta"], curve["prediction"], "o-",
                        color=colors[variant], label=variant)
    target = example["absolute"]["curves"][0]
    axes[0, 1].plot(target["theta"], target["target"], "k--", label="teacher")
    axes[0, 1].set_xlabel(r"rotation $\theta$")
    axes[0, 1].set_ylabel(r"$\sum_i MF_i\cdot dS_i/d\theta$")
    axes[0, 1].set_title("Unseen environment curve")
    axes[0, 1].legend(frameon=False)

    for axis, name, label in (
        (axes[1, 0], "j", "J-like coefficient"),
        (axes[1, 1], "d", "D-like coefficient"),
    ):
        values = []
        for variant in runs:
            metric = example[variant]
            target_values = np.asarray(metric[f"{name}_target"])
            prediction = np.asarray(metric[f"{name}_prediction"])
            axis.scatter(target_values, prediction, label=variant,
                         color=colors[variant], alpha=0.8)
            values.extend(target_values.tolist() + prediction.tolist())
        lower, upper = min(values), max(values)
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        axis.set_xlabel("teacher")
        axis.set_ylabel("prediction")
        axis.set_title(label)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
