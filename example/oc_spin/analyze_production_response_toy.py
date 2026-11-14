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
    parser.add_argument(
        "--compact-plot", action="store_true",
        help="Plot four representative arms while retaining all JSON data.",
    )
    args = parser.parse_args()
    output = args.output or args.root / "production_response_toy.png"
    paths = sorted(args.root.glob("*/seed*/result.json"))
    if not paths:
        raise FileNotFoundError(f"no completed runs under {args.root}")
    runs = {}
    recipes = {}
    for path in paths:
        entry = json.loads(path.read_text())
        recipe = entry["recipe"]
        label = recipe["label"]
        recipes[label] = recipe
        runs.setdefault(label, []).append(entry)
    labels = tuple(sorted(
        runs,
        key=lambda label: (
            recipes[label]["variant"] != "standard",
            recipes[label]["stage2_lambda_m"],
            recipes[label]["lambda_m"],
            recipes[label]["generator_weight"],
        ),
    ))
    runs = {label: sorted(runs[label], key=lambda entry: entry["seed"])
            for label in labels}
    standard_labels = [
        label for label in labels if recipes[label]["variant"] == "standard"
    ]
    if not standard_labels:
        raise ValueError("at least one standard recipe is required")
    reference = min(
        standard_labels,
        key=lambda label: (
            abs(recipes[label]["stage2_lambda_m"] - 1.0),
            abs(recipes[label]["lambda_m"] - 0.1),
        ),
    )

    summary = {}
    summary["reference"] = reference
    for label, entries in runs.items():
        summary[label] = {"recipe": recipes[label]}
        summary[label].update({
            metric: {
                "mean": float(np.mean([e["metrics"][metric] for e in entries])),
                "std": float(np.std([e["metrics"][metric] for e in entries])),
                "values": [e["metrics"][metric] for e in entries],
            } for metric in METRICS
        })
        if all("generalization" in entry["metrics"] for entry in entries):
            summary[label]["generalization"] = {
                split: {
                    metric: {
                        "mean": float(np.mean([
                            entry["metrics"]["generalization"][split][metric]
                            for entry in entries
                        ])),
                        "std": float(np.std([
                            entry["metrics"]["generalization"][split][metric]
                            for entry in entries
                        ])),
                    }
                    for metric in (
                        "energy_rmse_ev_atom", "mforce_rmse_ev_mub",
                        "mforce_nrmse",
                    )
                }
                for split in ("all", "ordinary", "response")
            }
    reference_by_seed = {entry["seed"]: entry for entry in runs[reference]}
    for label in labels:
        if label == reference:
            continue
        candidate_by_seed = {entry["seed"]: entry for entry in runs[label]}
        shared = sorted(set(reference_by_seed) & set(candidate_by_seed))
        if not shared:
            continue
        summary[f"paired::{label}/reference"] = {
            "reference": reference,
            "seeds": shared,
            **{
                metric: float(np.mean([
                    candidate_by_seed[seed]["metrics"][metric]
                    / reference_by_seed[seed]["metrics"][metric]
                    for seed in shared
                ]))
                for metric in METRICS
            },
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
            "zero_spin_mforce_exact_all": all(
                health["derivatives"]["spin_limits"]["zero"]["mforce_exact_zero"]
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
    summary["response_reweighting"] = {
        label: {
            "strict_rank_gain": sorted({
                entry["health"]["response_reweighting"]
                ["strict_observation_rank_gain"]
                for entry in entries
            }),
            "generator_operator_rank": sorted({
                entry["health"]["response_reweighting"]
                ["generator_operator"]["effective_rank"]
                for entry in entries
            }),
            "aligned_curvature_gain_min": [
                entry["health"]["response_reweighting"]
                ["aligned_curvature_gain_min"]
                for entry in entries
            ],
            "aligned_curvature_gain_max": [
                entry["health"]["response_reweighting"]
                ["aligned_curvature_gain_max"]
                for entry in entries
            ],
        }
        for label, entries in runs.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    palette = plt.get_cmap("turbo")
    denominator = max(1, len(labels) - 1)
    colors = {
        label: palette(index / denominator)
        for index, label in enumerate(labels)
    }
    pretty = {
        label: (
            ("standard" if recipes[label]["variant"] == "standard" else "generator")
            + rf"  $\lambda_m$ {recipes[label]['lambda_m']:g}"
            + rf"$\rightarrow${recipes[label]['stage2_lambda_m']:g}"
            + (
                rf", $\lambda_R$={recipes[label]['generator_weight']:g}"
                if recipes[label]["variant"] == "generator" else ""
            )
        )
        for label in labels
    }
    plot_labels = labels
    if args.compact_plot and len(labels) > 4:
        generator_labels = [
            label for label in labels
            if recipes[label]["variant"] == "generator"
        ]
        best_standard = min(
            standard_labels,
            key=lambda label: summary[label]["generator_nrmse"]["mean"],
        )

        def nearest_generator(stage2_m, generator_weight):
            return min(
                generator_labels,
                key=lambda label: (
                    abs(recipes[label]["stage2_lambda_m"] - stage2_m),
                    abs(recipes[label]["generator_weight"] - generator_weight),
                ),
            )

        plot_labels = tuple(dict.fromkeys((
            reference,
            best_standard,
            nearest_generator(1.0, 0.1),
            nearest_generator(10.0, 0.1),
        )))
    x = np.arange(len(METRICS))
    width = min(0.34, 0.82 / len(plot_labels))
    offsets = (
        np.arange(len(plot_labels)) - (len(plot_labels) - 1) / 2
    ) * width
    for offset, label in zip(offsets, plot_labels):
        means = [summary[label][m]["mean"] for m in METRICS]
        stds = [summary[label][m]["std"] for m in METRICS]
        axes[0, 0].bar(x + offset, means, width, yerr=stds,
                       label=pretty[label], color=colors[label], capsize=3)
    axes[0, 0].set_xticks(x, ["generator", "J-like", "D-like"])
    axes[0, 0].set_ylabel("blind NRMSE")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("Sealed blind response")

    example = {label: runs[label][0]["metrics"] for label in plot_labels}
    for label in plot_labels:
        curve = example[label]["curves"][0]
        axes[0, 1].plot(curve["theta"], curve["prediction"], "o-",
                        color=colors[label], label=pretty[label])
    target = example[reference]["curves"][0]
    axes[0, 1].plot(target["theta"], target["target"], "k--", label="teacher")
    axes[0, 1].set_xlabel(r"rotation $\theta$")
    axes[0, 1].set_ylabel(r"$\sum_i MF_i\cdot dS_i/d\theta$")
    axes[0, 1].set_title("Unseen environment curve")
    axes[0, 1].legend(frameon=False)

    for axis, name, title in (
        (axes[1, 0], "j", "J-like coefficient"),
        (axes[1, 1], "d", "D-like coefficient"),
    ):
        values = []
        for recipe_label in plot_labels:
            metric = example[recipe_label]
            target_values = np.asarray(metric[f"{name}_target"])
            prediction = np.asarray(metric[f"{name}_prediction"])
            axis.scatter(
                target_values, prediction, label=pretty[recipe_label],
                color=colors[recipe_label], alpha=0.8,
            )
            values.extend(target_values.tolist() + prediction.tolist())
        lower, upper = min(values), max(values)
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        axis.set_xlabel("teacher")
        axis.set_ylabel("prediction")
        axis.set_title(title)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)

    if all("generalization" in summary[label] for label in labels):
        tradeoff, tradeoff_axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        x_metric = "mforce_rmse_ev_mub"
        for axis, metric, title in zip(
            tradeoff_axes,
            METRICS,
            ("Projected response", "J-like", "D-like"),
        ):
            for label in labels:
                x_stats = summary[label]["generalization"]["ordinary"][
                    x_metric
                ]
                y_stats = summary[label][metric]
                axis.errorbar(
                    x_stats["mean"], y_stats["mean"],
                    xerr=x_stats["std"], yerr=y_stats["std"],
                    marker=("o" if recipes[label]["variant"] == "generator"
                            else "s"),
                    color=colors[label], capsize=3, linestyle="none",
                    label=pretty[label],
                )
            axis.set_xlabel("ordinary-slice MF RMSE (eV/$\mu_B$)")
            axis.set_ylabel("blind NRMSE")
            axis.set_title(title)
            axis.grid(alpha=0.2)
        tradeoff_axes[0].legend(frameon=False, fontsize=8)
        tradeoff.tight_layout()
        tradeoff.savefig(
            output.with_name(output.stem + "_tradeoff" + output.suffix),
            dpi=180,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
