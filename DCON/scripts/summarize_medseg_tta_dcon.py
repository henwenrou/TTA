#!/usr/bin/env python3
"""Summarize DCON-adapted MedSeg-TTA runs.

Only the target-domain "Overall mean dice by sample" metric is used. DCON's
optional source-domain evaluation block is ignored when it exists.
"""

import argparse
import re
from pathlib import Path


TASKS = [
    ("SABSCT->CHAOST2", "SABSCT", "sabsct_to_chaost2"),
    ("CHAOST2->SABSCT", "CHAOST2", "chaost2_to_sabsct"),
    ("bSSFP->LGE", "bSSFP", "bssfp_to_lge"),
    ("LGE->bSSFP", "LGE", "lge_to_bssfp"),
]

BASELINE = {
    "SABSCT->CHAOST2": 0.8147,
    "CHAOST2->SABSCT": 0.7769,
    "bSSFP->LGE": 0.8571,
    "LGE->bSSFP": 0.8654,
}

LEGACY_EXP_NAMES = {
    "none": {
        "SABSCT->CHAOST2": "none_dcon_sc_sabsct",
        "CHAOST2->SABSCT": "none_dcon_cs_chaost2",
        "bSSFP->LGE": "none_dcon_bl_bssfp",
        "LGE->bSSFP": "none_dcon_lb_lge",
    },
    "tent": {
        "SABSCT->CHAOST2": "tent_dcon_sc_sabsct",
        "CHAOST2->SABSCT": "tent_dcon_cs_chaost2",
        "bSSFP->LGE": "tent_dcon_bl_bssfp",
        "LGE->bSSFP": "tent_dcon_lb_lge",
    },
    "norm_test": {
        "SABSCT->CHAOST2": "norm_test_dcon_sc_sabsct",
        "CHAOST2->SABSCT": "norm_test_dcon_cs_chaost2",
        "bSSFP->LGE": "norm_test_dcon_bl_bssfp",
        "LGE->bSSFP": "norm_test_dcon_lb_lge",
    },
    "norm_alpha": {
        "SABSCT->CHAOST2": "norm_alpha_dcon_sc_sabsct",
        "CHAOST2->SABSCT": "norm_alpha_dcon_cs_chaost2",
        "bSSFP->LGE": "norm_alpha_dcon_bl_bssfp",
        "LGE->bSSFP": "norm_alpha_dcon_lb_lge",
    },
    "norm_ema": {
        "SABSCT->CHAOST2": "norm_ema_dcon_sc_sabsct",
        "CHAOST2->SABSCT": "norm_ema_dcon_cs_chaost2",
        "bSSFP->LGE": "norm_ema_dcon_bl_bssfp",
        "LGE->bSSFP": "norm_ema_dcon_lb_lge",
    },
    "cotta": {
        "SABSCT->CHAOST2": "cotta_dcon_sc_sabsct",
        "CHAOST2->SABSCT": "cotta_dcon_cs_chaost2",
        "bSSFP->LGE": "cotta_dcon_bl_bssfp",
        "LGE->bSSFP": "cotta_dcon_lb_lge",
    },
    "memo": {
        "SABSCT->CHAOST2": "memo_dcon_sc_sabsct",
        "CHAOST2->SABSCT": "memo_dcon_cs_chaost2",
        "bSSFP->LGE": "memo_dcon_bl_bssfp",
        "LGE->bSSFP": "memo_dcon_lb_lge",
    },
    "asm": {
        "SABSCT->CHAOST2": "asm_dcon_sc_chaost2",
        "CHAOST2->SABSCT": "asm_dcon_cs_sabsct",
        "bSSFP->LGE": "asm_dcon_bl_lge",
        "LGE->bSSFP": "asm_dcon_lb_bssfp",
    },
    "sm_ppm": {
        "SABSCT->CHAOST2": "smppm_dcon_sc_chaost2",
        "CHAOST2->SABSCT": "smppm_dcon_cs_sabsct",
        "bSSFP->LGE": "smppm_dcon_bl_lge",
        "LGE->bSSFP": "smppm_dcon_lb_bssfp",
    },
    "gtta": {
        "SABSCT->CHAOST2": "gtta_dcon_sc_chaost2",
        "CHAOST2->SABSCT": "gtta_dcon_cs_sabsct",
        "bSSFP->LGE": "gtta_dcon_bl_lge",
        "LGE->bSSFP": "gtta_dcon_lb_bssfp",
    },
    "gold": {
        "SABSCT->CHAOST2": "gold_dcon_sc_sabsct",
        "CHAOST2->SABSCT": "gold_dcon_cs_chaost2",
        "bSSFP->LGE": "gold_dcon_bl_bssfp",
        "LGE->bSSFP": "gold_dcon_lb_lge",
    },
}

METHOD_ALIASES = {
    "sm_ppm": ["sm_ppm", "smppm"],
    "dg_tta": ["dg_tta", "dgtta"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DCON TTA methods by target Overall Dice by sample."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["results_medseg_tta", "results", "ckpts"],
        help="Experiment roots to search. Expected layout: <root>/<source>/<expname>/log/out.csv.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["none", "tent", "dg_tta", "memo", "asm", "sm_ppm", "gtta", "gold"],
        help="Methods to include.",
    )
    return parser.parse_args()


def target_overall_from_out_csv(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    target_block = text.split("Test mode evaluation", 1)[-1]
    target_block = target_block.split("Test on source domain", 1)[0]
    target_block = target_block.split("test for source domain", 1)[0]
    match = re.search(r"Overall mean dice by sample:?\s*([0-9]*\.?[0-9]+)", target_block)
    if match is None:
        return None
    return float(match.group(1))


def candidate_expnames(method, task, suffix):
    names = [f"{method}_dcon_{suffix}"]
    for alias in METHOD_ALIASES.get(method, []):
        names.append(f"{alias}_dcon_{suffix}")
    names.extend(LEGACY_EXP_NAMES.get(method, {}).get(task, "").split())
    return [name for name in names if name]


def find_out_csv(roots, source, expnames):
    for root in roots:
        for expname in expnames:
            candidate = Path(root) / source / expname / "log" / "out.csv"
            if candidate.exists():
                return candidate
    return None


def values_for_method(method, roots):
    values = {}
    for task, source, suffix in TASKS:
        path = find_out_csv(roots, source, candidate_expnames(method, task, suffix))
        values[task] = None if path is None else target_overall_from_out_csv(path)
    return values


def fmt(value):
    return "-" if value is None else f"{value:.4f}"


def average(values):
    present = [v for v in values.values() if v is not None]
    if len(present) != len(TASKS):
        return None
    return sum(present) / len(present)


def print_markdown(methods, rows):
    headers = ["method"] + [task for task, _, _ in TASKS] + ["avg", "avg_vs_none"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")

    baseline_values = rows.get("none")
    baseline_avg = average(baseline_values) if baseline_values else None
    if baseline_avg is None:
        baseline_avg = average(BASELINE)

    for method in methods:
        values = rows[method]
        avg = average(values)
        delta = None if avg is None or baseline_avg is None else avg - baseline_avg
        cells = [method] + [fmt(values[task]) for task, _, _ in TASKS] + [fmt(avg), fmt(delta)]
        print("| " + " | ".join(cells) + " |")


def main():
    args = parse_args()
    rows = {method: values_for_method(method, args.roots) for method in args.methods}
    print_markdown(args.methods, rows)


if __name__ == "__main__":
    main()
