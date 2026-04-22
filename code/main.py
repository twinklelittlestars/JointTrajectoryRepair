import argparse
import json
import signal
import time

import numpy as np
import pandas as pd
from TJCR import brute_force_exact_solution as tjcr_brute_force_exact_solution
from TJCR import fuse_attr_solutions_with_shared_positions
from TJCR import main_precision_repair as tjcr_main_precision_repair
from TJCR import patch_after_shared_fusion
from TJER import apply_repair_to_dataframe as tjer_apply_repair_to_dataframe
from TJER import brute_force_exact_solution as tjer_brute_force_exact_solution
from TJER import main_precision_repair as tjer_main_precision_repair
from TJGR import approximate_repair_tjgr

DATASET_CONFIGS = {
    "pvs": {
        "path_err": "path/data/PVS/dirty.csv",
        "path_clean": "path/data/PVS/clean.csv",
        "pos_cols": ["latitude", "longitude", "elevation"],
        "attr_cols": ["bearing", "speed_meters_per_second", "distance_meters"],
        "attr_thresholds": {
            "bearing": 50.0,
            "speed_meters_per_second": 20000.0,
            "distance_meters": 18000.0,
        },
    },
    "onboard": {
        "path_err": "path/data/Onboard/dirty.csv",
        "path_clean": "path/data/Onboard/clean.csv",
        "pos_cols": ["lat", "lon", "alt"],
        "attr_cols": ["vel_n_m_s", "vel_e_m_s", "vel_d_m_s"],
        "attr_thresholds": {
            "vel_n_m_s": 150.0,
            "vel_e_m_s": 100.0,
            "vel_d_m_s": 180.0,
        },
    },
    "opensky": {
        "path_err": "path/data/OpenSky/dirty.csv",
        "path_clean": "path/data/OpenSky/clean.csv",
        "pos_cols": ["lat", "lon", "geoaltitude"],
        "attr_cols": ["velocity", "heading", "vertrate", "baroaltitude"],
        "attr_thresholds": {
            "velocity": 25000.0,
            "heading": 30000.0,
            "vertrate": 12000.0,
            "baroaltitude": 30000.0,
        },
    },
}


def parse_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_thresholds(text):
    data = json.loads(text)
    return {str(k): float(v) for k, v in data.items()}


def resolve_config(args):
    if args.dataset in DATASET_CONFIGS:
        return DATASET_CONFIGS[args.dataset]

    if (
        args.path_err is None
        or args.path_clean is None
        or args.pos_cols is None
        or args.attr_cols is None
        or args.attr_thresholds is None
    ):
        raise ValueError(
            "For a custom dataset, you must provide "
            "--path_err --path_clean --pos_cols --attr_cols --attr_thresholds"
        )

    return {
        "path_err": args.path_err,
        "path_clean": args.path_clean,
        "pos_cols": parse_list(args.pos_cols),
        "attr_cols": parse_list(args.attr_cols),
        "attr_thresholds": parse_thresholds(args.attr_thresholds),
    }


def compute_column_scales_iqr(df_ref, cols, eps=1e-8):
    scales = {}
    for c in cols:
        x = pd.to_numeric(df_ref[c], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) == 0:
            scales[c] = float("nan")
            continue

        q25 = np.percentile(x, 25)
        q75 = np.percentile(x, 75)
        s = q75 - q25

        if (not np.isfinite(s)) or s < eps:
            s = eps

        scales[c] = float(s)
    return scales


def compute_overall_nrmse_nmae(df_repaired, df_clean, cols, scales):
    n = min(len(df_repaired), len(df_clean))
    if n == 0:
        return float("nan"), float("nan")

    A = (
        df_repaired[cols]
        .iloc[:n]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    B = (
        df_clean[cols]
        .iloc[:n]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    valid_row = (~np.isnan(A).any(axis=1)) & (~np.isnan(B).any(axis=1))
    if valid_row.sum() == 0:
        return float("nan"), float("nan")

    A = A[valid_row]
    B = B[valid_row]
    n_valid = A.shape[0]

    diff = A - B
    diff_abs = np.abs(diff)

    for j, c in enumerate(cols):
        diff[:, j] = diff[:, j] / scales[c]
        diff_abs[:, j] = diff_abs[:, j] / scales[c]

    m_plus_1 = len(cols) + 1
    overall_nrmse = float(np.sqrt(np.sum(diff**2) / (n_valid * m_plus_1)))
    overall_nmae = float(np.sum(diff_abs) / (n_valid * m_plus_1))

    return overall_nrmse, overall_nmae


def run_tjer(df_err, pos_cols, attr_cols, attr_thresholds, w):
    all_attr_repairs, mask_buckets = tjer_main_precision_repair(
        df_err, pos_cols, attr_cols, attr_thresholds
    )

    global_best, _ = tjer_brute_force_exact_solution(
        df=df_err,
        mask_buckets=mask_buckets,
        pos_cols=pos_cols,
        attr_cols=attr_cols,
        attr_thresholds=attr_thresholds,
        step=1.0,
        win_size=w,
        pruning_stats=None,
    )

    if global_best is None:
        raise RuntimeError("TJER failed to produce a valid repair result.")

    df_repaired = tjer_apply_repair_to_dataframe(
        df_err, global_best, pos_cols, attr_cols
    )
    return df_repaired


def run_tjcr(df_err, pos_cols, attr_cols, attr_thresholds, w):
    per_attr_info = tjcr_main_precision_repair(
        df_err, pos_cols, attr_cols, attr_thresholds
    )

    final_attr_solutions = {}
    for attr in attr_cols:
        global_best = tjcr_brute_force_exact_solution(
            df=df_err,
            attr=attr,
            attr_info=per_attr_info[attr],
            pos_cols=pos_cols,
            attr_threshold=attr_thresholds[attr],
            win_size=w,
        )
        final_attr_solutions[attr] = global_best

    _, df_repaired, _ = fuse_attr_solutions_with_shared_positions(
        df_original=df_err,
        final_attr_solutions=final_attr_solutions,
        pos_cols=pos_cols,
        attr_cols=attr_cols,
    )

    df_repaired = patch_after_shared_fusion(
        df_repaired=df_repaired,
        pos_cols=pos_cols,
        attr_cols=attr_cols,
        attr_thresholds=attr_thresholds,
        max_candidates=5,
        win_size=w,
        max_iter_per_attr=3,
    )

    return df_repaired


def run_tjgr(df_err, pos_cols, attr_cols, attr_thresholds, w):
    return approximate_repair_tjgr(
        df_std=df_err,
        pos_cols=pos_cols,
        attr_cols=attr_cols,
        attr_thresholds=attr_thresholds,
        win_size=w,
    )


def _timeout_handler(signum, frame):
    raise TimeoutError("TJER exceeded the 24-hour time limit.")


def run_tjer_with_timeout(
    df_err, pos_cols, attr_cols, attr_thresholds, w, timeout_sec=86400
):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        return run_tjer(df_err, pos_cols, attr_cols, attr_thresholds, w)
    finally:
        signal.alarm(0)


def process_one_icao24_group_tjer(df_err_g, pos_cols, attr_cols, attr_thresholds, w):
    all_cols = pos_cols + attr_cols
    df_err_g_sorted = df_err_g.sort_values("time", kind="mergesort").copy()

    df_err_alg = df_err_g_sorted[all_cols].copy()
    valid_mask = ~df_err_alg.isna().any(axis=1)
    df_err_alg_valid = df_err_alg.loc[valid_mask].copy().reset_index(drop=True)

    if len(df_err_alg_valid) < 3:
        return None, 0.0

    t0 = time.time()
    df_repaired_alg = run_tjer_with_timeout(
        df_err_alg_valid, pos_cols, attr_cols, attr_thresholds, w
    )
    elapsed = time.time() - t0

    repaired_g_sorted = df_err_g_sorted.copy()
    repaired_g_sorted.loc[valid_mask, all_cols] = df_repaired_alg[all_cols].to_numpy()
    return repaired_g_sorted, elapsed


def process_one_icao24_group_tjcr(df_err_g, pos_cols, attr_cols, attr_thresholds, w):
    all_cols = pos_cols + attr_cols
    df_err_g_sorted = df_err_g.sort_values("time", kind="mergesort").copy()

    df_err_alg = df_err_g_sorted[all_cols].copy()
    valid_mask = ~df_err_alg.isna().any(axis=1)
    df_err_alg_valid = df_err_alg.loc[valid_mask].copy().reset_index(drop=True)

    if len(df_err_alg_valid) < 3:
        return None, 0.0

    t0 = time.time()
    df_repaired_alg = run_tjcr(
        df_err_alg_valid, pos_cols, attr_cols, attr_thresholds, w
    )
    elapsed = time.time() - t0

    repaired_g_sorted = df_err_g_sorted.copy()
    repaired_g_sorted.loc[valid_mask, all_cols] = df_repaired_alg[all_cols].to_numpy()
    return repaired_g_sorted, elapsed


def process_one_icao24_group_tjgr(df_err_g, pos_cols, attr_cols, attr_thresholds, w):
    all_cols = pos_cols + attr_cols
    df_err_g_sorted = df_err_g.sort_values("time", kind="mergesort").copy()

    df_err_alg = df_err_g_sorted[all_cols].copy()
    valid_mask = ~df_err_alg.isna().any(axis=1)
    df_err_alg_valid = df_err_alg.loc[valid_mask].copy().reset_index(drop=True)

    if len(df_err_alg_valid) < 3:
        return None, 0.0

    t0 = time.time()
    df_repaired_alg = run_tjgr(
        df_err_alg_valid, pos_cols, attr_cols, attr_thresholds, w
    )
    elapsed = time.time() - t0

    repaired_g_sorted = df_err_g_sorted.copy()
    repaired_g_sorted.loc[valid_mask, all_cols] = df_repaired_alg[all_cols].to_numpy()
    return repaired_g_sorted, elapsed


def run_opensky_grouped(df_err, df_clean, alg, pos_cols, attr_cols, attr_thresholds, w):
    all_cols = pos_cols + attr_cols
    required_cols = ["icao24", "time"] + all_cols

    for name, df in [("dirty", df_err), ("clean", df_clean)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} file is missing columns: {missing}")

    err_ids = set(df_err["icao24"].dropna().unique())
    clean_ids = set(df_clean["icao24"].dropna().unique())
    common_ids = sorted(err_ids & clean_ids)

    df_repaired_all = df_err.copy()
    total_repair_time = 0.0

    for icao24 in common_ids:
        df_err_g = df_err[df_err["icao24"] == icao24].copy()
        df_clean_g = df_clean[df_clean["icao24"] == icao24].copy()

        if len(df_err_g) < 3 or len(df_clean_g) < 3:
            continue

        if alg == "TJER":
            repaired_g, elapsed = process_one_icao24_group_tjer(
                df_err_g, pos_cols, attr_cols, attr_thresholds, w
            )
        elif alg == "TJCR":
            repaired_g, elapsed = process_one_icao24_group_tjcr(
                df_err_g, pos_cols, attr_cols, attr_thresholds, w
            )
        elif alg == "TJGR":
            repaired_g, elapsed = process_one_icao24_group_tjgr(
                df_err_g, pos_cols, attr_cols, attr_thresholds, w
            )
        else:
            raise ValueError(f"Unknown algorithm: {alg}")

        if repaired_g is None:
            continue

        df_repaired_all.loc[repaired_g.index, all_cols] = repaired_g[all_cols]
        total_repair_time += elapsed

    df_err_eval = df_err.sort_values(["icao24", "time"], kind="mergesort").reset_index(
        drop=True
    )
    df_clean_eval = df_clean.sort_values(
        ["icao24", "time"], kind="mergesort"
    ).reset_index(drop=True)
    df_repaired_eval = df_repaired_all.sort_values(
        ["icao24", "time"], kind="mergesort"
    ).reset_index(drop=True)

    valid_mask_eval = (
        (~df_err_eval[all_cols].isna().any(axis=1))
        & (~df_clean_eval[all_cols].isna().any(axis=1))
        & (~df_repaired_eval[all_cols].isna().any(axis=1))
    )

    df_clean_eval = df_clean_eval.loc[valid_mask_eval].reset_index(drop=True)
    df_repaired_eval = df_repaired_eval.loc[valid_mask_eval].reset_index(drop=True)

    return df_repaired_all, df_repaired_eval, df_clean_eval, total_repair_time, all_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg", type=str, required=True, choices=["TJER", "TJCR", "TJGR"]
    )
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--path_err", type=str, default=None)
    parser.add_argument("--path_clean", type=str, default=None)
    parser.add_argument("--pos_cols", type=str, default=None)
    parser.add_argument("--attr_cols", type=str, default=None)
    parser.add_argument("--attr_thresholds", type=str, default=None)

    parser.add_argument("--w", type=int, default=7)
    args = parser.parse_args()

    cfg = resolve_config(args)
    path_err = cfg["path_err"]
    path_clean = cfg["path_clean"]
    pos_cols = cfg["pos_cols"]
    attr_cols = cfg["attr_cols"]
    attr_thresholds = cfg["attr_thresholds"]

    repair_cols = pos_cols + attr_cols

    df_err = pd.read_csv(path_err)
    df_clean = pd.read_csv(path_clean)

    if args.dataset == "opensky":
        print(f"----------------{args.alg}----------------")
        start_time = time.time()
        _, df_repaired_eval, df_clean_eval, total_repair_time, eval_cols = (
            run_opensky_grouped(
                df_err=df_err,
                df_clean=df_clean,
                alg=args.alg,
                pos_cols=pos_cols,
                attr_cols=attr_cols,
                attr_thresholds=attr_thresholds,
                w=args.w,
            )
        )
        end_time = time.time()

        scales = compute_column_scales_iqr(df_clean_eval, eval_cols)
        nrmse, nmae = compute_overall_nrmse_nmae(
            df_repaired=df_repaired_eval,
            df_clean=df_clean_eval,
            cols=eval_cols,
            scales=scales,
        )
        print(f"time: {total_repair_time:.6f}, nrmse: {nrmse:.6f}, nmae: {nmae:.6f}")
        print(f"wall_clock_time: {end_time - start_time:.6f}")
        return

    df_err = df_err[repair_cols].dropna().reset_index(drop=True)
    df_clean = df_clean[repair_cols].dropna().reset_index(drop=True)

    print(f"----------------{args.alg}----------------")
    start_time = time.time()

    if args.alg == "TJER":
        df_repaired = run_tjer_with_timeout(
            df_err, pos_cols, attr_cols, attr_thresholds, args.w
        )
    elif args.alg == "TJCR":
        df_repaired = run_tjcr(df_err, pos_cols, attr_cols, attr_thresholds, args.w)
    elif args.alg == "TJGR":
        df_repaired = run_tjgr(df_err, pos_cols, attr_cols, attr_thresholds, args.w)
    else:
        raise ValueError(f"Unknown algorithm: {args.alg}")

    end_time = time.time()

    scales = compute_column_scales_iqr(df_clean, repair_cols)
    nrmse, nmae = compute_overall_nrmse_nmae(
        df_repaired=df_repaired,
        df_clean=df_clean,
        cols=repair_cols,
        scales=scales,
    )

    print(f"time: {end_time - start_time:.6f}, nrmse: {nrmse:.6f}, nmae: {nmae:.6f}")


if __name__ == "__main__":
    main()
