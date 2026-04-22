import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def detect_violations(df_std, attr_cols, pos_cols, attr_thresholds, eps=1e-8):
    if len(pos_cols) == 2:
        dx = df_std[pos_cols[0]].diff()
        dy = df_std[pos_cols[1]].diff()
        df_std["deltaS"] = np.sqrt(dx**2 + dy**2)
    elif len(pos_cols) == 3:
        dx = df_std[pos_cols[0]].diff()
        dy = df_std[pos_cols[1]].diff()
        dz = df_std[pos_cols[2]].diff()
        df_std["deltaS"] = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        raise ValueError("pos_cols must contain 2 or 3 coordinates")

    n = len(df_std)
    viol_triplets = {}

    for attr in attr_cols:
        df_std["delta_attr"] = df_std[attr].diff()
        deltaA = df_std["delta_attr"].values
        deltaS = df_std["deltaS"].values.copy()

        W = np.zeros_like(deltaA)
        for i in range(len(deltaA)):
            if np.isnan(deltaA[i]):
                W[i] = 0
                continue
            if abs(deltaA[i]) < eps:
                if abs(deltaS[i]) < eps:
                    W[i] = 0
                else:
                    W[i] = 0.5 * np.sign(deltaS[i]) * (deltaS[i] * eps)
                continue
            if abs(deltaS[i]) < eps:
                W[i] = 0.5 * np.sign(deltaA[i]) * (deltaA[i] * eps)
            else:
                W[i] = 0.5 * deltaA[i] * deltaS[i]

        df_std[f"W_{attr}"] = W

        deltaA_diff = np.zeros_like(W)
        for i in range(len(W)):
            if i == 0:
                deltaA_diff[i] = abs(0 - W[i + 1])
            elif i == len(W) - 1:
                deltaA_diff[i] = abs(W[i] - 0)
            else:
                deltaA_diff[i] = abs(W[i] - W[i + 1])

        df_std[f"deltaA_diff_{attr}"] = deltaA_diff

        threshold = attr_thresholds.get(attr, None)
        if threshold is None:
            raise ValueError(f"Missing threshold for attribute {attr}")

        trips = []
        for i in range(1, n - 1):
            if deltaA_diff[i] > threshold:
                trips.append((i - 1, i, i + 1))

        viol_triplets[attr] = trips

    return viol_triplets


def compute_W_series(df, pos_cols, attr, eps=1e-8):
    n = len(df)

    if len(pos_cols) == 2:
        dx = df[pos_cols[0]].diff().values
        dy = df[pos_cols[1]].diff().values
        ds = np.sqrt(dx**2 + dy**2)
    elif len(pos_cols) == 3:
        dx = df[pos_cols[0]].diff().values
        dy = df[pos_cols[1]].diff().values
        dz = df[pos_cols[2]].diff().values
        ds = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        raise ValueError("pos_cols must contain 2 or 3 coordinates")

    da = df[attr].diff().values

    W = np.zeros(n, dtype=float)
    for i in range(n):
        if i == 0 or np.isnan(da[i]) or np.isnan(ds[i]):
            W[i] = 0.0
            continue

        if abs(da[i]) < eps:
            if abs(ds[i]) < eps:
                W[i] = 0.0
            else:
                W[i] = 0.5 * np.sign(ds[i]) * (ds[i] * eps)
        elif abs(ds[i]) < eps:
            W[i] = 0.5 * np.sign(da[i]) * (da[i] * eps)
        else:
            W[i] = 0.5 * da[i] * ds[i]

    return W


def local_deltaA_diff(df, pos_cols, attr, a, b, c, eps=1e-8):
    pa = df.loc[a, pos_cols].values.astype(float)
    pb = df.loc[b, pos_cols].values.astype(float)
    pc = df.loc[c, pos_cols].values.astype(float)

    if np.any(np.isnan(pa)) or np.any(np.isnan(pb)) or np.any(np.isnan(pc)):
        return np.nan

    ds_ab = float(np.linalg.norm(pb - pa))
    ds_bc = float(np.linalg.norm(pc - pb))

    da_ab = df.loc[b, attr] - df.loc[a, attr]
    da_bc = df.loc[c, attr] - df.loc[b, attr]

    if abs(da_ab) < eps:
        if abs(ds_ab) < eps:
            W_ab = 0.0
        else:
            W_ab = 0.5 * np.sign(ds_ab) * (ds_ab * eps)
    elif abs(ds_ab) < eps:
        W_ab = 0.5 * np.sign(da_ab) * (da_ab * eps)
    else:
        W_ab = 0.5 * da_ab * ds_ab

    if abs(da_bc) < eps:
        if abs(ds_bc) < eps:
            W_bc = 0.0
        else:
            W_bc = 0.5 * np.sign(ds_bc) * (ds_bc * eps)
    elif abs(ds_bc) < eps:
        W_bc = 0.5 * np.sign(da_bc) * (da_bc * eps)
    else:
        W_bc = 0.5 * da_bc * ds_bc

    return abs(W_ab - W_bc)


def detect_points_by_W_feasibility(W, delta):
    n = len(W)
    R = set()

    for i in range(1, n - 1):
        low, high = W[i] - delta, W[i] + delta
        left_ok = low <= W[i - 1] <= high
        right_ok = low <= W[i + 1] <= high

        if (not left_ok) and (not right_ok):
            R.add(i)

    return R


def prune_R_to_minimal_cover(trips, R):
    R = set(R)
    covered = [tau for tau in trips if (tau[0] in R or tau[1] in R or tau[2] in R)]

    for p in list(R):
        R_tmp = R - {p}

        still_cover = True
        for tau in covered:
            if not (tau[0] in R_tmp or tau[1] in R_tmp or tau[2] in R_tmp):
                still_cover = False
                break

        if still_cover:
            R.remove(p)

    return R


def point_influence_loss(df, pos_cols, attr, p, win_size=7, eps=1e-8):
    assert win_size % 2 == 1
    n = len(df)
    half = win_size // 2

    if p - half < 0:
        L, R = 0, min(n - 1, win_size - 1)
    elif p + half >= n:
        R, L = n - 1, max(0, n - win_size)
    else:
        L, R = p - half, p + half

    win = df.iloc[L : R + 1]

    local_idx = p - L
    if local_idx < 0 or local_idx >= len(win):
        return float("inf")

    win_wo_p = win.drop(win.index[local_idx])

    if len(win_wo_p) < 2:
        return float("inf")

    X = win_wo_p[pos_cols].values
    y = win_wo_p[attr].values
    reg = LinearRegression().fit(X, y)

    X_p = df.iloc[p][pos_cols].values.reshape(1, -1)
    y_p = float(df.iloc[p][attr])

    y_p_pred = float(reg.predict(X_p)[0])
    residuals = (y_p_pred - y_p) ** 2

    std = df[attr].std()
    if std < eps:
        std = 1.0

    return float(residuals / (std**2))


def find_uncovered_triplets(trips, R):
    return [
        tau
        for tau in trips
        if (tau[0] not in R and tau[1] not in R and tau[2] not in R)
    ]


def pick_responsible_point_by_loss(df, pos_cols, attr, tau, win_size):
    a, b, c = tau
    losses = {
        a: point_influence_loss(df, pos_cols, attr, a, win_size),
        b: point_influence_loss(df, pos_cols, attr, b, win_size),
        c: point_influence_loss(df, pos_cols, attr, c, win_size),
    }
    return max(losses, key=losses.get)


def prune_to_minimal_cover_2(trips, R):
    R = set(R)
    for tau in trips:
        if not (tau[0] in R or tau[1] in R or tau[2] in R):
            raise ValueError("R does not cover all trips")

    changed = True
    while changed:
        changed = False
        for p in list(R):
            R_tmp = R - {p}

            still_cover_all = True
            for tau in trips:
                if not (tau[0] in R_tmp or tau[1] in R_tmp or tau[2] in R_tmp):
                    still_cover_all = False
                    break

            if still_cover_all:
                R.remove(p)
                changed = True
                break

    return R


def check_global_minimality(trips, R):
    R = set(R)
    for tau in trips:
        if not (tau[0] in R or tau[1] in R or tau[2] in R):
            return False, []

    redundant = []
    for p in R:
        R_tmp = R - {p}
        still_cover_all = True

        for tau in trips:
            if not (tau[0] in R_tmp or tau[1] in R_tmp or tau[2] in R_tmp):
                still_cover_all = False
                break

        if still_cover_all:
            redundant.append(p)

    return (len(redundant) == 0), redundant


def build_reference_from_clean_neighbors(df, i, R, pos_cols, attr):
    n = len(df)

    left = i - 1
    while left >= 0 and left in R:
        left -= 1

    right = i + 1
    while right < n and right in R:
        right += 1

    refs_pos = []
    refs_attr = []

    if left >= 0:
        refs_pos.append(df.loc[left, pos_cols].values.astype(float))
        refs_attr.append(float(df.loc[left, attr]))

    if right < n:
        refs_pos.append(df.loc[right, pos_cols].values.astype(float))
        refs_attr.append(float(df.loc[right, attr]))

    if len(refs_pos) == 0:
        return df.loc[i, pos_cols].values.astype(float), float(df.loc[i, attr])

    ref_pos = np.mean(refs_pos, axis=0)
    ref_attr = float(np.mean(refs_attr))
    return ref_pos, ref_attr


def local_feasibility_check(df_tmp, pos_cols, attr, i, threshold):
    n = len(df_tmp)
    triplets = []
    if i - 2 >= 0:
        triplets.append((i - 2, i - 1, i))
    if i - 1 >= 0 and i + 1 < n:
        triplets.append((i - 1, i, i + 1))
    if i + 2 < n:
        triplets.append((i, i + 1, i + 2))

    for a, b, c in triplets:
        diff = local_deltaA_diff(df_tmp, pos_cols, attr, a, b, c, eps=1e-8)
        if diff > threshold:
            return False
    return True


def apply_option(df, i, pos_cols, attr, ref_pos, ref_attr, mode):
    df_tmp = df.copy()
    if mode in ("pos", "both"):
        for d, col in enumerate(pos_cols):
            df_tmp.loc[i, col] = ref_pos[d]
    if mode in ("attr", "both"):
        df_tmp.loc[i, attr] = ref_attr
    return df_tmp


def choose_best_repair_for_point(
    df, pos_cols, attr, i, ref_pos, ref_attr, threshold, win_size=7
):
    modes = ["attr", "pos", "both"]
    best = None
    best_loss = float("inf")

    for mode in modes:
        df_tmp = apply_option(df, i, pos_cols, attr, ref_pos, ref_attr, mode)
        if not local_feasibility_check(df_tmp, pos_cols, attr, i, threshold):
            continue
        loss = point_influence_loss(df_tmp, pos_cols, attr, i, win_size)

        if loss < best_loss:
            best_loss = loss
            best = df_tmp

    return best


def approximate_repair_tjgr(
    df_std,
    pos_cols,
    attr_cols,
    attr_thresholds,
    win_size=7,
):
    eps = 1e-8
    n = len(df_std)

    attr_repaired_dfs = {}
    attr_R_sets = {}

    for attr in attr_cols:
        df_rep = df_std.copy()
        threshold = attr_thresholds[attr]

        viol_triplets = detect_violations(df_rep, [attr], pos_cols, {attr: threshold})
        trips = viol_triplets.get(attr, [])

        if len(trips) == 0:
            attr_repaired_dfs[attr] = df_rep.copy()
            attr_R_sets[attr] = set()
            continue

        W = compute_W_series(df_rep, pos_cols, attr)
        R = detect_points_by_W_feasibility(W, threshold)
        R = prune_R_to_minimal_cover(trips, R)

        uncovered = find_uncovered_triplets(trips, R)
        while uncovered:
            tau = uncovered[0]
            p = pick_responsible_point_by_loss(df_rep, pos_cols, attr, tau, win_size)
            R.add(p)
            uncovered = [t for t in uncovered if p not in t]

        R = prune_to_minimal_cover_2(trips, R)

        for i in sorted(R):
            ref_pos, ref_attr = build_reference_from_clean_neighbors(
                df_rep, i, R, pos_cols, attr
            )
            best_df = choose_best_repair_for_point(
                df_rep, pos_cols, attr, i, ref_pos, ref_attr, threshold, win_size
            )
            if best_df is not None:
                df_rep = best_df

        attr_repaired_dfs[attr] = df_rep.copy()
        attr_R_sets[attr] = set(R)

    pos_repair_log = defaultdict(list)

    for attr in attr_cols:
        df_attr = attr_repaired_dfs[attr]
        for i in range(n):
            changed = False
            for col in pos_cols:
                if abs(df_attr.loc[i, col] - df_std.loc[i, col]) > eps:
                    changed = True
                    break
            if changed:
                pos_vec = df_attr.loc[i, pos_cols].values.astype(float)
                pos_repair_log[i].append(pos_vec)

    df_geo = df_std.copy()
    for i, pos_list in pos_repair_log.items():
        avg_pos = np.mean(pos_list, axis=0)
        for d, col in enumerate(pos_cols):
            df_geo.loc[i, col] = avg_pos[d]

    df_rep = df_geo.copy()
    for attr in attr_cols:
        df_attr = attr_repaired_dfs[attr]
        for i in range(n):
            if abs(df_attr.loc[i, attr] - df_std.loc[i, attr]) > eps:
                df_rep.loc[i, attr] = df_attr.loc[i, attr]

    viol_triplets_round2 = detect_violations(
        df_rep, attr_cols, pos_cols, attr_thresholds
    )

    R_pos = set(pos_repair_log.keys())

    max_iter_per_attr = 3

    for attr in attr_cols:
        threshold = attr_thresholds[attr]

        for _ in range(max_iter_per_attr):
            trips = viol_triplets_round2.get(attr, [])
            if len(trips) == 0:
                break

            for tau in trips:
                a, b, c = tau
                cand = [
                    (a, point_influence_loss(df_rep, pos_cols, attr, a, win_size)),
                    (b, point_influence_loss(df_rep, pos_cols, attr, b, win_size)),
                    (c, point_influence_loss(df_rep, pos_cols, attr, c, win_size)),
                ]
                cand.sort(key=lambda x: x[1], reverse=True)

                fixed = False
                for p, _loss in cand:
                    _, ref_attr = build_reference_from_clean_neighbors(
                        df_rep, p, R_pos, pos_cols, attr
                    )
                    old = df_rep.loc[p, attr]
                    df_rep.loc[p, attr] = ref_attr

                    if local_feasibility_check(df_rep, pos_cols, attr, p, threshold):
                        fixed = True
                        break
                    else:
                        df_rep.loc[p, attr] = old

                if not fixed:
                    pass

            viol_triplets_round2 = detect_violations(
                df_rep, attr_cols, pos_cols, attr_thresholds
            )

    return df_rep
