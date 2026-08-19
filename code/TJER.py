import time
from collections import defaultdict
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def detect_violations(df_std, attr_cols, pos_cols, attr_thresholds, eps=1e-8):
    if len(pos_cols) == 2:
        dx = df_std[pos_cols[0]].diff()
        dy = df_std[pos_cols[1]].diff()
        df_std["delta_s"] = (dx**2 + dy**2) ** 0.5
    elif len(pos_cols) == 3:
        dx = df_std[pos_cols[0]].diff()
        dy = df_std[pos_cols[1]].diff()
        dz = df_std[pos_cols[2]].diff()
        df_std["delta_s"] = (dx**2 + dy**2 + dz**2) ** 0.5
    else:
        raise ValueError("pos_cols must contain 2 or 3 coordinates")

    n = len(df_std)
    viol_triplets = {}

    for attr in attr_cols:
        df_std[f"delta_{attr}"] = df_std[attr].diff()
        delta_A = df_std[f"delta_{attr}"].values
        delta_s = df_std["delta_s"].values.copy()

        W = np.zeros_like(delta_A)
        for i in range(len(delta_A)):
            if np.isnan(delta_A[i]):
                W[i] = 0
                continue
            if abs(delta_A[i]) < eps:
                if abs(delta_s[i]) < eps:
                    W[i] = 0
                else:
                    W[i] = 0.5 * np.sign(delta_s[i]) * (delta_s[i] ** 2)
                continue
            if abs(delta_s[i]) < eps:
                W[i] = 0.5 * np.sign(delta_A[i]) * (delta_A[i] ** 2)
            else:
                W[i] = 0.5 * delta_A[i] * delta_s[i]

        df_std[f"W_{attr}"] = W

        delta_A_diff = np.zeros_like(W)
        for i in range(len(W)):
            if i == 0:
                delta_A_diff[i] = abs(0 - W[i + 1])
            elif i == len(W) - 1:
                delta_A_diff[i] = abs(W[i] - 0)
            else:
                delta_A_diff[i] = abs(W[i] - W[i + 1])

        df_std[f"delta_A_diff_{attr}"] = delta_A_diff

        threshold = attr_thresholds.get(attr, None)
        if threshold is None:
            raise ValueError(f"Missing threshold for attribute {attr}")

        trips = []
        for i in range(1, n - 1):
            if delta_A_diff[i] >= threshold:
                trips.append((i - 1, i, i + 1))

        viol_triplets[attr] = trips

    return viol_triplets


class RepairState:
    def __init__(self, n):
        self.fix_s = [0] * n
        self.fix_x = [0] * n

    def copy(self):
        new = RepairState(len(self.fix_s))
        new.fix_s = self.fix_s.copy()
        new.fix_x = self.fix_x.copy()
        return new

    def __hash__(self):
        return hash((tuple(self.fix_s), tuple(self.fix_x)))

    def __eq__(self, other):
        return self.fix_s == other.fix_s and self.fix_x == other.fix_x

    def __repr__(self):
        return f"RepairState(s={self.fix_s}, x={self.fix_x})"


def enumerate_repairs_for_attr(attr, viol_triplets, n):
    start_time = pd.Timestamp.now()
    DP = [set() for _ in range(n + 1)]
    DP[0].add(RepairState(n))

    repair_modes = [(1, 0), (0, 1), (1, 1)]
    viol_list = viol_triplets[attr]

    for i in range(n):
        triplets = [t for t in viol_list if i in t]
        if not triplets:
            DP[i + 1] = DP[i].copy()
            continue

        appear_tail = any(t[2] == i for t in triplets)
        for R in DP[i]:
            if not appear_tail:
                DP[i + 1].add(R)
                for fs, fx in repair_modes:
                    new_R = R.copy()
                    if fs:
                        new_R.fix_s[i] = 1
                    if fx:
                        new_R.fix_x[i] = 1
                    DP[i + 1].add(new_R)
                continue

            if appear_tail:
                prev_fixed = False
                if i >= 2:
                    prev_fixed = (
                        R.fix_s[i - 2]
                        or R.fix_x[i - 2]
                        or R.fix_s[i - 1]
                        or R.fix_x[i - 1]
                    )
                if prev_fixed:
                    DP[i + 1].add(R)
                    for fs, fx in repair_modes:
                        new_R = R.copy()
                        if fs:
                            new_R.fix_s[i] = 1
                        if fx:
                            new_R.fix_x[i] = 1
                        DP[i + 1].add(new_R)
                    continue
                else:
                    for fs, fx in repair_modes:
                        new_R = R.copy()
                        if fs:
                            new_R.fix_s[i] = 1
                        if fx:
                            new_R.fix_x[i] = 1
                        DP[i + 1].add(new_R)
                continue

    repairs = list(DP[n])

    seen = set()
    unique_repairs = []
    for st in repairs:
        key = (tuple(st.fix_s), tuple(st.fix_x))
        if key not in seen:
            seen.add(key)
            unique_repairs.append(st)

    end_time = pd.Timestamp.now()
    return unique_repairs


def is_minimal(state, viol_triplets_attr):
    fix_s = np.array(state.fix_s, dtype=int)
    fix_x = np.array(state.fix_x, dtype=int)
    fix_mask = (fix_s | fix_x) > 0

    def covers_all(mask):
        for tri in viol_triplets_attr:
            if not any(mask[list(tri)]):
                return False
        return True

    if not covers_all(fix_mask):
        return False

    fix_indices = np.where(fix_mask)[0]
    for idx in fix_indices:
        new_mask = fix_mask.copy()
        new_mask[idx] = False
        if covers_all(new_mask):
            return False
    return True


def filter_minimal_repairs(repairs, viol_triplets_attr):
    start_time = pd.Timestamp.now()
    minimal = []
    for r in repairs:
        if is_minimal(r, viol_triplets_attr):
            minimal.append(r)
    end_time = pd.Timestamp.now()
    return minimal


def build_mask_buckets(all_attr_repairs, attr_cols):
    buckets = defaultdict(lambda: defaultdict(list))
    for attr in attr_cols:
        states = all_attr_repairs.get(attr, [])
        for st in states:
            mask = tuple(st.fix_s)
            buckets[mask][attr].append(st)

    required_attrs = set(attr_cols)
    mask_buckets = {}
    valid_mask_count = 0

    for mask, attr_to_states in buckets.items():
        attrs_in_bucket = set(attr_to_states.keys())
        if attrs_in_bucket == required_attrs:
            mask_buckets[mask] = attr_to_states
            valid_mask_count += 1

    return mask_buckets


def group_violation_triplets(trips):
    if not trips:
        return []

    groups = []
    current_group = [trips[0]]

    for i in range(1, len(trips)):
        tri_new = trips[i]

        has_intersection = False
        for tri_in_group in current_group:
            if set(tri_new) & set(tri_in_group):
                has_intersection = True
                break

        if has_intersection:
            current_group.append(tri_new)
        else:
            groups.append(current_group)
            current_group = [tri_new]

    groups.append(current_group)

    return groups


def get_split_ranges_for_groups(groups, n):
    if not groups:
        return []

    ranges = []

    group_ranges = []
    for group in groups:
        all_indices = set()
        for a, b, c in group:
            all_indices.update([a, b, c])

        min_idx = min(all_indices)
        max_idx = max(all_indices)
        group_ranges.append((min_idx, max_idx))

    for i, (min_idx, max_idx) in enumerate(group_ranges):
        if i == 0:
            start = 0
        else:
            prev_max = group_ranges[i - 1][1]
            if prev_max + 1 < min_idx:
                gap_size = min_idx - prev_max - 1
                mid_point = prev_max + 1 + gap_size // 2
                start = mid_point
            else:
                start = min_idx

        if i == len(group_ranges) - 1:
            if n - 1 - max_idx > 2:
                end = max_idx
            else:
                end = n - 1
        else:
            next_min = group_ranges[i + 1][0]
            if max_idx + 1 < next_min:
                gap_size = next_min - max_idx - 1
                mid_point = max_idx + 1 + gap_size // 2
                end = mid_point - 1
            else:
                end = max_idx

        ranges.append((start, end))
    last_max = group_ranges[-1][1]
    if n - 1 - last_max > 2:
        clean_start = last_max + 1
        clean_end = n - 1
        ranges.append((clean_start, clean_end))
    return ranges


def merge_group_repairs_for_attr(group_repairs_with_ranges, n):
    if not group_repairs_with_ranges:
        return []
    repairs_lists = [repairs for repairs, _ in group_repairs_with_ranges]
    all_combos = list(product(*repairs_lists))

    merged_repairs = []

    for combo in all_combos:
        merged = RepairState(n)

        for group_idx, state in enumerate(combo):
            start, end = group_repairs_with_ranges[group_idx][1]
            local_n = end - start + 1

            for local_i in range(local_n):
                global_i = start + local_i
                merged.fix_s[global_i] |= state.fix_s[local_i]
                merged.fix_x[global_i] |= state.fix_x[local_i]

        merged_repairs.append(merged)

    return merged_repairs


def main_precision_repair(df_std, pos_cols, attr_cols, attr_thresholds):
    n = len(df_std)
    viol_triplets = detect_violations(df_std, attr_cols, pos_cols, attr_thresholds)

    all_attr_repairs = {}

    for attr in attr_cols:
        trips_for_attr = viol_triplets[attr]
        groups = group_violation_triplets(trips_for_attr)
        split_ranges = get_split_ranges_for_groups(groups, n)

        group_repairs_with_ranges = []

        for group, (start, end) in zip(groups, split_ranges):
            local_n = end - start + 1

            local_group = []
            for a, b, c in group:
                local_group.append((a - start, b - start, c - start))

            repairs = enumerate_repairs_for_attr(
                attr,
                {attr: local_group},
                local_n,
            )
            minimal_repairs = filter_minimal_repairs(repairs, local_group)
            group_repairs_with_ranges.append((minimal_repairs, (start, end)))

        all_attr_repairs[attr] = merge_group_repairs_for_attr(
            group_repairs_with_ranges, n
        )

    mask_buckets = build_mask_buckets(all_attr_repairs, attr_cols)
    return all_attr_repairs, mask_buckets


def generate_all_minimal_repairs(mask_buckets, attr_cols):
    minimal_repairs = []

    for mask_s, attr_to_states in mask_buckets.items():

        attr_mask_x_options = {}
        for attr in attr_cols:
            states = attr_to_states[attr]
            mask_x_set = set(tuple(st.fix_x) for st in states)
            attr_mask_x_options[attr] = list(mask_x_set)

        attr_names = list(attr_cols)
        mask_x_product = product(*[attr_mask_x_options[a] for a in attr_names])

        for mask_x_combo in mask_x_product:
            mask_x = {attr_names[i]: mask_x_combo[i] for i in range(len(attr_names))}

            minimal_repairs.append({"mask_s": mask_s, "mask_x": mask_x})

    return minimal_repairs


def local_deltaA_diff(df, pos_cols, attr, a, b, c):
    pa = df.loc[a, pos_cols].to_numpy(dtype=float)
    pb = df.loc[b, pos_cols].to_numpy(dtype=float)
    pc = df.loc[c, pos_cols].to_numpy(dtype=float)

    ds_ab = np.linalg.norm(pb - pa)
    ds_bc = np.linalg.norm(pc - pb)

    attr_ab = float(df.loc[b, attr] - df.loc[a, attr])
    attr_bc = float(df.loc[c, attr] - df.loc[b, attr])

    W_ab = 0.5 * attr_ab * ds_ab
    W_bc = 0.5 * attr_bc * ds_bc

    return abs(W_ab - W_bc)


def prune_pos_candidates_by_local_triplets(
    df, minimal_repair, pos_cols, attr_cols, pos_ranges, attr_thresholds
):
    n = len(df)
    mask_s = minimal_repair["mask_s"]
    mask_x = minimal_repair["mask_x"]

    for idx in list(pos_ranges.keys()):

        raw_triplets = [
            (idx - 2, idx - 1, idx),
            (idx - 1, idx, idx + 1),
            (idx, idx + 1, idx + 2),
        ]

        valid_triplets = {attr: [] for attr in attr_cols}

        for attr in attr_cols:
            if mask_x[attr][idx] == 1:
                continue

            for a, b, c in raw_triplets:
                if a < 0 or c >= n:
                    continue

                ok = True
                for t in (a, b, c):
                    if mask_x[attr][t] == 1:
                        ok = False
                        break
                    if t != idx and mask_s[t] == 1:
                        ok = False
                        break

                if ok:
                    valid_triplets[attr].append((a, b, c))

        if all(len(valid_triplets[attr]) == 0 for attr in attr_cols):
            continue

        all_vecs = pos_ranges[idx]

        pruned_vecs = []

        for vec in all_vecs:
            df_tmp = df.copy()
            for d, col in enumerate(pos_cols):
                df_tmp.loc[idx, col] = vec[d]

            feasible = True

            for attr in attr_cols:
                trips = valid_triplets[attr]
                if not trips:
                    continue

                thr = attr_thresholds[attr]

                for a, b, c in trips:
                    deltaA = local_deltaA_diff(df_tmp, pos_cols, attr, a, b, c)
                    if deltaA > thr:
                        feasible = False
                        break

                if not feasible:
                    break

            if feasible:
                pruned_vecs.append(vec)

        pos_ranges[idx] = pruned_vecs

    return pos_ranges


def estimate_repair_ranges(
    df,
    minimal_repair,
    pos_cols,
    attr_cols,
    step=1.0,
    max_candidates=5,
    attr_thresholds=None,
    win_size=5,
    pruning_stats=None,
    repair_id=None,
):
    mask_s = minimal_repair["mask_s"]
    mask_x = minimal_repair["mask_x"]
    n = len(df)

    correct_pairs = []
    for i in range(n - 1):
        if mask_s[i] == 0 and mask_s[i + 1] == 0:
            correct_pairs.append((i, i + 1))

    pos_theta = {}
    for col in pos_cols:
        max_diff = (
            max(abs(df.loc[j, col] - df.loc[i, col]) for i, j in correct_pairs)
            if correct_pairs
            else 0
        )
        pos_theta[col] = max_diff

    attr_theta = {}
    for attr in attr_cols:
        max_diff = (
            max(abs(df.loc[j, attr] - df.loc[i, attr]) for i, j in correct_pairs)
            if correct_pairs
            else 0
        )
        attr_theta[attr] = max_diff

    pos_fix_positions = [i for i in range(n) if mask_s[i] == 1]

    attr_fix_positions = {
        attr: [i for i in range(n) if mask_x[attr][i] == 1] for attr in attr_cols
    }

    def find_clean_neighbor(idx, mask):
        if idx - 1 >= 0 and mask[idx - 1] == 0:
            return idx - 1
        if idx + 1 < len(mask) and mask[idx + 1] == 0:
            return idx + 1

    pos_ranges = {}

    for idx in pos_fix_positions:
        pos_ranges[idx] = {}
        neigh = find_clean_neighbor(idx, mask_s)

        for col in pos_cols:
            ref_val = df.loc[neigh, col]
            theta = pos_theta[col]

            lo = ref_val - theta
            hi = ref_val + theta

            if max_candidates == 1:
                candidates = np.array([ref_val])
            else:
                candidates = np.linspace(lo, hi, max_candidates)

            pos_ranges[idx][col] = candidates

    attr_ranges = {attr: {} for attr in attr_cols}

    for attr in attr_cols:
        for idx in attr_fix_positions[attr]:
            neigh = find_clean_neighbor(idx, mask_x[attr])
            ref_val = df.loc[neigh, attr]
            theta = attr_theta[attr]

            lo = ref_val - theta
            hi = ref_val + theta

            if max_candidates == 1:
                candidates = np.array([ref_val])
            else:
                candidates = np.linspace(lo, hi, max_candidates)

            attr_ranges[attr][idx] = candidates

    pos_ranges_comb = {}
    for idx, dic in pos_ranges.items():
        dim_lists = [dic[col] for col in pos_cols]
        combos = [tuple(float(v) for v in vals) for vals in product(*dim_lists)]
        pos_ranges_comb[idx] = combos
    pos_ranges = pos_ranges_comb

    before_counts = {idx: len(cands) for idx, cands in pos_ranges.items()}
    pos_ranges = prune_pos_candidates_by_local_triplets(
        df, minimal_repair, pos_cols, attr_cols, pos_ranges, attr_thresholds
    )
    after_counts = {idx: len(cands) for idx, cands in pos_ranges.items()}

    if pruning_stats is not None:
        sum_before = sum(before_counts.values())
        sum_after = sum(after_counts.values())

        pruning_stats["pos"]["candidate_sum_before"] += sum_before
        pruning_stats["pos"]["candidate_sum_after"] += sum_after
        pruning_stats["pos"]["num_minimal_repairs"] += 1

        pruning_stats["pos"]["records"].append(
            {
                "repair_id": repair_id,
                "before_sum": sum_before,
                "after_sum": sum_after,
                "pruning_rate": (
                    0.0 if sum_before == 0 else 1.0 - sum_after / sum_before
                ),
                "before_counts": deepcopy(before_counts),
                "after_counts": deepcopy(after_counts),
            }
        )

    return pos_ranges, attr_ranges


def check_area_constraint(df_modified, pos_cols, attr, attr_thresholds, eps=1e-8):
    if len(pos_cols) == 2:
        dx = df_modified[pos_cols[0]].diff()
        dy = df_modified[pos_cols[1]].diff()
        deltaS = np.sqrt(dx**2 + dy**2)
    elif len(pos_cols) == 3:
        dx = df_modified[pos_cols[0]].diff()
        dy = df_modified[pos_cols[1]].diff()
        dz = df_modified[pos_cols[2]].diff()
        deltaS = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        raise ValueError("pos_cols must contain 2 or 3 coordinates")

    deltaA = df_modified[attr].diff().values
    deltaS_vals = deltaS.values

    W = np.zeros_like(deltaA)
    for i in range(len(deltaA)):
        if np.isnan(deltaA[i]):
            W[i] = 0
            continue
        if abs(deltaS_vals[i]) < eps:
            W[i] = 0.5 * np.sign(deltaA[i]) * (deltaA[i] ** 2)
        else:
            W[i] = 0.5 * deltaA[i] * deltaS_vals[i]

    deltaA_diff = np.zeros_like(W)
    for i in range(len(W)):
        if i == 0:
            deltaA_diff[i] = abs(0 - W[i + 1])
        elif i == len(W) - 1:
            deltaA_diff[i] = abs(W[i] - 0)
        else:
            deltaA_diff[i] = abs(W[i] - W[i + 1])

    threshold = attr_thresholds[attr]

    return np.max(deltaA_diff) <= threshold


def sliding_window_regression_loss(df, pos_cols, attr, win_size=5):
    n = len(df)
    half = win_size // 2
    errors = []

    first_window = df.iloc[0:win_size]
    Xf = first_window[pos_cols].values
    yf = first_window[attr].values
    reg_first = LinearRegression().fit(Xf, yf)

    for i in range(0, half):
        X_i = df.loc[[i], pos_cols].values
        y_pred = reg_first.predict(X_i)[0]
        y_true = df.loc[i, attr]
        errors.append((y_pred - y_true) ** 2)

    for i in range(half, n - half):
        window = df.iloc[i - half : i + half + 1]
        X = window[pos_cols].values
        y = window[attr].values

        reg = LinearRegression().fit(X, y)

        X_i = df.loc[[i], pos_cols].values
        y_pred = reg.predict(X_i)[0]
        y_true = df.loc[i, attr]
        errors.append((y_pred - y_true) ** 2)

    last_window = df.iloc[n - win_size : n]
    Xl = last_window[pos_cols].values
    yl = last_window[attr].values
    reg_last = LinearRegression().fit(Xl, yl)

    for i in range(n - half, n):
        X_i = df.loc[[i], pos_cols].values
        y_pred = reg_last.predict(X_i)[0]
        y_true = df.loc[i, attr]
        errors.append((y_pred - y_true) ** 2)

    if not errors:
        return 0.0

    std = df[attr].std()
    if std < 1e-8:
        std = 1.0

    return np.sum(errors) / (std**2)


def prune_attr_candidates_by_local_triplets(
    df_pos_fixed,
    minimal_repair,
    pos_cols,
    attr,
    attr_ranges_attr,
    attr_thresholds,
    attr_cols=None,
    eps=1e-8,
):
    n = len(df_pos_fixed)
    mask_x_attr = minimal_repair["mask_x"][attr]

    new_attr_ranges = {}

    for idx, cand_array in attr_ranges_attr.items():
        if cand_array is None or len(cand_array) == 0:
            new_attr_ranges[idx] = cand_array
            continue

        raw_triplets = [
            (idx - 2, idx - 1, idx),
            (idx - 1, idx, idx + 1),
            (idx, idx + 1, idx + 2),
        ]

        valid_triplets = []

        for a, b, c in raw_triplets:
            if a < 0 or c >= n:
                continue
            if idx not in (a, b, c):
                continue

            ok = True
            for t in (a, b, c):
                if t == idx:
                    continue
                if mask_x_attr[t] == 1:
                    ok = False
                    break

            if ok:
                valid_triplets.append((a, b, c))

        if len(valid_triplets) == 0:
            new_attr_ranges[idx] = cand_array
            continue

        thr = attr_thresholds[attr]
        good_vals = []

        for v in cand_array:
            df_tmp = df_pos_fixed.copy()
            df_tmp.loc[idx, attr] = v

            feasible = True
            for a, b, c in valid_triplets:
                deltaA = local_deltaA_diff(df_tmp, pos_cols, attr, a, b, c)
                if deltaA > thr:
                    feasible = False
                    break

            if feasible:
                good_vals.append(v)

        new_attr_ranges[idx] = np.array(good_vals)

    return new_attr_ranges


def enumerate_one_minimal_repair(
    df,
    minimal_repair,
    pos_cols,
    attr_cols,
    pos_ranges,
    attr_ranges,
    attr_thresholds,
    win_size=5,
    pruning_stats=None,
    repair_id=None,
):

    n = len(df)
    mask_s = minimal_repair["mask_s"]
    mask_x = minimal_repair["mask_x"]

    fix_positions = [i for i in range(n) if mask_s[i] == 1]

    all_pos_candidate_lists = []
    for idx in fix_positions:
        cand_list = pos_ranges[idx]
        if len(cand_list) == 0:
            return None
        all_pos_candidate_lists.append(cand_list)

    all_pos_combos = list(product(*all_pos_candidate_lists))

    best_total_loss = float("inf")
    best_repair = None

    for combo_i, pos_combo in enumerate(all_pos_combos):
        df_pos_repaired = df.copy()

        for k, idx in enumerate(fix_positions):
            vec = pos_combo[k]
            for d, col in enumerate(pos_cols):
                df_pos_repaired.loc[idx, col] = vec[d]

        attr_best_combos = {}
        attr_min_losses = {}

        for attr in attr_cols:
            attr_fix_positions = [i for i in range(n) if mask_x[attr][i] == 1]

            local_attr_ranges = {
                idx: attr_ranges[attr].get(idx, np.array([]))
                for idx in attr_fix_positions
            }

            before1 = {idx: len(local_attr_ranges[idx]) for idx in attr_fix_positions}
            local_attr_ranges = prune_attr_candidates_by_local_triplets(
                df_pos_fixed=df_pos_repaired,
                minimal_repair=minimal_repair,
                pos_cols=pos_cols,
                attr=attr,
                attr_ranges_attr=local_attr_ranges,
                attr_thresholds=attr_thresholds,
                attr_cols=attr_cols,
            )
            after1 = {idx: len(local_attr_ranges[idx]) for idx in attr_fix_positions}
            for idx in attr_fix_positions:
                b, a = before1[idx], after1[idx]
                ratio = a / b if b > 0 else 0

            if pruning_stats is not None:
                sum_before = sum(before1.values())
                sum_after = sum(after1.values())

                pruning_stats["attr"]["candidate_sum_before"] += sum_before
                pruning_stats["attr"]["candidate_sum_after"] += sum_after
                pruning_stats["attr"]["per_attr_before"][attr] += sum_before
                pruning_stats["attr"]["per_attr_after"][attr] += sum_after
                pruning_stats["attr"]["num_attr_blocks"] += 1

                pruning_stats["attr"]["records"].append(
                    {
                        "repair_id": repair_id,
                        "attr": attr,
                        "before_sum": sum_before,
                        "after_sum": sum_after,
                        "pruning_rate": (
                            0.0 if sum_before == 0 else 1.0 - sum_after / sum_before
                        ),
                        "before_counts": deepcopy(before1),
                        "after_counts": deepcopy(after1),
                    }
                )
            impossible = False
            attr_candidates_lists = []

            for idx in attr_fix_positions:
                cands = local_attr_ranges.get(idx, [])
                if cands is None or len(cands) == 0:
                    impossible = True
                    break
                attr_candidates_lists.append(cands)

            if impossible:
                attr_min_losses[attr] = float("inf")
                attr_best_combos[attr] = {}
                continue

            if len(attr_fix_positions) > 0:
                all_attr_combos = list(product(*attr_candidates_lists))
            else:
                all_attr_combos = [()]

            min_loss = float("inf")
            best_attr_combo = None

            for combo in all_attr_combos:
                df_full = df_pos_repaired.copy()

                for j, idx in enumerate(attr_fix_positions):
                    df_full.loc[idx, attr] = combo[j]

                if not check_area_constraint(df_full, pos_cols, attr, attr_thresholds):
                    continue

                loss = sliding_window_regression_loss(df_full, pos_cols, attr, win_size)

                if loss < min_loss:
                    min_loss = loss
                    best_attr_combo = combo

            if best_attr_combo is None:
                attr_min_losses[attr] = float("inf")
                attr_best_combos[attr] = {}
                continue

            attr_min_losses[attr] = min_loss
            attr_best_combos[attr] = dict(zip(attr_fix_positions, best_attr_combo))

        total_loss = sum(attr_min_losses.values())

        if total_loss < best_total_loss:
            best_total_loss = total_loss
            best_repair = {
                "pos_values": {
                    idx: {col: pos_combo[i][d] for d, col in enumerate(pos_cols)}
                    for i, idx in enumerate(fix_positions)
                },
                "attr_values": attr_best_combos,
                "total_loss": best_total_loss,
                "attr_losses": attr_min_losses,
            }
    return best_repair


def brute_force_exact_solution(
    df,
    mask_buckets,
    pos_cols,
    attr_cols,
    attr_thresholds,
    step=1.0,
    win_size=5,
    pruning_stats=None,
):

    start_time = time.time()

    all_minimal_repairs = generate_all_minimal_repairs(mask_buckets, attr_cols)

    if len(all_minimal_repairs) == 0:
        return None, pruning_stats

    global_best_loss = float("inf")
    global_best = None

    for mr_idx, minimal_repair in enumerate(all_minimal_repairs):
        pos_ranges, attr_ranges = estimate_repair_ranges(
            df,
            minimal_repair,
            pos_cols,
            attr_cols,
            step=step,
            max_candidates=5,
            attr_thresholds=attr_thresholds,
            win_size=win_size,
            pruning_stats=pruning_stats,
            repair_id=mr_idx,
        )

        best_repair = enumerate_one_minimal_repair(
            df,
            minimal_repair,
            pos_cols,
            attr_cols,
            pos_ranges,
            attr_ranges,
            attr_thresholds,
            win_size,
            pruning_stats=pruning_stats,
            repair_id=mr_idx,
        )

        if best_repair is None:
            continue

        if best_repair["total_loss"] < global_best_loss:
            global_best_loss = best_repair["total_loss"]
            global_best = {
                "minimal_repair": minimal_repair,
                "pos_values": best_repair["pos_values"],
                "attr_values": best_repair["attr_values"],
                "total_loss": best_repair["total_loss"],
                "attr_losses": best_repair["attr_losses"],
            }

    end_time = time.time()
    return global_best, pruning_stats


def apply_repair_to_dataframe(df, global_best, pos_cols, attr_cols):
    df_repaired = df.copy()

    pos_values = global_best["pos_values"]
    for idx, pos_dict in pos_values.items():
        for pos_col, value in pos_dict.items():
            df_repaired.loc[idx, pos_col] = value

    attr_values = global_best["attr_values"]
    for attr, attr_dict in attr_values.items():
        for idx, value in attr_dict.items():
            df_repaired.loc[idx, attr] = value

    return df_repaired
