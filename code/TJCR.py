import time
from collections import defaultdict
from itertools import product

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

    y_p_pred = reg.predict(X_p)[0]
    residuals = (y_p_pred - y_p) ** 2

    std = df[attr].std()
    if std < eps:
        std = 1.0

    return float(residuals / (std**2))


def build_candidate_points_by_triplet_loss(
    df, viol_triplets_attr, pos_cols, attr, win_size=7
):
    points = set()
    for i, j, k in viol_triplets_attr:
        points.add(i)
        points.add(j)
        points.add(k)

    point_loss = {}
    for p in points:
        point_loss[p] = point_influence_loss(
            df=df, pos_cols=pos_cols, attr=attr, p=p, win_size=win_size
        )

    triplet_repr = {}
    C = set()

    for i, j, k in viol_triplets_attr:
        li = point_loss[i]
        lj = point_loss[j]
        lk = point_loss[k]

        if li >= lj and li >= lk:
            p_star = i
        elif lj >= li and lj >= lk:
            p_star = j
        else:
            p_star = k

        triplet_repr[(i, j, k)] = p_star
        C.add(p_star)

    return point_loss, triplet_repr, C


def enumerate_repairs_for_attr(attr, viol_triplets, n, allowed_positions):
    if allowed_positions is None:
        allowed_positions = set(range(n))
    start_time = pd.Timestamp.now()
    DP = [set() for _ in range(n + 1)]
    DP[0].add(RepairState(n))

    viol_list = viol_triplets[attr]

    for i in range(n):
        can_fix_i = i in allowed_positions
        triplets = [t for t in viol_list if i in t]
        if not triplets or not can_fix_i:
            DP[i + 1] = DP[i].copy()
            continue

        appear_tail = any(t[2] == i for t in triplets)
        for R in DP[i]:
            if appear_tail:
                prev_fixed = False
                if i >= 2:
                    prev_fixed = R.fix_s[i - 2] or R.fix_s[i - 1]
                if prev_fixed:
                    DP[i + 1].add(R)
                    new_R = R.copy()
                    new_R.fix_s[i] = 1
                    DP[i + 1].add(new_R)
                    continue
                else:
                    new_R = R.copy()
                    new_R.fix_s[i] = 1
                    DP[i + 1].add(new_R)
                continue
            must_fix_i = False

            for t in triplets:
                if t[2] == i:
                    continue

                other_pos = [p for p in t if p != i]
                if all(p not in allowed_positions for p in other_pos):
                    must_fix_i = True
                    break

            if must_fix_i:
                new_R = R.copy()
                new_R.fix_s[i] = 1
                DP[i + 1].add(new_R)
            else:
                DP[i + 1].add(R)

                new_R = R.copy()
                new_R.fix_s[i] = 1
                DP[i + 1].add(new_R)

    repairs = list(DP[n])

    seen = set()
    unique_repairs = []
    for st in repairs:
        key = tuple(st.fix_s)
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


def classify_triplet_causes(df, pos_cols, attr_cols, viol_triplets, eps=1e-9):
    triplet_type = {}
    pos_diff = df[pos_cols].diff()
    deltaS = np.sqrt((pos_diff**2).sum(axis=1)).dropna()
    mu_S = deltaS.mean()
    sigma_S = deltaS.std() + eps
    threshold_S = mu_S + 2.0 * sigma_S

    for attr in attr_cols:
        deltaA = df[attr].diff().abs().dropna()
        mu_A = deltaA.mean()
        sigma_A = deltaA.std() + eps
        threshold_A = mu_A + 2.0 * sigma_A

        type_map = {}
        for i, j, k in viol_triplets[attr]:
            dA1 = abs(df.loc[j, attr] - df.loc[i, attr])
            dA2 = abs(df.loc[k, attr] - df.loc[j, attr])

            pi = df.loc[i, pos_cols].to_numpy(dtype=float)
            pj = df.loc[j, pos_cols].to_numpy(dtype=float)
            pk = df.loc[k, pos_cols].to_numpy(dtype=float)
            dS1 = np.linalg.norm(pj - pi)
            dS2 = np.linalg.norm(pk - pj)

            pos_exceeds = (dS1 > threshold_S) or (dS2 > threshold_S)
            attr_exceeds = (dA1 > threshold_A) or (dA2 > threshold_A)

            if pos_exceeds and attr_exceeds:
                t = "both"
            elif pos_exceeds:
                t = "pos"
            elif attr_exceeds:
                t = "attr"
            else:
                t = "both"

            type_map[(i, j, k)] = t

        triplet_type[attr] = type_map

    return triplet_type


def extract_points_from_state(R):
    return {i for i, v in enumerate(R.fix_s) if v == 1}


def reduce_point_modes_for_one_attr_solution(
    points_set, viol_triplets_attr, triplet_type_attr
):
    incident = {p: [] for p in points_set}
    for t in viol_triplets_attr:
        i, j, k = t
        for p in (i, j, k):
            if p in incident:
                incident[p].append(t)

    point_mode = {}

    for p, triplets in incident.items():
        need_pos = False
        need_attr = False

        for t in triplets:
            t_type = triplet_type_attr[t]
            if t_type in ("pos", "both"):
                need_pos = True
            if t_type in ("attr", "both"):
                need_attr = True

        if need_pos and need_attr:
            point_mode[p] = "both"
        elif need_pos:
            point_mode[p] = "pos"
        else:
            point_mode[p] = "attr"

    return point_mode


def main_precision_repair(df_std, pos_cols, attr_cols, attr_thresholds):
    n = len(df_std)
    viol_triplets = detect_violations(df_std, attr_cols, pos_cols, attr_thresholds)

    all_attr_repairs = {}

    for attr in attr_cols:
        trips_for_attr = viol_triplets[attr]

        point_loss, triplet_repr, C_global = build_candidate_points_by_triplet_loss(
            df=df_std,
            viol_triplets_attr=trips_for_attr,
            pos_cols=pos_cols,
            attr=attr,
            win_size=7,
        )

        groups = group_violation_triplets(trips_for_attr)
        split_ranges = get_split_ranges_for_groups(groups, n)

        group_repairs_with_ranges = []

        for group, (start, end) in zip(groups, split_ranges):
            local_n = end - start + 1

            C_g_global = {triplet_repr[(a, b, c)] for (a, b, c) in group}
            allowed_positions_local = {p - start for p in C_g_global}

            local_group = []
            for a, b, c in group:
                local_group.append((a - start, b - start, c - start))

            repairs = enumerate_repairs_for_attr(
                attr,
                {attr: local_group},
                local_n,
                allowed_positions=allowed_positions_local,
            )

            minimal_repairs = filter_minimal_repairs(repairs, local_group)
            group_repairs_with_ranges.append((minimal_repairs, (start, end)))

        final_repairs = merge_group_repairs_for_attr(group_repairs_with_ranges, n)
        all_attr_repairs[attr] = final_repairs

    triplet_type = classify_triplet_causes(
        df=df_std,
        pos_cols=pos_cols,
        attr_cols=attr_cols,
        viol_triplets=viol_triplets,
        eps=1e-9,
    )

    attr_repairs = {}
    attr_solution_modes = {}

    for attr in attr_cols:
        repairs_list = all_attr_repairs[attr]
        attr_repairs[attr] = repairs_list

        modes_list = []
        for R in repairs_list:
            S = extract_points_from_state(R)
            modes = reduce_point_modes_for_one_attr_solution(
                points_set=S,
                viol_triplets_attr=viol_triplets[attr],
                triplet_type_attr=triplet_type[attr],
            )
            modes_list.append(modes)

        attr_solution_modes[attr] = modes_list

    return {
        attr: {
            "repairs": attr_repairs[attr],
            "point_modes": attr_solution_modes[attr],
        }
        for attr in attr_cols
    }


def local_deltaA_diff(df, pos_cols, attr, a, b, c):
    pa = df.loc[a, pos_cols].values.astype(float)
    pb = df.loc[b, pos_cols].values.astype(float)
    pc = df.loc[c, pos_cols].values.astype(float)

    ds_ab = np.linalg.norm(pb - pa)
    ds_bc = np.linalg.norm(pc - pb)
    attr_ab = df.loc[b, attr] - df.loc[a, attr]
    attr_bc = df.loc[c, attr] - df.loc[b, attr]

    W_ab = 0.5 * attr_ab * ds_ab
    W_bc = 0.5 * attr_bc * ds_bc

    return abs(W_ab - W_bc)


def local_feasibility_check(df_tmp, pos_cols, attr, p, threshold, points_set):
    n = len(df_tmp)
    triplets = []

    if p - 2 >= 0:
        triplets.append((p - 2, p - 1, p))
    if p - 1 >= 0 and p + 1 < n:
        triplets.append((p - 1, p, p + 1))
    if p + 2 < n:
        triplets.append((p, p + 1, p + 2))

    for a, b, c in triplets:
        if any(q in points_set and q != p for q in (a, b, c)):
            continue

        diff = local_deltaA_diff(df_tmp, pos_cols, attr, a, b, c)
        if diff > threshold:
            return False

    return True


def _candidates_in_open_interval(L, R, max_candidates, eps=1e-12):
    if max_candidates <= 0:
        return []

    M = (L + R) / 2.0

    if max_candidates == 1:
        return [float(M)]

    cand = np.linspace(L + eps, R - eps, max_candidates + 2)[1:-1]
    idx = np.argmin(np.abs(cand - M))
    cand[idx] = M
    return [float(x) for x in cand]


def estimate_repair_ranges(
    df,
    attr,
    points_set,
    point_modes,
    pos_cols,
    attr_threshold,
    win_size=7,
    max_candidates=5,
):
    n = len(df)
    pos_arrays = {col: df[col].to_numpy(dtype=float) for col in pos_cols}
    attr_array = df[attr].to_numpy(dtype=float)
    clean_points = {i for i in range(n) if i not in points_set}

    correct_pairs = [
        (i, i + 1)
        for i in range(n - 1)
        if i in clean_points and (i + 1) in clean_points
    ]

    pos_theta = {}
    for col in pos_cols:
        arr = pos_arrays[col]
        diffs = [abs(arr[j] - arr[i]) for (i, j) in correct_pairs]
        pos_theta[col] = float(np.median(diffs)) if len(diffs) > 0 else 0.0

    diffs_attr = [abs(attr_array[j] - attr_array[i]) for (i, j) in correct_pairs]
    attr_theta = float(np.median(diffs_attr)) if len(diffs_attr) > 0 else 0.0

    left_clean = [None] * n
    right_clean = [None] * n

    last_clean = None
    for i in range(n):
        left_clean[i] = last_clean
        if i in clean_points:
            last_clean = i

    next_clean = None
    for i in range(n - 1, -1, -1):
        right_clean[i] = next_clean
        if i in clean_points:
            next_clean = i

    pos_ranges = {}
    attr_ranges = {}

    for p in points_set:
        mode = point_modes[p]
        left_n = left_clean[p]
        right_n = right_clean[p]
        if mode in ("pos", "both"):
            cands_per_col = []

            for col in pos_cols:
                if left_n is not None and right_n is not None:
                    L = float(pos_arrays[col][left_n])
                    R = float(pos_arrays[col][right_n])

                    if L == R:
                        cand = [L] * max_candidates
                    else:
                        if L > R:
                            L, R = R, L
                        cand = _candidates_in_open_interval(L, R, max_candidates)

                elif left_n is not None or right_n is not None:
                    ref_idx = left_n if left_n is not None else right_n
                    ref = float(pos_arrays[col][ref_idx])
                    theta = pos_theta[col]

                    if max_candidates == 1 or theta == 0:
                        cand = [ref]
                    else:
                        cand = [
                            float(x)
                            for x in np.linspace(
                                ref - theta, ref + theta, max_candidates
                            )
                        ]

                else:
                    ref = float(pos_arrays[col][p])
                    theta = pos_theta[col]

                    if max_candidates == 1 or theta == 0:
                        cand = [ref]
                    else:
                        cand = [
                            float(x)
                            for x in np.linspace(
                                ref - theta, ref + theta, max_candidates
                            )
                        ]

                if len(cand) == 1 and max_candidates > 1:
                    cand = cand * max_candidates

                cands_per_col.append(cand)

            num_pos_cands = min(len(c) for c in cands_per_col) if cands_per_col else 0
            pos_ranges[p] = [
                tuple(cands_per_col[d][t] for d in range(len(pos_cols)))
                for t in range(num_pos_cands)
            ]

        if mode in ("attr", "both"):
            if left_n is not None and right_n is not None:
                L = float(attr_array[left_n])
                R = float(attr_array[right_n])

                if L == R:
                    cand = [L]
                else:
                    if L > R:
                        L, R = R, L
                    cand = _candidates_in_open_interval(L, R, max_candidates)

            elif left_n is not None or right_n is not None:
                ref_idx = left_n if left_n is not None else right_n
                ref = float(attr_array[ref_idx])

                if max_candidates == 1 or attr_theta == 0:
                    cand = [ref]
                else:
                    cand = [
                        float(x)
                        for x in np.linspace(
                            ref - attr_theta, ref + attr_theta, max_candidates
                        )
                    ]
            else:
                ref = float(attr_array[p])

                if max_candidates == 1 or attr_theta == 0:
                    cand = [ref]
                else:
                    cand = [
                        float(x)
                        for x in np.linspace(
                            ref - attr_theta, ref + attr_theta, max_candidates
                        )
                    ]

            attr_ranges[p] = cand

    return pos_ranges, attr_ranges


def optimize_points_locally(
    df,
    attr,
    points_set,
    point_modes,
    pos_ranges,
    attr_ranges,
    pos_cols,
    attr_threshold,
    win_size=7,
):
    df_current = df.copy()
    total_loss = 0.0

    final_pos_values = {}
    final_attr_values = {}

    for p in sorted(points_set):
        mode = point_modes[p]

        best_loss = float("inf")
        best_pos = None
        best_attr = None

        if mode == "attr":
            candidates = [(None, v) for v in attr_ranges.get(p, [])]

        elif mode == "pos":
            candidates = [(xy, None) for xy in pos_ranges.get(p, [])]

        elif mode == "both":
            candidates = [
                (xy, v) for xy in pos_ranges.get(p, []) for v in attr_ranges.get(p, [])
            ]
        else:
            raise ValueError(f"Invalid mode: {mode}")

        for pos_val, attr_val in candidates:
            df_tmp = df_current.copy()

            if pos_val is not None:
                if len(pos_val) != len(pos_cols):
                    continue
                for d, col in enumerate(pos_cols):
                    df_tmp.loc[p, col] = pos_val[d]

            if attr_val is not None:
                df_tmp.loc[p, attr] = attr_val

            if not local_feasibility_check(
                df_tmp,
                pos_cols,
                attr,
                p,
                attr_threshold,
                points_set,
            ):
                continue

            loss = point_influence_loss(df_tmp, pos_cols, attr, p, win_size)

            if loss < best_loss:
                best_loss = loss
                best_pos = pos_val
                best_attr = attr_val

        if best_loss == float("inf"):
            best_pos = None
            best_attr = None
            best_loss = 0.0

        if best_pos is not None:
            for d, col in enumerate(pos_cols):
                df_current.loc[p, col] = best_pos[d]
            final_pos_values[p] = {col: best_pos[d] for d, col in enumerate(pos_cols)}

        if best_attr is not None:
            df_current.loc[p, attr] = best_attr
            final_attr_values[p] = best_attr

        total_loss += best_loss

    return {
        "pos_values": final_pos_values,
        "attr_values": final_attr_values,
        "total_loss": total_loss,
    }


def brute_force_exact_solution(
    df,
    attr,
    attr_info,
    pos_cols,
    attr_threshold,
    win_size=7,
):
    start_time = time.time()

    repairs_list = attr_info["repairs"]
    point_modes_list = attr_info["point_modes"]

    best_solution = None
    best_loss = float("inf")

    for idx, (R, point_modes) in enumerate(zip(repairs_list, point_modes_list)):
        repair_points = extract_points_from_state(R)
        pos_ranges, attr_ranges = estimate_repair_ranges(
            df=df,
            attr=attr,
            points_set=repair_points,
            point_modes=point_modes,
            pos_cols=pos_cols,
            attr_threshold=attr_threshold,
            win_size=win_size,
            max_candidates=5,
        )

        solution = optimize_points_locally(
            df=df,
            attr=attr,
            points_set=repair_points,
            point_modes=point_modes,
            pos_ranges=pos_ranges,
            attr_ranges=attr_ranges,
            pos_cols=pos_cols,
            attr_threshold=attr_threshold,
            win_size=win_size,
        )

        if solution is None:
            continue

        total_loss = solution["total_loss"]

        packed_solution = {
            "points": list(repair_points),
            "point_modes": point_modes,
            "pos_values": solution.get("pos_values", {}),
            "attr_values": solution.get("attr_values", {}),
            "total_loss": total_loss,
        }

        if total_loss < best_loss:
            best_loss = total_loss
            best_solution = packed_solution

    end_time = time.time()
    return best_solution


def apply_repair_to_dataframe(df, global_best, pos_cols, attr):
    df_repaired = df.copy()

    pos_values = global_best.get("pos_values", {})
    for idx, pos_dict in pos_values.items():
        for col, val in pos_dict.items():
            df_repaired.loc[idx, col] = val

    attr_values = global_best.get("attr_values", {})
    for idx, val in attr_values.items():
        df_repaired.loc[idx, attr] = val

    return df_repaired


def fuse_attr_solutions_with_shared_positions(
    df_original,
    final_attr_solutions,
    pos_cols,
    attr_cols,
    eps=1e-8,
):
    n = len(df_original)
    pos_repair_log = defaultdict(list)

    for attr in attr_cols:
        sol = final_attr_solutions.get(attr, None)
        if sol is None:
            continue

        pos_values = sol.get("pos_values", {})
        for idx, pos_dict in pos_values.items():
            pos_vec = np.array([float(pos_dict[col]) for col in pos_cols], dtype=float)
            pos_repair_log[idx].append(pos_vec)

    df_geo = df_original.copy()

    for idx, pos_list in pos_repair_log.items():
        avg_pos = np.mean(pos_list, axis=0)
        for d, col in enumerate(pos_cols):
            df_geo.loc[idx, col] = avg_pos[d]
    df_final = df_geo.copy()

    for attr in attr_cols:
        sol = final_attr_solutions.get(attr, None)
        if sol is None:
            continue

        attr_values = sol.get("attr_values", {})
        for idx, val in attr_values.items():
            df_final.loc[idx, attr] = float(val)

    return df_geo, df_final, pos_repair_log


def choose_patch_points_from_violations_by_rank(
    df,
    pos_cols,
    attr,
    trips,
    rank_idx=0,
    win_size=7,
):
    patch_points = set()

    for a, b, c in trips:
        cand = [
            (a, point_influence_loss(df, pos_cols, attr, a, win_size)),
            (b, point_influence_loss(df, pos_cols, attr, b, win_size)),
            (c, point_influence_loss(df, pos_cols, attr, c, win_size)),
        ]
        cand.sort(key=lambda x: x[1], reverse=True)
        chosen_idx = min(rank_idx, len(cand) - 1)
        patch_points.add(cand[chosen_idx][0])

    return patch_points


def build_attr_only_candidate_ranges(
    df,
    attr,
    patch_points,
    pos_cols,
    attr_threshold,
    max_candidates=5,
):
    dummy_point_modes = {p: "attr" for p in patch_points}

    _, attr_ranges = estimate_repair_ranges(
        df=df,
        attr=attr,
        points_set=patch_points,
        point_modes=dummy_point_modes,
        pos_cols=pos_cols,
        attr_threshold=attr_threshold,
        win_size=7,
        max_candidates=max_candidates,
    )
    return attr_ranges


def patch_after_shared_fusion(
    df_repaired,
    pos_cols,
    attr_cols,
    attr_thresholds,
    max_candidates=5,
    win_size=7,
    max_iter_per_attr=3,
):
    df_patch = df_repaired.copy()

    for attr in attr_cols:
        threshold = attr_thresholds[attr]

        for it in range(max_iter_per_attr):
            rank_idx = it

            viol = detect_violations(
                df_patch.copy(),
                [attr],
                pos_cols,
                {attr: threshold},
            )
            trips = viol.get(attr, [])

            if len(trips) == 0:
                break

            patch_points = choose_patch_points_from_violations_by_rank(
                df=df_patch,
                pos_cols=pos_cols,
                attr=attr,
                trips=trips,
                rank_idx=rank_idx,
                win_size=win_size,
            )

            if len(patch_points) == 0:
                continue

            attr_ranges = build_attr_only_candidate_ranges(
                df=df_patch,
                attr=attr,
                patch_points=patch_points,
                pos_cols=pos_cols,
                attr_threshold=threshold,
                max_candidates=max_candidates,
            )

            changed = False

            for p in sorted(patch_points):
                candidates = attr_ranges.get(p, [])
                if len(candidates) == 0:
                    continue

                best_val = None
                best_loss = float("inf")
                old_val = float(df_patch.loc[p, attr])

                for v in candidates:
                    df_tmp = df_patch.copy()
                    df_tmp.loc[p, attr] = float(v)

                    if not local_feasibility_check(
                        df_tmp,
                        pos_cols,
                        attr,
                        p,
                        threshold,
                        {p},
                    ):
                        continue

                    loss = point_influence_loss(
                        df_tmp,
                        pos_cols,
                        attr,
                        p,
                        win_size=win_size,
                    )

                    if loss < best_loss:
                        best_loss = loss
                        best_val = float(v)

                if best_val is not None and abs(best_val - old_val) > 1e-12:
                    df_patch.loc[p, attr] = best_val
                    changed = True

            if not changed:
                continue

        final_viol = detect_violations(
            df_patch.copy(),
            [attr],
            pos_cols,
            {attr: threshold},
        )

    return df_patch
