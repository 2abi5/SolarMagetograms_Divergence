"""
split_dataset.py -- region-level train/val/test split with no leakage.

Patches overlap (stride 8 < size 16) and consecutive 96 min frames of a region
are near duplicates, so the split is by REGION GROUP, never by patch or frame.
A region that rotates off the disk and returns ~27 days later gets a new
HARPNUM, so HARPs that share a NOAA active-region number are merged into one
group (transitively). NOAA also renumbers a returning region, so HARPs are
additionally linked when one reappears at the same Carrington position
(|dlon| <= 15 deg, |dlat| <= 10 deg) 5-40 days after the other was last seen
(far-side transit ~14 days plus limb margins); these links are listed in the
output. HARPs with neither link are their own group. Groups are assigned to train/val/test (70/15/15 of patches) by a seeded
randomised greedy search that picks the most balanced of many candidate
assignments.

Usage (after `source env.sh`):
    python pipeline/split_dataset.py --run pilot --seed 0
    -> splits/pilot.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ("train", "val", "test")


def returning_links(pos, max_dlon=15.0, max_dlat=10.0, min_gap_days=5.0, max_gap_days=40.0):
    """(earlier HARP, later HARP) pairs that look like one region returning:
    the later one starts 5-40 days after the earlier one was last seen, at the
    same Carrington longitude (wrapped) and latitude within the tolerances.
    pos: one row per HARP with carr_lon, carr_lat (deg), t_first, t_last."""
    rows = pos.to_dict("records")
    links = []
    for a in rows:
        for b in rows:
            if a["harpnum"] == b["harpnum"]:
                continue
            gap = (b["t_first"] - a["t_last"]).total_seconds() / 86400.0
            if not (min_gap_days <= gap <= max_gap_days):
                continue
            dlon = abs((b["carr_lon"] - a["carr_lon"] + 180.0) % 360.0 - 180.0)
            if dlon <= max_dlon and abs(b["carr_lat"] - a["carr_lat"]) <= max_dlat:
                links.append((int(a["harpnum"]), int(b["harpnum"])))
    return sorted(links)


def harp_positions(reg, run_dir):
    """Per HARP: circular-mean Carrington longitude and median latitude of the
    CEA reference point (CRVAL1/2 from the saved SHARP header), first/last time."""
    rows = []
    for r in reg.itertuples():
        with np.load(os.path.join(run_dir, r.region_file), allow_pickle=False) as d:
            h = json.loads(str(d["sharp_header_json"]))
        t = pd.Timestamp(r.mdi_t_rec.replace("_TAI", "").replace(".", "-", 2).replace("_", " "))
        rows.append({"harpnum": int(r.harpnum), "lon": float(h["crval1"]), "lat": float(h["crval2"]), "t": t})
    df = pd.DataFrame(rows)
    out = []
    for hnum, g in df.groupby("harpnum"):
        ang = np.deg2rad(g.lon)
        lon = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0)
        out.append({"harpnum": int(hnum), "carr_lon": lon, "carr_lat": float(g.lat.median()),
                    "t_first": g.t.min(), "t_last": g.t.max()})
    return pd.DataFrame(out)


def region_groups(reg, extra_links=()):
    """harpnum -> group id. HARPs connected through a shared nonzero NOAA_AR or
    an extra (e.g. returning-region) link form one group ("noaa:<smallest AR>"
    if the group has an AR, else "harp:<smallest HARPNUM>")."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for h in reg.harpnum.unique():
        find(("harp", int(h)))
    for h, a in reg[["harpnum", "noaa_ar"]].drop_duplicates().itertuples(index=False):
        if pd.notna(a) and int(a) > 0:
            union(("harp", int(h)), ("ar", int(a)))
    for h1, h2 in extra_links:
        if ("harp", int(h1)) in parent and ("harp", int(h2)) in parent:
            union(("harp", int(h1)), ("harp", int(h2)))

    members = {}
    for node in list(parent):
        members.setdefault(find(node), []).append(node)
    groups = {}
    for nodes in members.values():
        ars = sorted(n[1] for n in nodes if n[0] == "ar")
        harps = sorted(n[1] for n in nodes if n[0] == "harp")
        gid = f"noaa:{ars[0]}" if ars else f"harp:{harps[0]}"
        for h in harps:
            groups[h] = gid
    return groups


def shared_noaa_report(reg):
    """{NOAA_AR: [HARPNUMs]} for every AR carried by more than one HARP."""
    out = {}
    for a, sub in reg[reg.noaa_ar.fillna(0).astype(int) > 0].groupby("noaa_ar"):
        harps = sorted(int(h) for h in sub.harpnum.unique())
        if len(harps) > 1:
            out[int(a)] = harps
    return out


def month_of(reg):
    """'YYYY.MM' of each row's MDI T_REC (e.g. 2011.02.15_09:36:00_TAI -> 2011.02)."""
    return reg.mdi_t_rec.astype(str).str.slice(0, 7)


def make_split(reg, seed=0, fractions=(0.70, 0.15, 0.15), min_groups=2, n_candidates=500, links=(),
               stratify=None):
    """Assign region groups to train/val/test, balanced by patch count.

    Each candidate visits the groups in a random order and gives each one to
    the split furthest below its target patch count. The candidate with the
    smallest worst-case relative deviation (with at least `min_groups` groups
    in val and test) wins. Deterministic for a given seed.

    stratify="<column>" (Phase 8: months) also balances every stratum: each group
    goes to the split that most lacks patches in the group's own strata, and the
    score adds the mean over strata of the fraction of patches misallocated.
    """
    if stratify is not None:
        return _make_stratified_split(reg, seed, fractions, min_groups, n_candidates, links, stratify)
    groups = region_groups(reg, links)
    reg = reg.assign(group=reg.harpnum.map(groups))
    g_patches = reg.groupby("group").n_patches.sum()
    names = sorted(g_patches.index)
    sizes = np.array([int(g_patches[g]) for g in names])
    total = sizes.sum()
    targets = np.array(fractions) * total
    rng = np.random.default_rng(seed)

    best, best_score = None, np.inf
    for _ in range(n_candidates):
        load = np.zeros(3)
        count = np.zeros(3, int)
        assign = np.empty(len(names), int)
        for i in rng.permutation(len(names)):
            s = int(np.argmax(targets - load))
            assign[i] = s
            load[s] += sizes[i]
            count[s] += 1
        if count[1] < min_groups or count[2] < min_groups:
            continue
        score = np.max(np.abs(load - targets) / targets)
        if score < best_score:
            best, best_score = assign.copy(), score
    if best is None:
        raise ValueError(f"cannot give val and test {min_groups} groups each from {len(names)} groups")

    out = {}
    for s_idx, s in enumerate(SPLITS):
        chosen = [names[i] for i in range(len(names)) if best[i] == s_idx]
        sub = reg[reg.group.isin(chosen)]
        out[s] = {"groups": sorted(chosen),
                  "harps": sorted(int(h) for h in sub.harpnum.unique()),
                  "pairs": int(len(sub)),
                  "patches": int(sub.n_patches.sum())}
    return out


def _make_stratified_split(reg, seed, fractions, min_groups, n_candidates, links, stratify):
    groups = region_groups(reg, links)
    reg = reg.assign(group=reg.harpnum.map(groups))
    tab = reg.pivot_table(index="group", columns=stratify, values="n_patches", aggfunc="sum", fill_value=0)
    names = sorted(tab.index)
    M = tab.reindex(names).to_numpy(float)                     # (groups, strata)
    sizes = M.sum(axis=1)
    dist = M / np.maximum(sizes[:, None], 1.0)
    targets = np.array(fractions) * sizes.sum()
    T = np.outer(fractions, M.sum(axis=0))                     # (3, strata)
    rng = np.random.default_rng(seed)

    best, best_score = None, np.inf
    for _ in range(n_candidates):
        L = np.zeros_like(T)
        count = np.zeros(3, int)
        assign = np.empty(len(names), int)
        for i in rng.permutation(len(names)):
            need = ((T - L) / np.maximum(T, 1.0)) @ dist[i]
            s = int(np.argmax(need))
            assign[i] = s
            L[s] += M[i]
            count[s] += 1
        if count[1] < min_groups or count[2] < min_groups:
            continue
        load = L.sum(axis=1)
        score = (np.max(np.abs(load - targets) / targets)
                 + np.mean(0.5 * np.abs(L - T).sum(axis=0) / np.maximum(M.sum(axis=0), 1.0)))
        if score < best_score:
            best, best_score = assign.copy(), score
    if best is None:
        raise ValueError(f"cannot give val and test {min_groups} groups each from {len(names)} groups")

    out = {}
    for s_idx, s in enumerate(SPLITS):
        chosen = [names[i] for i in range(len(names)) if best[i] == s_idx]
        sub = reg[reg.group.isin(chosen)]
        out[s] = {"groups": sorted(chosen),
                  "harps": sorted(int(h) for h in sub.harpnum.unique()),
                  "pairs": int(len(sub)),
                  "patches": int(sub.n_patches.sum())}
    return out


def write_split(reg, path, seed=0, run="", fractions=(0.70, 0.15, 0.15), min_groups=2, links=(),
                stratify=None):
    split = make_split(reg, seed=seed, fractions=fractions, min_groups=min_groups, links=links, stratify=stratify)
    groups = region_groups(reg, links)
    total = sum(split[s]["patches"] for s in SPLITS)
    train = reg[reg.harpnum.isin(split["train"]["harps"])]
    per_harp_train = train.groupby("harpnum").n_patches.sum().sort_values(ascending=False)
    info = {
        "run": run, "seed": int(seed), "fractions": list(fractions),
        "grouping": "HARPs sharing a nonzero NOAA_AR are merged (transitively); "
                    "otherwise one group per HARPNUM",
        "splits": split,
        "patch_fraction": {s: round(split[s]["patches"] / total, 4) for s in SPLITS},
        "groups": {g: sorted(int(h) for h, gg in groups.items() if gg == g)
                   for g in sorted(set(groups.values()))},
        "shared_noaa": {str(a): hs for a, hs in shared_noaa_report(reg).items()},
        "returning_links": [list(map(int, l)) for l in links],
        **({"stratified_by": stratify,
            "stratum_patch_fraction": {
                s: {str(k): round(float(v), 4) for k, v in
                    (reg[reg.harpnum.isin(split[s]["harps"])].groupby(stratify).n_patches.sum()
                     / reg.groupby(stratify).n_patches.sum()).fillna(0).items()}
                for s in SPLITS}} if stratify else {}),
        "largest_train_region": {"harpnum": int(per_harp_train.index[0]),
                                 "share_of_train_patches": round(float(per_harp_train.iloc[0])
                                                                 / float(per_harp_train.sum()), 4)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=1)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: splits/<run>.json")
    ap.add_argument("--min_groups", type=int, default=2, help="minimum region groups in val and in test")
    ap.add_argument("--stratify_months", action="store_true",
                    help="Phase 8: also balance every calendar month (of the MDI time) across the splits")
    args = ap.parse_args(argv)
    run_dir = os.path.join(args.data_root, args.run)
    reg = pd.read_csv(os.path.join(run_dir, "regions.csv"))
    if args.stratify_months:
        reg = reg.assign(month=month_of(reg))
    out = args.out or os.path.join(ROOT, "splits", f"{args.run}.json")
    pos = harp_positions(reg, run_dir)
    links = returning_links(pos)
    info = write_split(reg, out, seed=args.seed, run=args.run, min_groups=args.min_groups, links=links,
                       stratify="month" if args.stratify_months else None)
    print(f"Wrote {out}")
    for s in SPLITS:
        d = info["splits"][s]
        print(f"  {s:5s}: {len(d['groups']):3d} groups, {len(d['harps']):3d} HARPs, "
              f"{d['pairs']:5d} (region, time) pairs, {d['patches']:6d} patches "
              f"({info['patch_fraction'][s]:.1%})")
    print(f"  HARPs sharing a NOAA AR: {info['shared_noaa'] or 'none'}")
    print(f"  returning-region links (same Carrington position, 5-40 d apart): {links or 'none'}")
    lt = info["largest_train_region"]
    flag = "  -> above 25%: use the per-region capped sampler" if lt["share_of_train_patches"] > 0.25 else ""
    print(f"  largest training region: HARP {lt['harpnum']} with "
          f"{lt['share_of_train_patches']:.1%} of training patches{flag}")


if __name__ == "__main__":
    main()
