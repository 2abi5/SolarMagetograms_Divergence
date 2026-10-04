"""
Datasets for patches and full regions (training_plan.md Phase 4, section 2b).

Arrays on disk are normalised (/3500) with NaNs where data is missing. Items
return NaN-free tensors (NaN -> 0) plus boolean masks, so NaNs never enter a
network and every loss/metric can ignore invalid pixels:

    lr       (1, h, w)    MDI LOS                          float32
    hr       (3, 4h, 4w)  SHARP Bp, Bt, Br                 float32
    mask     (1, 4h, 4w)  True where all three HR channels are finite
    lr_mask  (1, h, w)    True where the LR pixel is finite
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


PATCH_LR = 16


def split_harps(split_json, split):
    with open(split_json) as f:
        return [int(h) for h in json.load(f)["splits"][split]["harps"]]


def _to_item(lr, hr):
    lr_mask = np.isfinite(lr)
    mask = np.isfinite(hr).all(axis=0)
    return {"lr": torch.from_numpy(np.nan_to_num(lr, nan=0.0)[None].astype(np.float32)),
            "hr": torch.from_numpy(np.nan_to_num(hr, nan=0.0).astype(np.float32)),
            "mask": torch.from_numpy(mask[None]),
            "lr_mask": torch.from_numpy(lr_mask[None])}


def _load_pair(path):
    with np.load(path, allow_pickle=False) as d:
        return d["lr"].astype(np.float32), d["hr"].astype(np.float32)


class PatchDataset(Dataset):
    """Patches listed in <run_dir>/manifest.csv, optionally restricted to HARPs.
    cache=True loads every patch into RAM once (tens of thousands of small
    files; RAM is not a constraint on this server)."""

    def __init__(self, run_dir, harps=None, manifest="manifest.csv", cache=True, n_threads=16,
                 geometry_dir=None, augment=False, seed=0):
        rows = pd.read_csv(os.path.join(run_dir, manifest))
        if harps is not None:
            rows = rows[rows.harpnum.isin(list(harps))]
        self.rows = rows.reset_index(drop=True)
        self.run_dir = run_dir
        self.harpnums = self.rows.harpnum.to_numpy()
        self._lr = self._hr = None
        # pilot v2 (sr/geometry.py): viewing-geometry channels and vector-aware augmentation
        self._geo = None
        if geometry_dir is not None and len(self.rows):
            from sr.geometry import geometry_path
            regs = pd.read_csv(os.path.join(run_dir, "regions.csv"))
            region_of = dict(zip(zip(regs.harpnum, regs.mdi_t_rec.astype(str)), regs.region_file))
            maps, geo = {}, []
            for r in self.rows.itertuples():
                f = region_of[(r.harpnum, str(r.mdi_t_rec))]
                if f not in maps:
                    maps[f] = np.load(geometry_path(geometry_dir, f))
                geo.append(maps[f][:, r.row:r.row + PATCH_LR, r.col:r.col + PATCH_LR])
            self._geo = np.stack(geo).astype(np.float32)
        self.augment, self.seed, self.epoch = bool(augment), int(seed), 0
        if cache and len(self.rows):
            paths = [os.path.join(run_dir, p) for p in self.rows.patch_file]
            with ThreadPoolExecutor(n_threads) as ex:
                pairs = list(ex.map(_load_pair, paths))
            self._lr = np.stack([p[0] for p in pairs])
            self._hr = np.stack([p[1] for p in pairs])

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, i):
        if self._lr is not None:
            lr, hr = self._lr[i], self._hr[i]
        else:
            lr, hr = _load_pair(os.path.join(self.run_dir, self.rows.patch_file[i]))
        if self._geo is None and not self.augment:
            item = _to_item(lr, hr)
        else:
            x = lr[None] if self._geo is None else np.concatenate([lr[None], self._geo[i]])
            if self.augment:
                from sr.geometry import augment
                flip, mx, my = np.random.default_rng([self.seed, self.epoch, int(i)]).random(3) < 0.5
                x, hr = augment(x, hr, flip=flip, mirror_x=mx, mirror_y=my)
            item = _to_item(x[0], hr)
            item["lr"] = torch.from_numpy(np.nan_to_num(x, nan=0.0).astype(np.float32))
        item["harpnum"] = int(self.harpnums[i])
        item["index"] = i
        return item


class RegionDataset(Dataset):
    """Full aligned regions listed in <run_dir>/regions.csv (for evaluation)."""

    def __init__(self, run_dir, harps=None, regions="regions.csv"):
        rows = pd.read_csv(os.path.join(run_dir, regions))
        if harps is not None:
            rows = rows[rows.harpnum.isin(list(harps))]
        self.rows = rows.reset_index(drop=True)
        self.run_dir = run_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        lr, hr = _load_pair(os.path.join(self.run_dir, row.region_file))
        item = _to_item(lr, hr)
        item.update(harpnum=int(row.harpnum), mdi_t_rec=str(row.mdi_t_rec),
                    region_file=str(row.region_file), index=i)
        return item


def capped_region_weights(harpnums, max_share=0.25):
    """Per-sample weights such that no region's share of the sampled patches
    exceeds max_share; the excess is redistributed to the other regions in
    proportion to their patch counts (iterated until nothing exceeds the cap).
    Uniform weights if no region exceeds max_share."""
    harpnums = np.asarray(harpnums)
    regions, counts = np.unique(harpnums, return_counts=True)
    share = capped_shares(counts, max_share)
    per_region = dict(zip(regions, share / counts))
    return np.array([per_region[h] for h in harpnums])


def capped_shares(counts, max_share=0.25):
    """Shares proportional to `counts`, with no group above max_share; the excess
    goes to the other groups in proportion to their counts (iterated)."""
    counts = np.asarray(counts, float)
    share = counts / counts.sum()
    capped = np.zeros(len(counts), bool)
    while True:
        over = (share > max_share + 1e-12) & ~capped
        if not over.any():
            break
        capped |= over
        free = 1.0 - max_share * capped.sum()
        rest = counts * ~capped
        share = np.where(capped, max_share, free * rest / max(rest.sum(), 1))
    return share


def capped_region_sampler(harpnums, max_share=0.25, seed=0):
    w = capped_region_weights(harpnums, max_share)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(w),
                                 replacement=True, generator=g)


# --------------------------------------------------------------------------
# Crops cut on the fly from full regions (Phase 8: datasets built with
# --save_mode regions, so no patch files exist)
# --------------------------------------------------------------------------

def _window_sums(a, k):
    """Sum over every k x k window of a 2D integer array (valid positions only)."""
    s = np.pad(a.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]


def valid_crop_positions(lr, hr, patch_lr, scale=4, min_finite=0.95):
    """(row, col) LR offsets of every patch_lr x patch_lr crop that passes the
    pipeline's validity rule (extract_patches: >= min_finite finite in the LR crop
    and in the 3-channel HR crop), at every offset, in row-major order."""
    ny, nx = lr.shape
    if ny < patch_lr or nx < patch_lr:
        return np.zeros((0, 2), int)
    k = patch_lr * scale
    lr_ok = _window_sums(np.isfinite(lr), patch_lr) / (patch_lr * patch_lr) >= min_finite
    hr_fin = np.isfinite(hr).sum(axis=0)
    if hr_fin.shape[0] < k or hr_fin.shape[1] < k:
        return np.zeros((0, 2), int)
    hr_ok = (_window_sums(hr_fin, k) / (hr.shape[0] * k * k) >= min_finite)[::scale, ::scale]
    r, c = min(lr_ok.shape[0], hr_ok.shape[0]), min(lr_ok.shape[1], hr_ok.shape[1])
    return np.argwhere(lr_ok[:r, :c] & hr_ok[:r, :c])


class _RegionCache:
    def __init__(self, run_dir, harps, patch_lr, scale, regions, n_threads, geometry_dir=None, every=1):
        rows = pd.read_csv(os.path.join(run_dir, regions))
        if harps is not None:
            rows = rows[rows.harpnum.isin(list(harps))]
        if every > 1:   # every N-th frame of each HARP in time order (consecutive frames are near-duplicates)
            rows = pd.concat([g.sort_values("mdi_t_rec").iloc[::every] for _, g in rows.groupby("harpnum", sort=True)])
        rows = rows.reset_index(drop=True)
        paths = [os.path.join(run_dir, f) for f in rows.region_file]
        with ThreadPoolExecutor(n_threads) as ex:
            pairs = list(ex.map(_load_pair, paths))
        pos = [valid_crop_positions(lr, hr, patch_lr, scale) for lr, hr in pairs]
        keep = [i for i, p in enumerate(pos) if len(p)]
        self.rows = rows.iloc[keep].reset_index(drop=True)
        self.pairs = [pairs[i] for i in keep]
        self.positions = [pos[i] for i in keep]
        self.run_dir, self.patch_lr, self.scale = run_dir, patch_lr, scale
        self.geometry = None
        if geometry_dir is not None:
            from sr.geometry import geometry_path
            self.geometry = [np.load(geometry_path(geometry_dir, f)) for f in self.rows.region_file]

    def crop(self, k, r, c, index, augment_flags=None):
        lr, hr = self.pairs[k]
        p, s = self.patch_lr, self.scale
        lr_c, hr_c = lr[r:r + p, c:c + p], hr[:, s * r:s * r + s * p, s * c:s * c + s * p]
        if self.geometry is None and augment_flags is None:
            item = _to_item(lr_c, hr_c)
        else:
            x = lr_c[None] if self.geometry is None else np.concatenate([lr_c[None], self.geometry[k][:, r:r + p, c:c + p]])
            if augment_flags is not None:
                from sr.geometry import augment
                x, hr_c = augment(x, hr_c, *augment_flags)
            item = _to_item(x[0], hr_c)
            item["lr"] = torch.from_numpy(np.nan_to_num(x, nan=0.0).astype(np.float32))
        item.update(harpnum=int(self.rows.harpnum[k]), region=k, row=int(r), col=int(c), index=index)
        return item


class RegionGridDataset(Dataset):
    """Fixed grid crops (stride `stride`) from full regions: the same patches the
    pipeline writes with --save_mode patches. Used for validation in regions mode."""

    def __init__(self, run_dir, harps=None, patch_lr=16, stride=8, scale=4, regions="regions.csv", n_threads=16,
                 geometry_dir=None, every=1):
        self.cache = _RegionCache(run_dir, harps, patch_lr, scale, regions, n_threads, geometry_dir, every)
        self.rows = self.cache.rows
        self.index = [(k, r, c) for k, p in enumerate(self.cache.positions)
                      for r, c in p if r % stride == 0 and c % stride == 0]
        self.harpnums = np.array([int(self.rows.harpnum[k]) for k, _, _ in self.index])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return self.cache.crop(*self.index[i], index=i)


class RegionCropDataset(Dataset):
    """Random patch_lr x patch_lr crops (with the matching HR crop) cut on the fly
    from full regions (training_plan.md Phase 8). Item i of epoch e is drawn with
    a generator seeded by (seed, e, i): a region with probability proportional to
    its number of valid crop positions (per-HARP shares capped at max_share if
    given), then a uniformly random valid position. `samples_per_epoch` defaults
    to the number of stride-8 grid patches, i.e. the size of a patch-mode epoch."""

    def __init__(self, run_dir, harps=None, patch_lr=16, scale=4, samples_per_epoch=None, max_share=None,
                 seed=0, regions="regions.csv", n_threads=16, grid_stride=8, geometry_dir=None, augment=False):
        self.cache = _RegionCache(run_dir, harps, patch_lr, scale, regions, n_threads, geometry_dir)
        self.augment = bool(augment)
        self.rows, self.positions = self.cache.rows, self.cache.positions
        n = np.array([len(p) for p in self.positions], float)
        harps_k = self.rows.harpnum.to_numpy()
        groups, inv = np.unique(harps_k, return_inverse=True)
        totals = np.bincount(inv, weights=n)
        share = capped_shares(totals, max_share) if max_share else totals / totals.sum()
        prob = share[inv] * n / totals[inv]
        self.cum = np.cumsum(prob / prob.sum())
        self.cum[-1] = 1.0
        self.harpnums = harps_k
        if samples_per_epoch is None:
            samples_per_epoch = int(sum(((p[:, 0] % grid_stride == 0) & (p[:, 1] % grid_stride == 0)).sum()
                                        for p in self.positions))
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed, self.epoch = int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples_per_epoch

    def draw(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, int(i)])
        k = min(int(np.searchsorted(self.cum, rng.random(), side="right")), len(self.cum) - 1)
        r, c = self.positions[k][rng.integers(len(self.positions[k]))]
        return k, int(r), int(c)

    def __getitem__(self, i):
        flags = None
        if self.augment:   # own stream, so the drawn position is the same with and without augmentation
            flags = tuple(np.random.default_rng([self.seed, self.epoch, int(i), 1]).random(3) < 0.5)
        return self.cache.crop(*self.draw(i), index=i, augment_flags=flags)
