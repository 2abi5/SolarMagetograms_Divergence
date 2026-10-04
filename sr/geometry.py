"""
Viewing geometry as network input, and vector-aware augmentation (pilot v2;
following Munoz-Jaramillo et al. 2024, who give the network a "location channel"
and augment with polarity flips and N-S / E-W reflections).

For a vector target, the radial distance alone is not enough: MDI measures
    B_LOS = c_p * Bp + c_t * Bt + c_r * Br
and the three coefficients depend on the pixel's position relative to the MDI
observer. `region_los_coefficients` returns (c_p, c_t, c_r) at every MDI (LR)
pixel of a saved region; c_r is the cosine of the heliocentric angle (Munoz's
location information), c_p and c_t add the direction. Conventions as in
data_active_region_created.los_projection (Bp westward, Bt southward, Br radial,
positive toward the observer). The LR WCS stored with each region is the MDI grid
after alignment refinement, so the geometry is evaluated where the MDI pixels are.

Augmentation keeps B_LOS = c . B consistent:
    polarity flip   MDI -> -MDI, (Bp, Bt, Br) -> -(Bp, Bt, Br), geometry unchanged
    E-W mirror (x)  flip columns, Bp -> -Bp, c_p -> -c_p   (dlon -> -dlon)
    N-S mirror (y)  flip rows,    Bt -> -Bt, c_t -> -c_t   (lat, B0 -> -lat, -B0)
Both mirrors map the horizontal divergence of the target onto itself (mirrored).
"""
import os

import numpy as np


def los_coefficients(dlon, lat, b0):
    """(c_p, c_t, c_r) for longitude from the observer's central meridian `dlon`,
    latitude `lat` and observer latitude `b0` (radians)."""
    cl, sl = np.cos(lat), np.sin(lat)
    cd, sd = np.cos(dlon), np.sin(dlon)
    cb, sb = np.cos(b0), np.sin(b0)
    return np.stack([-sd * cb, sl * cd * cb - cl * sb, cl * cd * cb + sl * sb])


def region_los_coefficients(region_path):
    """(3, ny, nx) float32 (c_p, c_t, c_r) at the MDI pixels of a saved region file."""
    import astropy.io.fits as fits
    import astropy.units as u
    import sunpy.coordinates  # noqa: F401  (registers the solar frames)
    from astropy.time import Time
    from astropy.wcs import WCS

    with np.load(region_path, allow_pickle=False) as d:
        header = fits.Header.fromstring(str(d["lr_wcs_header"]))
        ny, nx = d["lr"].shape
        obs_lon, obs_lat = float(d["mdi_obs_lon_deg"]), float(d["mdi_obs_lat_deg"])
        t = str(d["mdi_t_rec"]).replace("_TAI", "")
    stamp = Time(t.replace(".", "-", 2).replace("_", "T"), scale="tai")
    yy, xx = np.mgrid[0:ny, 0:nx]
    world = WCS(header).pixel_to_world(xx, yy)
    hgs = world.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst(obstime=stamp))
    dlon = np.deg2rad(hgs.lon.to_value(u.deg) - obs_lon)
    lat = np.deg2rad(hgs.lat.to_value(u.deg))
    return los_coefficients(dlon, lat, np.deg2rad(obs_lat)).astype(np.float32)


def geometry_path(geometry_dir, region_file):
    return os.path.join(geometry_dir, os.path.splitext(os.path.basename(region_file))[0] + ".npy")


def augment(lr, hr, flip=False, mirror_x=False, mirror_y=False):
    """Vector-aware augmentation of one pair. lr (C, h, w): channel 0 = MDI, channels 1-3
    (if present) = (c_p, c_t, c_r); hr (3, H, W) = (Bp, Bt, Br). Returns new arrays."""
    lr, hr = np.array(lr, copy=True), np.array(hr, copy=True)
    geom = lr.shape[0] >= 4
    if mirror_x:
        lr, hr = lr[..., ::-1], hr[..., ::-1]
        hr[0] = -hr[0]
        if geom:
            lr[1] = -lr[1]
    if mirror_y:
        lr, hr = lr[..., ::-1, :], hr[..., ::-1, :]
        hr[1] = -hr[1]
        if geom:
            lr[2] = -lr[2]
    if flip:
        lr[0] = -lr[0]
        hr = -hr
    return np.ascontiguousarray(lr), np.ascontiguousarray(hr)
