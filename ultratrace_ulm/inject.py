"""Synthetic-bubble injection and null-data construction (J3, bead mb-fmj).

WHY THIS EXISTS
---------------
This dataset has no ground truth, and the two validation habits we fell back on
have both failed:

* "matches the reference" is circular -- we already reproduce the reference
  (1,451 vs 1,421 tracks >= 35) and the reference is not ground truth;
* held-out position prediction is beaten by tracker-free interpolation, so it
  does not measure the tracker.

The remaining GT-free option is to *make* ground truth: add bubble echoes we
authored, with known trajectories, into the beamformed **complex** volume before
clutter filtering, so they traverse the same SVD, the same normalization and the
same detector as real ones. Recovery of those injected bubbles is then a real
detection rate, stratifiable by speed, depth and SNR.

The companion piece is null data -- material that *cannot* contain a bubble but
whose statistics are realistic -- so a false-alarm rate can be calibrated. Every
comparison this instrument powers is made at **matched detection density** or
**matched false-alarm rate**, never at a matched threshold: J2 produced a garbage
result precisely by comparing two arms at a fixed 2-sigma cut when their noise
floors differed, and that class of error has to be impossible here.

THE SIGNAL MODEL
----------------
A point scatterer at continuous position ``p`` contributes, at output voxel
``r`` of a baseband-beamformed volume,

    s(r) = A . h(r - p),     h(u) = env(u) . exp(i . k_z . u_z)

The beamformer applies the focusing phase ``exp(+i.2k.z)`` for output voxel
``z`` while the echo carries ``exp(-i.2k.z0)`` from the scatterer at ``z0``, so
the residual phase depends only on ``z - z0``: the complex PSF carries an axial
carrier at ``k_z ~ 2k``.  Two consequences we rely on:

1. **Doppler is automatic.** At a *fixed* voxel the phase is ``k_z.(z - z0(t))``,
   which rotates at ``k_z.v_z`` as the scatterer moves -- exactly the Doppler
   rate that decides where a bubble lands in the SVD's temporal spectrum, i.e.
   whether the clutter filter eats it. Nothing extra has to be bolted on, and a
   slow bubble is correctly hard.
2. **We must not assume the shape.** ``h`` is measured from the data
   (:func:`estimate_complex_psf`) rather than modelled, because elevation is the
   weak axis here (0.5547 mm voxels synthesized from 8 physical rows, against
   0.2 mm = lambda/4 in-plane) and no analytic Gaussian we could write down
   would get that anisotropy right.

At lambda/4 axial sampling the carrier sits at ~pi rad/voxel, i.e. at the
aliasing limit, so its *sign* is not identifiable. That is harmless: we estimate
``k_z`` from the data and use the same estimate everywhere, and the detector
works on magnitude, where the carrier cancels. Sub-voxel placement therefore
demodulates by the estimated carrier, shifts the smooth envelope with a spline,
and re-modulates analytically -- never interpolating a signal at Nyquist.

WHAT IS *NOT* MODELLED (and why)
--------------------------------
* **Grating lobes.** J5 came back REFUTED, so the elevation lobe structure we
  worried about is not there to model.
* **Intra-frame smear.** A bubble at 130 mm/s moves ~0.59 mm during the 4-transmit
  compounding window (~4.5 ms), i.e. ~3 axial voxels. We model the *phase*
  consequence of that motion (the compounding gain below), not the spatial
  smear, which would blur the injected PSF slightly. Injected fast bubbles are
  therefore marginally *easier* than real ones; recovery at high speed is an
  upper bound.
* **Non-linear bubble response / destruction.** Amplitude is constant over a
  track's life.

THE COMPOUNDING NULL (default ON)
---------------------------------
The 4-angle coherent sum is a velocity-selective filter: a scatterer moving
axially advances the round-trip phase by ``dphi = 4.pi.v_z / (lambda.PRF)``
between consecutive transmits, so the 4-phasor sum has gain
``|sin(2.dphi) / (4.sin(dphi/2))|`` -- a **null at lambda.FR/2 = 88.97 mm/s** on
this data (PRF = 4.FR), and -1.7 to -7.5 dB across the measured bulk. Because we
inject *after* the sum, that gain has to be applied by hand or synthetic bubbles
would be unfairly detectable at exactly the speed real ones vanish. It is on by
default and can be switched off (``compounding_gain=False``) by J2, whose whole
purpose is to recover that loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import map_coordinates

__all__ = [
    "Geometry",
    "ComplexPSF",
    "Bubble",
    "estimate_complex_psf",
    "reference_noise_sigma",
    "compounding_gain",
    "make_bubble_plan",
    "inject_bubbles",
    "null_block_shuffle",
    "null_phase_surrogate",
    "null_quiet_crop",
    "NULL_DESCRIPTIONS",
]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Geometry:
    """Physical scale of a ``(frames, elev, z, x)`` beamformed volume.

    ``d_elev`` is deliberately not equal to ``d_z``/``d_x``: this probe
    synthesizes 25 elevation planes at 0.5547 mm from 8 physical rows, while the
    in-plane grid is lambda/4 = 0.2 mm. Every voxel<->mm conversion in this
    module goes through here so that "speed in mm/s" means the same thing on
    every axis.
    """

    d_elev_mm: float = 0.5547
    d_z_mm: float = 0.2
    d_x_mm: float = 0.2
    frame_rate_hz: float = 222.43
    wavelength_mm: float = 0.8
    n_angles: int = 4
    z0_mm: float = 0.0  # depth of z index 0, for reporting only

    @property
    def voxel_mm(self) -> np.ndarray:
        return np.array([self.d_elev_mm, self.d_z_mm, self.d_x_mm], dtype=np.float64)

    def mms_to_voxels_per_frame(self, v_mms: Sequence[float]) -> np.ndarray:
        """(v_elev, v_z, v_x) in mm/s -> displacement per frame in voxels."""
        return np.asarray(v_mms, dtype=np.float64) / (self.voxel_mm * self.frame_rate_hz)

    def depth_mm(self, z_index: np.ndarray | float) -> np.ndarray:
        return self.z0_mm + np.asarray(z_index, dtype=np.float64) * self.d_z_mm


# --------------------------------------------------------------------------- #
# Empirical complex PSF
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ComplexPSF:
    """Measured complex point-spread function, split into envelope and carrier.

    ``env`` is the slowly-varying complex envelope on the patch grid and
    ``k_axial`` the axial carrier in rad/voxel, such that the PSF sampled at
    integer offsets ``u`` is ``env[u] * exp(1j * k_axial * u_z)``. Keeping them
    apart is what lets :meth:`sample` place a bubble at a sub-voxel position
    without interpolating a Nyquist-rate carrier.
    """

    env: np.ndarray  # complex64 (2*re+1, 2*rz+1, 2*rx+1)
    k_axial: float
    radius: tuple[int, int, int]
    n_patches: int
    peak_abs: float = 1.0
    norm: float = 0.0  # set in __post_init__: the on-grid peak magnitude

    def __post_init__(self) -> None:
        if self.norm <= 0:
            object.__setattr__(self, "norm", float(np.max(np.abs(self._raw((0.0, 0.0, 0.0))))))

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.env.shape)  # type: ignore[return-value]

    def kernel(self) -> np.ndarray:
        """The measured PSF itself, for inspection/tests."""
        return self.sample((0.0, 0.0, 0.0))

    def sample(self, delta: Sequence[float]) -> np.ndarray:
        """PSF sampled on the integer patch grid for a source offset ``delta``.

        ``delta`` is the source's sub-voxel position relative to the patch
        centre, in voxels, ordered (elev, z, x). The kernel is scaled by the
        **on-grid** peak, not by its own peak, so a source sitting between
        voxels correctly loses peak amplitude -- a real sub-voxel detectability
        variation that per-sample renormalization would erase and that would
        otherwise make injected bubbles uniformly easier than real ones.
        """
        return (self._raw(delta) / max(self.norm, 1e-30)).astype(np.complex64)

    def _raw(self, delta: Sequence[float]) -> np.ndarray:
        de, dz, dx = (float(v) for v in delta)
        re, rz, rx = self.radius
        # env(u - delta): sample the envelope at index (i - delta).
        gi = np.meshgrid(
            np.arange(self.env.shape[0], dtype=np.float64) - de,
            np.arange(self.env.shape[1], dtype=np.float64) - dz,
            np.arange(self.env.shape[2], dtype=np.float64) - dx,
            indexing="ij",
        )
        coords = np.stack([g.ravel() for g in gi], axis=0)
        shifted = np.empty(coords.shape[1], dtype=np.complex128)
        shifted.real = map_coordinates(self.env.real, coords, order=3, mode="constant", cval=0.0)
        shifted.imag = map_coordinates(self.env.imag, coords, order=3, mode="constant", cval=0.0)
        shifted = shifted.reshape(self.env.shape)
        # Re-modulate analytically: exp(i k (u_z - delta_z)).
        uz = np.arange(-rz, rz + 1, dtype=np.float64) - dz
        carrier = np.exp(1j * self.k_axial * uz)[None, :, None]
        return (shifted * carrier).astype(np.complex64)


def _axial_carrier(patch: np.ndarray) -> float:
    """Dominant axial phase ramp of a complex PSF patch, in rad/voxel.

    Circular mean of the adjacent-voxel phase difference along z, weighted by
    the product of the two magnitudes so the bright core dominates.
    """
    a = patch[:, :-1, :]
    b = patch[:, 1:, :]
    w = np.abs(a) * np.abs(b)
    acc = np.sum(w * np.exp(1j * (np.angle(b) - np.angle(a))))
    return float(np.angle(acc)) if np.abs(acc) > 0 else 0.0


def estimate_complex_psf(
    volume: np.ndarray,
    detections: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    patch_radius: tuple[int, int, int] = (3, 6, 6),
    max_patches: int = 400,
    min_zscore: float = 6.0,
    isolation_radius: tuple[int, int, int] = (4, 9, 9),
) -> ComplexPSF:
    """Measure the complex PSF from isolated bright detections.

    Mirrors :func:`ultratrace_ulm.psf.estimate_empirical_psf` (same isolation and
    brightness gating) but averages the **complex** patches, which is what an
    injection into the complex volume needs. Two extra steps are required
    relative to the magnitude version:

    * **Phase alignment.** Each bubble carries an arbitrary scatterer phase, so
      raw complex averaging would cancel. Each patch is divided by the phase of
      its own centre voxel before averaging.
    * **Carrier/envelope split.** The averaged patch is demodulated by its
      measured axial carrier so the stored envelope is smooth and safe to
      interpolate.

    ``volume`` is the SVD-filtered **complex** volume ``(frames, elev, z, x)`` --
    filtered, so the patches are bubble echoes rather than tissue; complex, so
    the carrier survives. ``detections`` are the ``(spatial, intensity, zscore)``
    triples the pipeline's detectors emit, indexed by frame.
    """
    data = np.asarray(volume)
    if data.ndim != 4:
        raise ValueError(f"Expected volume (frames,elev,z,x), got {data.shape}")
    if not np.iscomplexobj(data):
        raise ValueError("estimate_complex_psf needs the COMPLEX volume, not magnitude")
    radius = tuple(int(v) for v in patch_radius)
    iso = np.asarray([int(v) for v in isolation_radius], dtype=np.int32)

    cands: list[tuple[float, int, np.ndarray]] = []
    for f, det in enumerate(detections):
        spatial, _, zs = det[0], det[1], det[2]
        for coord, z in zip(spatial, zs):
            if float(z) >= float(min_zscore):
                cands.append((float(z), f, np.asarray(coord, dtype=np.int32)))
    cands.sort(key=lambda item: item[0], reverse=True)

    re, rz, rx = radius
    patches: list[np.ndarray] = []
    for _, f, c in cands:
        if len(patches) >= int(max_patches):
            break
        frame_dets = detections[f][0]
        if len(frame_dets) > 1:
            near = np.all(np.abs(frame_dets.astype(np.int32) - c[None, :]) <= iso[None, :], axis=1)
            if int(np.count_nonzero(near)) > 1:
                continue  # not isolated: another detection inside the PSF support
        e, z, x = (int(v) for v in c)
        if e - re < 0 or z - rz < 0 or x - rx < 0:
            continue
        if e + re >= data.shape[1] or z + rz >= data.shape[2] or x + rx >= data.shape[3]:
            continue
        p = data[f, e - re : e + re + 1, z - rz : z + rz + 1, x - rx : x + rx + 1]
        centre = p[re, rz, rx]
        if np.abs(centre) <= 0:
            continue
        patches.append((p / (centre / np.abs(centre))).astype(np.complex128))

    if len(patches) < 8:
        raise ValueError(f"Only {len(patches)} isolated complex PSF patches found (need >= 8)")

    mean_patch = np.mean(np.stack(patches, axis=0), axis=0)
    k = _axial_carrier(mean_patch)
    uz = np.arange(-rz, rz + 1, dtype=np.float64)
    env = (mean_patch * np.exp(-1j * k * uz)[None, :, None]).astype(np.complex64)
    return ComplexPSF(
        env=env,
        k_axial=k,
        radius=radius,
        n_patches=len(patches),
        peak_abs=float(np.max(np.abs(mean_patch))),
    )


# --------------------------------------------------------------------------- #
# Reference noise (what "SNR" is measured against)
# --------------------------------------------------------------------------- #
def reference_noise_sigma(
    volume: np.ndarray,
    *,
    ref_rank: int = 8,
    n_z_bands: int = 8,
    max_frames: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth-resolved noise scale of the complex volume, per (elev, z-band).

    "Controlled SNR" is only meaningful against a defined noise. We use the
    residual after removing a **fixed** reference rank (default 8, the low end of
    the clutter literature's tissue-subspace estimates) so the reference does not
    move with the filter under test -- otherwise the SNR axis of the sweep would
    be defined by the thing the sweep is measuring.

    Because attenuation and the beam profile vary strongly with depth, the scale
    is computed per elevation plane *and* per depth band: injecting at a fixed
    SNR then means a fixed detectability contrast at every depth, and amplitude
    varies with depth on its own. Robust (MAD-based) so residual bubbles do not
    inflate it.

    Returns ``(sigma, band_edges)`` with ``sigma`` of shape ``(elev, n_z_bands)``
    and ``band_edges`` of length ``n_z_bands + 1`` in z-index units.
    """
    data = np.asarray(volume)
    if data.ndim != 4:
        raise ValueError(f"Expected volume (frames,elev,z,x), got {data.shape}")
    n_f, n_e, n_z, n_x = data.shape
    step = max(1, n_f // int(max_frames))
    sub = data[::step]
    mat = sub.reshape(sub.shape[0], -1)
    # Rank-`ref_rank` tissue removal via the Gram (frames are the small axis).
    gram = (mat @ mat.conj().T).astype(np.complex128)
    evals, u = np.linalg.eigh(gram)
    u = u[:, np.argsort(evals)[::-1]]
    k = int(min(max(ref_rank, 0), u.shape[1] - 1))
    if k > 0:
        uk = u[:, :k]
        mat = mat - uk @ (uk.conj().T @ mat)
    resid = np.abs(mat).reshape(sub.shape)

    edges = np.linspace(0, n_z, int(n_z_bands) + 1).astype(int)
    sigma = np.ones((n_e, int(n_z_bands)), dtype=np.float32)
    for e in range(n_e):
        for b in range(int(n_z_bands)):
            block = resid[:, e, edges[b] : edges[b + 1], :]
            if block.size == 0:
                continue
            # |x| of complex Gaussian noise is Rayleigh; median/sqrt(2 ln 2) = sigma.
            med = float(np.median(block))
            sigma[e, b] = max(med / 1.17741, 1e-12)
    return sigma, edges


# --------------------------------------------------------------------------- #
# Compounding gain
# --------------------------------------------------------------------------- #
def compounding_gain(v_axial_mms: float | np.ndarray, geom: Geometry) -> np.ndarray:
    """Coherent N-angle compounding gain for an axially moving scatterer.

    ``dphi = 4.pi.v_z/(lambda.PRF)`` per transmit, ``PRF = N.FR``; the phasor sum
    is the Dirichlet kernel ``|sin(N.dphi/2) / (N.sin(dphi/2))|``. Unity at
    ``v_z = 0``, first null at ``v_z = lambda.PRF/(2N) = lambda.FR/2`` -- 88.97
    mm/s here, the same speed as the tracker's old 2-voxel gate wall (89 mm/s),
    which is why the two effects were confounded in every track-level measurement.
    """
    n = int(geom.n_angles)
    prf = n * geom.frame_rate_hz
    dphi = 4.0 * np.pi * np.asarray(v_axial_mms, dtype=np.float64) / (geom.wavelength_mm * prf)
    num = np.sin(n * dphi / 2.0)
    den = n * np.sin(dphi / 2.0)
    out = np.where(np.abs(den) < 1e-12, 1.0, num / np.where(np.abs(den) < 1e-12, 1.0, den))
    return np.abs(out)


# --------------------------------------------------------------------------- #
# Bubble plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bubble:
    """One synthetic bubble: a straight, constant-velocity track."""

    bubble_id: int
    start_evzx: tuple[float, float, float]  # voxel coords at frame f0
    velocity_mms: tuple[float, float, float]  # (elev, z, x)
    f0: int
    lifetime: int
    snr_db: float
    phase: float
    speed_mms: float
    axial_frac: float  # |v_z| / |v|, 1 = pure axial (feels the null), 0 = in-plane
    z_band: int
    direction: str = "axial"


DEFAULT_SPEEDS_MMS: tuple[float, ...] = (5.0, 20.0, 45.0, 70.0, 88.97, 110.0, 130.0)
# The interesting range, measured: below ~6 dB nothing is recovered by any arm
# and above ~21 dB everything is, so a sweep centred lower wastes its samples on
# two flat ends. Peak-voxel amplitude over the depth-resolved reference sigma.
DEFAULT_SNR_DB: tuple[float, ...] = (6.0, 9.0, 12.0, 15.0, 18.0, 21.0)


def make_bubble_plan(
    shape: tuple[int, int, int, int],
    geom: Geometry,
    *,
    n_bubbles: int = 300,
    lifetime: int = 24,
    speeds_mms: Sequence[float] = DEFAULT_SPEEDS_MMS,
    snr_db: Sequence[float] = DEFAULT_SNR_DB,
    n_z_bands: int = 4,
    directions: Sequence[str] = ("axial", "lateral", "elev", "oblique"),
    margin: tuple[int, int, int] = (4, 10, 10),
    min_lifetime: int = 6,
    seed: int = 0,
) -> list[Bubble]:
    """A stratified sample of synthetic tracks over speed x direction x depth x SNR.

    The design is a randomized balanced assignment rather than a full factorial:
    the marginals the bead asks for (speed, depth, SNR) stay well powered while
    per-frame injected density stays realistic. At ``n_bubbles=300`` and
    ``lifetime=24`` over 240 frames the added density is ~30 bubbles/frame,
    comparable to the ~34/frame the sane-threshold pipeline finds -- so the
    injection does not itself change the statistics it is measured against. Run
    several independent realizations (different ``seed``) rather than one dense
    one for more samples per cell.

    Speeds deliberately span 0-130 mm/s and include 88.97, where the compounding
    null sits and where the tracker's old gate wall stood; ``directions`` splits
    axial from in-plane because the null is **axial only**, so a speed-only
    stratification would average the two and hide it.
    """
    rng = np.random.default_rng(seed)
    n_f, n_e, n_z, n_x = shape
    me, mz, mx = margin
    speeds = list(speeds_mms)
    snrs = list(snr_db)
    dirs = list(directions)

    # Draw cells from a SHUFFLED full factorial rather than by nested integer
    # division. The nested-division form silently degenerates whenever
    # `n_bubbles` is smaller than the product of the inner factors' sizes -- the
    # outermost factor then never advances off its first level, and one whole
    # axis of the design collapses to a constant without any error. (It did:
    # the first run of this sweep planted 900 bubbles that were all axial, which
    # confounded the speed stratification with the axial-only compounding null.)
    # A shuffled factorial cannot degenerate that way, and successive seeds
    # cover different cells so realizations accumulate rather than repeat.
    cells = [
        (si, ni, bi, di)
        for si in range(len(speeds))
        for ni in range(len(snrs))
        for bi in range(int(n_z_bands))
        for di in range(len(dirs))
    ]
    rng.shuffle(cells)

    band_edges = np.linspace(mz, n_z - mz, int(n_z_bands) + 1)
    bubbles: list[Bubble] = []
    for i in range(int(n_bubbles)):
        si, ni, band, di = cells[i % len(cells)]
        speed, snr, direction = speeds[si], snrs[ni], dirs[di]

        if direction == "axial":
            u = np.array([0.0, 1.0, 0.0])
        elif direction == "lateral":
            u = np.array([0.0, 0.0, 1.0])
        elif direction == "elev":
            u = np.array([1.0, 0.0, 0.0])
        else:  # oblique: an isotropic random direction
            u = rng.normal(size=3)
            u /= max(float(np.linalg.norm(u)), 1e-12)
        u = u * float(rng.choice([-1.0, 1.0]))
        v = u * float(speed)

        # Place so the whole track stays inside the volume with margin. A fast
        # axial bubble covers 2.9 voxels/frame, so at 130 mm/s a 24-frame track
        # crosses 67 of the 154 depth samples: the lifetime, not the start, is
        # what has to give. Shorten it rather than let a track leave the volume,
        # and never below `min_lifetime` -- a 2-frame track is not a track.
        lo_vol = np.array([me, mz, mx], dtype=np.float64)
        hi_vol = np.array([n_e - 1 - me, n_z - 1 - mz, n_x - 1 - mx], dtype=np.float64)
        per_frame = geom.mms_to_voxels_per_frame(v)
        span = hi_vol - lo_vol
        with np.errstate(divide="ignore", invalid="ignore"):
            max_life = np.where(np.abs(per_frame) > 1e-9, span / np.abs(per_frame), np.inf) + 1
        life = int(np.clip(np.floor(np.min(max_life)), min_lifetime, lifetime))
        disp = per_frame * (life - 1)

        lo_feas = np.maximum(lo_vol, lo_vol - disp)
        hi_feas = np.minimum(hi_vol, hi_vol - disp)
        # Prefer the assigned depth band, but the volume constraint wins: a fast
        # axial track simply cannot be confined to one band, so it spans several
        # and depth is read per-frame from the truth table instead.
        lo_z = max(lo_feas[1], float(band_edges[band]))
        hi_z = min(hi_feas[1], float(band_edges[band + 1]))
        if hi_z > lo_z:
            lo_feas[1], hi_feas[1] = lo_z, hi_z
        hi_feas = np.maximum(hi_feas, lo_feas)
        f0 = int(rng.integers(0, max(1, n_f - life)))
        start = lo_feas + rng.random(3) * (hi_feas - lo_feas)

        bubbles.append(
            Bubble(
                bubble_id=i,
                start_evzx=(float(start[0]), float(start[1]), float(start[2])),
                velocity_mms=(float(v[0]), float(v[1]), float(v[2])),
                f0=f0,
                lifetime=int(life),
                snr_db=float(snr),
                phase=float(rng.random() * 2.0 * np.pi),
                speed_mms=float(speed),
                axial_frac=float(abs(u[1])),
                z_band=int(band),
                direction=str(direction),
            )
        )
    return bubbles


# --------------------------------------------------------------------------- #
# Injection
# --------------------------------------------------------------------------- #
def inject_bubbles(
    volume: np.ndarray,
    bubbles: Iterable[Bubble],
    psf: ComplexPSF,
    geom: Geometry,
    sigma: np.ndarray,
    band_edges: np.ndarray,
    *,
    compounding_gain_on: bool = True,
    copy: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Add synthetic bubble echoes to a beamformed **complex** volume.

    ``sigma``/``band_edges`` come from :func:`reference_noise_sigma`; a bubble's
    peak voxel magnitude is ``sigma(elev, z-band) * 10**(snr_db/20)``, evaluated
    at its **start** position, so a given ``snr_db`` means the same contrast at
    every depth even though the absolute amplitude does not.

    Returns the volume and a flat truth table: one row per bubble-frame, with
    voxel and mm positions and the strata (speed, axial fraction, depth band,
    SNR) needed to stratify recovery. Bubble-frames whose PSF would cross a
    volume face are dropped from the truth table as well as from the volume, so
    recovery is never charged for a bubble that was never fully injected.
    """
    data = np.array(volume, copy=copy)
    if data.ndim != 4 or not np.iscomplexobj(data):
        raise ValueError(f"Expected complex volume (frames,elev,z,x), got {data.shape} {data.dtype}")
    n_f, n_e, n_z, n_x = data.shape
    re, rz, rx = psf.radius

    rows: dict[str, list] = {
        k: []
        for k in (
            "bubble_id", "frame", "e", "z", "x", "e_mm", "z_mm", "x_mm",
            "snr_db", "speed_mms", "axial_frac", "z_band", "gain_db", "amp", "direction",
        )
    }
    for b in bubbles:
        per_frame = geom.mms_to_voxels_per_frame(b.velocity_mms)
        band = int(np.clip(np.searchsorted(band_edges, b.start_evzx[1], side="right") - 1,
                           0, sigma.shape[1] - 1))
        e0 = int(np.clip(round(b.start_evzx[0]), 0, n_e - 1))
        gain = float(compounding_gain(b.velocity_mms[1], geom)) if compounding_gain_on else 1.0
        amp = float(sigma[e0, band]) * (10.0 ** (b.snr_db / 20.0)) * gain
        if amp <= 0:
            continue
        carrier = amp * np.exp(1j * b.phase)
        for t in range(b.lifetime):
            f = b.f0 + t
            if f >= n_f:
                break
            p = np.asarray(b.start_evzx, dtype=np.float64) + per_frame * t
            c = np.round(p).astype(int)
            if (c[0] - re < 0 or c[1] - rz < 0 or c[2] - rx < 0
                    or c[0] + re >= n_e or c[1] + rz >= n_z or c[2] + rx >= n_x):
                continue
            kernel = psf.sample(p - c)
            data[f,
                 c[0] - re : c[0] + re + 1,
                 c[1] - rz : c[1] + rz + 1,
                 c[2] - rx : c[2] + rx + 1] += (carrier * kernel).astype(data.dtype)
            rows["bubble_id"].append(b.bubble_id)
            rows["frame"].append(f)
            rows["e"].append(p[0]); rows["z"].append(p[1]); rows["x"].append(p[2])
            rows["e_mm"].append(p[0] * geom.d_elev_mm)
            rows["z_mm"].append(geom.z0_mm + p[1] * geom.d_z_mm)
            rows["x_mm"].append(p[2] * geom.d_x_mm)
            rows["snr_db"].append(b.snr_db)
            rows["speed_mms"].append(b.speed_mms)
            rows["axial_frac"].append(b.axial_frac)
            rows["direction"].append(b.direction)
            # Depth band is read from THIS frame's position, not the track's
            # start: a fast axial bubble crosses bands within its own lifetime,
            # and the question being asked ("is recovery depth-uniform?") is
            # about where the echo was when it was or was not detected.
            rows["z_band"].append(
                int(np.clip(np.searchsorted(band_edges, p[1], side="right") - 1,
                            0, sigma.shape[1] - 1))
            )
            rows["gain_db"].append(20.0 * np.log10(max(gain, 1e-6)))
            rows["amp"].append(amp)

    truth = {k: np.asarray(v) for k, v in rows.items()}
    for k in ("bubble_id", "frame", "z_band"):
        truth[k] = truth[k].astype(np.int64) if truth[k].size else np.zeros(0, dtype=np.int64)
    return data, truth


# --------------------------------------------------------------------------- #
# Null data
# --------------------------------------------------------------------------- #
NULL_DESCRIPTIONS: dict[str, str] = {
    "block_shuffle": (
        "Contiguous blocks of frames permuted. **MEASURED VERDICT: not a usable "
        "null for a per-frame detector -- do not report it as one.** Permuting "
        "frames leaves every frame's contents, including every real bubble, "
        "exactly intact; it removes trajectories, not echoes. A detector that "
        "works one frame at a time therefore sees the same material, and in the "
        "J3 sweep both block=1 and block=20 returned the matched target density "
        "(34.00/frame) for all 18 operating points -- i.e. zero information. Its "
        "only real effect is on the SVD, which loses the temporal coherence it "
        "uses to identify tissue; that shifts WHICH modes are removed but not how "
        "much signal is present. Keep it as the null for TRACKER-level claims, "
        "where destroying trajectories is exactly the right intervention, and use "
        "phase_surrogate or quiet_crop for detector-level false alarms."
    ),
    "phase_surrogate": (
        "Per-voxel temporal Fourier surrogate: each voxel's time series keeps its "
        "power spectrum exactly and gets an INDEPENDENT random phase per bin. "
        "PRESERVES: the per-voxel temporal spectrum (so tissue still looks slow "
        "and noise still looks fast -- the axis the clutter filter cuts on) and "
        "the marginal amplitude scale. BREAKS: inter-voxel coherence, and the "
        "mechanism is exact rather than hand-wavy -- the randomized array is "
        "still supported on the same frequency bins, so the clutter's rank goes "
        "from 'number of spatial modes' (a few) to 'bandwidth in bins' (tens). A "
        "fixed-rank cut therefore removes much less of it, leaving tissue residue "
        "and OVER-stating false alarms. Two corollaries worth stating: a "
        "monochromatic clutter mode would survive with rank 1 intact (so this "
        "null is only hard on realistically broadband clutter), and a single "
        "GLOBAL per-bin phase would leave the Gram spectrum exactly unchanged "
        "while failing to remove bubbles -- which is why the draw is per-voxel. "
        "Complementary to block_shuffle: hard on the SPATIAL low-rank assumption "
        "where block_shuffle is hard on the TEMPORAL one."
    ),
    "quiet_crop": (
        "A spatial crop of the real volume chosen where the baseline detector "
        "finds the fewest suprathreshold voxels. PRESERVES: everything -- it is "
        "real data, with real clutter, real noise and the real PSF. BREAKS: "
        "nothing, which is the problem: it is not guaranteed bubble-free, so any "
        "detection counted here may be a real bubble. It therefore gives a "
        "CONSERVATIVE (upper-bound) false-alarm rate and is the closest thing to "
        "an assumption-free null we have. Treat disagreement between quiet_crop "
        "and the synthetic nulls as the honest error bar on the false-alarm axis."
    ),
    "clean": (
        "Not a null: the un-injected real volume. Used for MATCHED DETECTION "
        "DENSITY, the primary operating-point match. It makes no assumption at "
        "all, which is why it, and not a synthetic null, is the primary axis; "
        "the nulls are the cross-check."
    ),
}


def null_block_shuffle(volume: np.ndarray, *, block: int = 20, seed: int = 0) -> np.ndarray:
    """Permute contiguous blocks of frames. See :data:`NULL_DESCRIPTIONS`."""
    data = np.asarray(volume)
    n_f = data.shape[0]
    b = max(1, int(block))
    starts = np.arange(0, n_f, b)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(starts))
    out = np.empty_like(data)
    cursor = 0
    for j in order:
        s = int(starts[j])
        e = min(s + b, n_f)
        out[cursor : cursor + (e - s)] = data[s:e]
        cursor += e - s
    return out


def null_phase_surrogate(volume: np.ndarray, *, seed: int = 0, chunk: int = 4) -> np.ndarray:
    """Per-voxel temporal Fourier surrogate. See :data:`NULL_DESCRIPTIONS`.

    The phase draw is **independent per voxel**, which is what makes this a
    bubble-free null: a single global per-bin phase would be a unitary temporal
    mixing that leaves the Gram spectrum -- and therefore the whole SVD -- exactly
    intact, but would also leave bright bubble blobs standing as superpositions
    of the bubble's own positions. Independent draws destroy those. The cost is
    that inter-voxel coherence goes too, so the tissue subspace is no longer
    spatially low-rank; that is documented in :data:`NULL_DESCRIPTIONS`.

    Data is complex (analytic), so there is no Hermitian constraint to preserve.
    Chunked over elevation to keep the complex128 FFT workspace bounded.
    """
    data = np.asarray(volume)
    n_f = data.shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty_like(data)
    for e0 in range(0, data.shape[1], max(1, int(chunk))):
        e1 = min(e0 + max(1, int(chunk)), data.shape[1])
        block = data[:, e0:e1]
        spec = np.fft.fft(block, axis=0)
        spec *= np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=spec.shape))
        out[:, e0:e1] = np.fft.ifft(spec, axis=0).astype(data.dtype, copy=False)
    return out


def null_quiet_crop(
    volume: np.ndarray,
    magnitude: np.ndarray,
    *,
    crop: tuple[int, int] = (64, 64),
    percentile: float = 99.5,
) -> tuple[np.ndarray, tuple[int, int]]:
    """The quietest (z, x) crop of the real volume. See :data:`NULL_DESCRIPTIONS`.

    "Quietest" = smallest high-percentile of the SVD-filtered magnitude over the
    whole record, which is the statistic bubbles move and mean intensity does
    not. Returns the cropped **complex** volume and the (z0, x0) origin.
    """
    mag = np.asarray(magnitude)
    n_z, n_x = mag.shape[2], mag.shape[3]
    cz, cx = int(min(crop[0], n_z)), int(min(crop[1], n_x))
    stat = np.percentile(mag, percentile, axis=0)  # (elev, z, x)
    stat = stat.mean(axis=0)  # (z, x)
    # Integral image for O(1) box sums.
    ii = np.pad(np.cumsum(np.cumsum(stat, axis=0), axis=1), ((1, 0), (1, 0)))
    best, best_zx = np.inf, (0, 0)
    for z0 in range(0, n_z - cz + 1, max(1, cz // 4)):
        for x0 in range(0, n_x - cx + 1, max(1, cx // 4)):
            s = (ii[z0 + cz, x0 + cx] - ii[z0, x0 + cx] - ii[z0 + cz, x0] + ii[z0, x0])
            if s < best:
                best, best_zx = float(s), (z0, x0)
    z0, x0 = best_zx
    return np.asarray(volume)[:, :, z0 : z0 + cz, x0 : x0 + cx], best_zx
