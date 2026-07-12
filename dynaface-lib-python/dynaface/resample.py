"""Deterministic, first-principles image resampling.

A faithful port of the .NET dynaface port's ``facial_dll/ImageUtils.cs``
(``ResizeBilinear`` / ``WarpAffine`` / ``InvertAffine``), used to replace the
OpenCV calls on the *measurement-affecting* preprocessing path
(``dynaface_onnx.py``: the BlazeFace/SPIGA/U-2-Net input resamples).

Why not just use ``cv2.resize`` / ``cv2.warpAffine``?  Those routines are
deterministic for a *fixed* OpenCV build, but their fixed-point ``INTER_LINEAR``
kernels differ across OpenCV versions and CPU architectures (e.g. Apple-Silicon
NEON vs. x86 AVX), which shifts detected landmarks by ~1px and makes measurements
non-reproducible across machines.  These routines use only elementwise
``float32`` arithmetic in a fixed order (no BLAS/matmul, no transcendentals), so
their output is bit-identical on any platform, and matches the .NET port's own
resamplers convention-for-convention so the two ports agree.

Conventions mirrored from ``ImageUtils``:
  * ``resize_bilinear`` -- half-pixel centers ``s = (d + 0.5)*scale - 0.5``
    clamped to ``[0, dim-1]`` with ``scale = src/dst``; the C# ``(byte)`` cast is
    a *truncation toward zero*, applied per lerp stage (horizontal x2, then
    vertical), NOT a round.
  * ``warp_affine`` -- takes the *forward* 2x3 matrix (same layout cv2 expects)
    and inverts it internally; addresses pixels directly (no +-0.5).  Nearest
    uses round-half-away-from-zero; bilinear uses float lerps then
    round-away + clamp to [0, 255].  Out-of-bounds taps (incl. negative
    coordinates) use ``fill``.
  * ``invert_affine`` -- closed-form 2x3 inverse with a near-singular guard that
    returns the identity (callers needing the singular case must check first).
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

_F32 = np.float32
_HALF = np.float32(0.5)
_ZERO = np.float32(0.0)
_SINGULAR_EPS = 1e-12


def _clamp_f32(
    v: NDArray[np.float32], lo: np.float32, hi: np.float32
) -> NDArray[np.float32]:
    # Matches C# ``v < min ? min : v > max ? max : v`` for finite inputs.
    return np.minimum(np.maximum(v, lo), hi).astype(_F32)


def _round_away(x: NDArray[np.float32]) -> NDArray[np.float32]:
    # Round half away from zero, matching MathF.Round(f, MidpointRounding.AwayFromZero).
    # np.round is banker's rounding (half-to-even), so it cannot be used here.
    return (np.sign(x) * np.floor(np.abs(x).astype(_F32) + _HALF)).astype(_F32)


def _lerp_trunc_u8(
    a: NDArray[np.uint8], b: NDArray[np.uint8], t: NDArray[np.float32]
) -> NDArray[np.uint8]:
    # C# LerpColor: (byte)(a + (b - a) * t), i.e. compute in float32 then truncate
    # toward zero. Inputs are in [0, 255] and t in [0, 1], so the result is
    # non-negative and truncation-toward-zero == floor; ``astype(uint8)`` matches
    # the C# ``(byte)`` cast exactly.
    af = a.astype(_F32)
    bf = b.astype(_F32)
    return (af + (bf - af) * t).astype(np.uint8)


def resize_bilinear(
    src: NDArray[np.uint8], dst_w: int, dst_h: int
) -> NDArray[np.uint8]:
    """Bilinear resize of a ``uint8`` image, matching ``ImageUtils.ResizeBilinear``.

    ``src`` is ``(H, W)`` or ``(H, W, C)`` ``uint8``; returns the same rank at
    ``(dst_h, dst_w[, C])``.
    """
    src = np.asarray(src)
    if src.dtype != np.uint8:
        raise TypeError(f"resize_bilinear expects uint8, got {src.dtype}")
    squeeze = src.ndim == 2
    if squeeze:
        src = src[:, :, None]
    h, w, _ = src.shape

    x_scale = _F32(w) / _F32(dst_w)
    y_scale = _F32(h) / _F32(dst_h)
    x_max = _F32(w - 1)
    y_max = _F32(h - 1)

    dx = np.arange(dst_w, dtype=_F32)
    dy = np.arange(dst_h, dtype=_F32)
    sx = _clamp_f32((dx + _HALF) * x_scale - _HALF, _ZERO, x_max)  # (dst_w,)
    sy = _clamp_f32((dy + _HALF) * y_scale - _HALF, _ZERO, y_max)  # (dst_h,)

    x0 = sx.astype(np.intp)  # truncation toward zero; sx >= 0
    y0 = sy.astype(np.intp)
    x1 = np.where(x0 + 1 < w, x0 + 1, x0)
    y1 = np.where(y0 + 1 < h, y0 + 1, y0)
    tx = (sx - x0.astype(_F32)).astype(_F32)[None, :, None]  # (1, dst_w, 1)
    ty = (sy - y0.astype(_F32)).astype(_F32)[:, None, None]  # (dst_h, 1, 1)

    c00 = src[np.ix_(y0, x0)]
    c10 = src[np.ix_(y0, x1)]
    c01 = src[np.ix_(y1, x0)]
    c11 = src[np.ix_(y1, x1)]

    top = _lerp_trunc_u8(c00, c10, tx)
    bot = _lerp_trunc_u8(c01, c11, tx)
    out = _lerp_trunc_u8(top, bot, ty)
    return out[:, :, 0] if squeeze else out


def resize_mask_bilinear(
    src: NDArray[np.float32], dst_w: int, dst_h: int
) -> NDArray[np.float32]:
    """Bilinear resize of a single-channel ``float32`` map.

    Matches ``SagittalProfile.ResizeMaskBilinear`` on the .NET side: same
    half-pixel-center convention as :func:`resize_bilinear`, but the lerps stay in
    float32 with no per-stage truncation (the map is a saliency mask, not pixels).
    """
    src = np.asarray(src, dtype=_F32)
    h, w = src.shape

    x_scale = _F32(w) / _F32(dst_w)
    y_scale = _F32(h) / _F32(dst_h)
    x_max = _F32(w - 1)
    y_max = _F32(h - 1)

    dx = np.arange(dst_w, dtype=_F32)
    dy = np.arange(dst_h, dtype=_F32)
    sx = _clamp_f32((dx + _HALF) * x_scale - _HALF, _ZERO, x_max)
    sy = _clamp_f32((dy + _HALF) * y_scale - _HALF, _ZERO, y_max)

    x0 = sx.astype(np.intp)
    y0 = sy.astype(np.intp)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx = (sx - x0.astype(_F32)).astype(_F32)[None, :]  # (1, dst_w)
    ty = (sy - y0.astype(_F32)).astype(_F32)[:, None]  # (dst_h, 1)

    v00 = src[np.ix_(y0, x0)]
    v10 = src[np.ix_(y0, x1)]
    v01 = src[np.ix_(y1, x0)]
    v11 = src[np.ix_(y1, x1)]
    v0 = v00 + (v10 - v00) * tx
    v1 = v01 + (v11 - v01) * tx
    return (v0 + (v1 - v0) * ty).astype(_F32)


def invert_affine(m: NDArray[np.float32]) -> NDArray[np.float32]:
    """Closed-form inverse of a forward 2x3 affine, matching ``ImageUtils.InvertAffine``.

    Returns the identity transform if ``m`` is (near-)singular.
    """
    m = np.asarray(m, dtype=_F32)
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    det = m00 * m11 - m01 * m10
    if abs(float(det)) < _SINGULAR_EPS:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=_F32)
    inv_det = _F32(1.0) / det
    i00 = m11 * inv_det
    i01 = -m01 * inv_det
    i10 = -m10 * inv_det
    i11 = m00 * inv_det
    i02 = -(i00 * m02 + i01 * m12)
    i12 = -(i10 * m02 + i11 * m12)
    return np.array([[i00, i01, i02], [i10, i11, i12]], dtype=_F32)


def warp_affine(
    src: NDArray[np.uint8],
    m: NDArray[np.float32],
    dst_w: int,
    dst_h: int,
    bilinear: bool = False,
    fill: Union[int, tuple] = 0,
) -> NDArray[np.uint8]:
    """Affine warp of a ``uint8`` image, matching ``ImageUtils.WarpAffine``.

    ``m`` is the *forward* (src->dst) 2x3 matrix (same layout ``cv2.warpAffine``
    expects); it is inverted internally.  ``fill`` is used for out-of-bounds taps.
    """
    src = np.asarray(src)
    if src.dtype != np.uint8:
        raise TypeError(f"warp_affine expects uint8, got {src.dtype}")
    squeeze = src.ndim == 2
    if squeeze:
        src = src[:, :, None]
    h, w, c = src.shape
    fill_arr = np.broadcast_to(np.asarray(fill, dtype=np.uint8), (c,))

    m = np.asarray(m, dtype=_F32)
    m00, m01, m10, m11 = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    det = m00 * m11 - m01 * m10
    if abs(float(det)) < _SINGULAR_EPS:
        out = np.empty((dst_h, dst_w, c), dtype=np.uint8)
        out[:] = fill_arr
        return out[:, :, 0] if squeeze else out

    inv = invert_affine(m)
    i00, i01, i02 = inv[0]
    i10, i11, i12 = inv[1]

    dx = np.arange(dst_w, dtype=_F32)[None, :]  # (1, dst_w)
    dy = np.arange(dst_h, dtype=_F32)[:, None]  # (dst_h, 1)
    sx = i00 * dx + i01 * dy + i02  # (dst_h, dst_w)
    sy = i10 * dx + i11 * dy + i12

    if not bilinear:
        ix = _round_away(sx).astype(np.intp)
        iy = _round_away(sy).astype(np.intp)
        out = _gather_or_fill(src, ix, iy, fill_arr)
        return out[:, :, 0] if squeeze else out

    x0 = np.floor(sx).astype(np.intp)
    y0 = np.floor(sy).astype(np.intp)
    tx = (sx - x0.astype(_F32)).astype(_F32)[:, :, None]
    ty = (sy - y0.astype(_F32)).astype(_F32)[:, :, None]

    c00 = _gather_or_fill(src, x0, y0, fill_arr).astype(_F32)
    c10 = _gather_or_fill(src, x0 + 1, y0, fill_arr).astype(_F32)
    c01 = _gather_or_fill(src, x0, y0 + 1, fill_arr).astype(_F32)
    c11 = _gather_or_fill(src, x0 + 1, y0 + 1, fill_arr).astype(_F32)

    r0 = c00 + (c10 - c00) * tx
    r1 = c01 + (c11 - c01) * tx
    v = r0 + (r1 - r0) * ty
    out = np.clip(_round_away(v), 0, 255).astype(np.uint8)
    return out[:, :, 0] if squeeze else out


def _gather_or_fill(
    src: NDArray[np.uint8],
    x: NDArray[np.intp],
    y: NDArray[np.intp],
    fill_arr: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    # Matches TapOrFill's ``(uint)x < (uint)width`` bounds test: negative or
    # >= dim coordinates fall back to fill.
    h, w, _ = src.shape
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    xc = np.clip(x, 0, w - 1)
    yc = np.clip(y, 0, h - 1)
    gathered = src[yc, xc]
    return np.where(valid[..., None], gathered, fill_arr)
