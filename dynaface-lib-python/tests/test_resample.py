"""Golden tests for dynaface.resample.

Strategy: an independent scalar-loop reference (a direct transcription of the
.NET ImageUtils formulas) is cross-checked for *exact* equality against the
vectorized numpy implementation on randomized inputs. That catches vectorization
bugs, and the identity/known-value cases lock the conventions (half-pixel
centers, truncating lerp, round-half-away-from-zero nearest).
"""

import numpy as np

from dynaface import resample

F32 = np.float32


# --------------------------------------------------------------------------
# Scalar-loop references (mirror ImageUtils.cs line-for-line)
# --------------------------------------------------------------------------
def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _lerp_u8(a, b, t):
    return np.uint8(F32(a) + (F32(b) - F32(a)) * t)


def _round_away(f):
    f = F32(f)
    return F32(np.sign(f) * np.floor(np.abs(f) + F32(0.5)))


def ref_resize(src, dst_w, dst_h):
    src = np.asarray(src)
    squeeze = src.ndim == 2
    if squeeze:
        src = src[:, :, None]
    h, w, c = src.shape
    out = np.zeros((dst_h, dst_w, c), np.uint8)
    xs, ys = F32(w) / F32(dst_w), F32(h) / F32(dst_h)
    xmax, ymax = F32(w - 1), F32(h - 1)
    for dy in range(dst_h):
        for dx in range(dst_w):
            sx = _clamp((F32(dx) + F32(0.5)) * xs - F32(0.5), F32(0), xmax)
            sy = _clamp((F32(dy) + F32(0.5)) * ys - F32(0.5), F32(0), ymax)
            x0, y0 = int(sx), int(sy)
            x1 = x0 + 1 if x0 + 1 < w else x0
            y1 = y0 + 1 if y0 + 1 < h else y0
            tx, ty = F32(sx - x0), F32(sy - y0)
            for ch in range(c):
                top = _lerp_u8(src[y0, x0, ch], src[y0, x1, ch], tx)
                bot = _lerp_u8(src[y1, x0, ch], src[y1, x1, ch], tx)
                out[dy, dx, ch] = _lerp_u8(top, bot, ty)
    return out[:, :, 0] if squeeze else out


def ref_warp(src, m, dst_w, dst_h, bilinear, fill):
    src = np.asarray(src)
    squeeze = src.ndim == 2
    if squeeze:
        src = src[:, :, None]
    h, w, c = src.shape
    fill_arr = np.broadcast_to(np.asarray(fill, np.uint8), (c,))
    out = np.zeros((dst_h, dst_w, c), np.uint8)
    inv = resample.invert_affine(m)
    i00, i01, i02 = inv[0]
    i10, i11, i12 = inv[1]

    def tap(x, y):
        if 0 <= x < w and 0 <= y < h:
            return src[y, x].astype(F32)
        return fill_arr.astype(F32)

    for dy in range(dst_h):
        for dx in range(dst_w):
            sx = F32(i00 * F32(dx) + i01 * F32(dy) + i02)
            sy = F32(i10 * F32(dx) + i11 * F32(dy) + i12)
            if not bilinear:
                ix, iy = int(_round_away(sx)), int(_round_away(sy))
                out[dy, dx] = src[iy, ix] if (0 <= ix < w and 0 <= iy < h) else fill_arr
            else:
                x0, y0 = int(np.floor(sx)), int(np.floor(sy))
                tx, ty = F32(sx - x0), F32(sy - y0)
                c00, c10 = tap(x0, y0), tap(x0 + 1, y0)
                c01, c11 = tap(x0, y0 + 1), tap(x0 + 1, y0 + 1)
                r0 = c00 + (c10 - c00) * tx
                r1 = c01 + (c11 - c01) * tx
                v = r0 + (r1 - r0) * ty
                for ch in range(c):
                    out[dy, dx, ch] = np.uint8(
                        min(255, max(0, int(_round_away(v[ch]))))
                    )
    return out[:, :, 0] if squeeze else out


# --------------------------------------------------------------------------
# resize_bilinear
# --------------------------------------------------------------------------
def test_resize_identity_returns_exact_copy():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(11, 7, 3), dtype=np.uint8)
    out = resample.resize_bilinear(img, 7, 11)
    np.testing.assert_array_equal(out, img)


def test_resize_matches_scalar_reference_downscale():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(17, 13, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        resample.resize_bilinear(img, 5, 9), ref_resize(img, 5, 9)
    )


def test_resize_matches_scalar_reference_upscale():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        resample.resize_bilinear(img, 19, 15), ref_resize(img, 19, 15)
    )


def test_resize_grayscale_2d():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(10, 10), dtype=np.uint8)
    out = resample.resize_bilinear(img, 4, 7)
    assert out.shape == (7, 4)
    np.testing.assert_array_equal(out, ref_resize(img, 4, 7))


def test_resize_is_deterministic():
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, size=(23, 29, 3), dtype=np.uint8)
    a = resample.resize_bilinear(img, 12, 16)
    b = resample.resize_bilinear(img, 12, 16)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------
# invert_affine
# --------------------------------------------------------------------------
def test_invert_affine_known():
    # forward: (x, y) -> (2x + 3, 4y + 5)
    m = np.array([[2.0, 0.0, 3.0], [0.0, 4.0, 5.0]], dtype=F32)
    inv = resample.invert_affine(m)
    expected = np.array([[0.5, 0.0, -1.5], [0.0, 0.25, -1.25]], dtype=F32)
    np.testing.assert_allclose(inv, expected, atol=1e-6)


def test_invert_affine_round_trip():
    m = np.array([[1.3, -0.2, 12.0], [0.15, 0.9, -4.0]], dtype=F32)
    inv2 = resample.invert_affine(resample.invert_affine(m))
    np.testing.assert_allclose(inv2, m, atol=1e-4)


def test_invert_affine_singular_returns_identity():
    m = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]], dtype=F32)
    inv = resample.invert_affine(m)
    np.testing.assert_array_equal(
        inv, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=F32)
    )


# --------------------------------------------------------------------------
# warp_affine
# --------------------------------------------------------------------------
def _identity_affine():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=F32)


def test_warp_identity_nearest_returns_copy():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, size=(9, 11, 3), dtype=np.uint8)
    out = resample.warp_affine(img, _identity_affine(), 11, 9, bilinear=False)
    np.testing.assert_array_equal(out, img)


def test_warp_integer_translation_with_fill():
    # forward shifts content right by 2, down by 1; out-of-bounds -> fill.
    img = np.arange(4 * 4, dtype=np.uint8).reshape(4, 4)
    m = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]], dtype=F32)
    out = resample.warp_affine(img, m, 4, 4, bilinear=False, fill=99)
    np.testing.assert_array_equal(out, ref_warp(img, m, 4, 4, False, 99))
    # top row and left two columns are fill
    assert out[0, 0] == 99 and out[0, 3] == 99 and out[1, 1] == 99


def test_warp_nearest_matches_scalar_reference():
    rng = np.random.default_rng(6)
    img = rng.integers(0, 256, size=(20, 24, 3), dtype=np.uint8)
    # scale ~1.6, small rotation-free skew + translation
    m = np.array([[1.6, 0.0, -3.0], [0.0, 1.6, 5.0]], dtype=F32)
    np.testing.assert_array_equal(
        resample.warp_affine(img, m, 32, 28, bilinear=False, fill=0),
        ref_warp(img, m, 32, 28, False, 0),
    )


def test_warp_bilinear_matches_scalar_reference():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(18, 22, 3), dtype=np.uint8)
    m = np.array([[1.3, 0.1, 2.0], [-0.05, 1.4, -2.0]], dtype=F32)
    np.testing.assert_array_equal(
        resample.warp_affine(img, m, 30, 26, bilinear=True, fill=0),
        ref_warp(img, m, 30, 26, True, 0),
    )


def test_warp_is_deterministic():
    rng = np.random.default_rng(8)
    img = rng.integers(0, 256, size=(15, 15, 3), dtype=np.uint8)
    m = np.array([[1.1, 0.2, 1.0], [-0.1, 1.2, 2.0]], dtype=F32)
    a = resample.warp_affine(img, m, 20, 20, bilinear=False, fill=7)
    b = resample.warp_affine(img, m, 20, 20, bilinear=False, fill=7)
    np.testing.assert_array_equal(a, b)
