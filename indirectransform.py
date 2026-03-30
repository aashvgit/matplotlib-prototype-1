
from __future__ import annotations

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.transforms import (
    Transform, TransformNode, BboxTransformTo, Bbox,
)
from typing import Callable
HERE = os.path.dirname(os.path.abspath(__file__))

class IndirectTransform(Transform):

    input_dims = 2
    output_dims = 2

    def __init__(self, func: Callable[[], Transform], **kwargs):
        if not callable(func):
            raise TypeError(
                f"IndirectTransform: func must be callable, got {type(func)!r}"
            )
        super().__init__(**kwargs)
        self._func = func
        self._invalid = TransformNode._INVALID_FULL
    @property 
    def _invalid(self):
        return TransformNode._INVALID_FULL

    @_invalid.setter
    def _invalid(self, value):
        pass

    def _resolve(self) -> Transform:
        t = self._func()
        if not isinstance(t, Transform):
            raise TypeError(
                f"IndirectTransform func must return a Transform, "
                f"got {type(t)!r}"
            )
        return t

    def transform(self, values):
        return self._resolve().transform(values)

    def transform_affine(self, values):
        return self._resolve().transform_affine(values)

    def transform_non_affine(self, values):
        return self._resolve().transform_non_affine(values)

    def get_affine(self):
        return self._resolve().get_affine()

    def inverted(self) -> Transform:
        return self._resolve().inverted()

    def __repr__(self) -> str:
        return f"IndirectTransform({self._func!r})"

def artist_bbox_transform(artist) -> IndirectTransform:
    def _factory() -> Transform:
        renderer = artist.get_figure()._get_renderer()
        bbox = artist.get_window_extent(renderer)
        return BboxTransformTo(bbox)

    return IndirectTransform(_factory)

def _run_unit_tests():
    print("Running unit tests...")
    passed = 0
    fixed = mtransforms.Affine2D().scale(3)
    it = IndirectTransform(lambda: fixed)
    result = it.transform(np.array([[1.0, 2.0]]))
    assert np.allclose(result, [[3.0, 6.0]]), f"Test 1 failed: {result}"
    passed += 1
    print("  [PASS] Test 1: basic transform delegation")

    # 2: always invalid
    assert it._invalid == TransformNode._INVALID_FULL
    passed += 1
    print("  [PASS] Test 2: _invalid is always _INVALID_FULL")
    it._invalid = TransformNode._VALID
    assert it._invalid == TransformNode._INVALID_FULL
    passed += 1
    print("  [PASS] Test 3: _invalid cannot be set to _VALID")
    try:
        IndirectTransform("not callable")
        assert False, "Should have raised TypeError"
    except TypeError:
        passed += 1
        print("  [PASS] Test 4: TypeError on non-callable func")
    bad = IndirectTransform(lambda: "oops")
    try:
        bad.transform(np.array([[0.0, 0.0]]))
        assert False, "Should have raised TypeError"
    except TypeError:
        passed += 1
        print("  [PASS] Test 5: TypeError when func returns non-Transform")
    scale2 = mtransforms.Affine2D().scale(2)
    it2 = IndirectTransform(lambda: scale2)
    inv = it2.inverted()
    result = inv.transform(np.array([[4.0, 6.0]]))
    assert np.allclose(result, [[2.0, 3.0]]), f"Test 6 failed: {result}"
    passed += 1
    print("  [PASS] Test 6: inverted() delegates correctly")
    aff = it2.get_affine()
    assert aff is not None
    passed += 1
    print("  [PASS] Test 7: get_affine() returns a value")

    tlist = [mtransforms.Affine2D().scale(1), mtransforms.Affine2D().scale(5)]
    idx = [0]
    def switcher():
        t = tlist[idx[0]]
        idx[0] = 1
        return t
    it3 = IndirectTransform(switcher)
    r1 = it3.transform(np.array([[1.0, 1.0]]))
    r2 = it3.transform(np.array([[1.0, 1.0]]))
    assert np.allclose(r1, [[1.0, 1.0]]) and np.allclose(r2, [[5.0, 5.0]])
    passed += 1
    print("  [PASS] Test 8: updates when func returns different transform each call")

    assert it.input_dims == 2 and it.output_dims == 2
    passed += 1
    print("  [PASS] Test 9: input_dims=2, output_dims=2")

    offset = mtransforms.Affine2D().translate(10, 20)
    composed = IndirectTransform(lambda: mtransforms.Affine2D().scale(2)) + offset
    result = composed.transform(np.array([[1.0, 1.0]]))
    assert np.allclose(result, [[12.0, 22.0]]), f"Test 10 failed: {result}"
    passed += 1
    print("  [PASS] Test 10: composes with + operator")

    print(f"\nAll {passed}/10 tests passed on matplotlib {matplotlib.__version__}")
    return passed

def demo_annotate_relative_to_text():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Demo 1 — annotate relative to a Text artist's bbox", fontsize=11)

    base = ax.text(5, 5, "Base artist",
                   ha="center", va="center", fontsize=13, color="steelblue",
                   bbox=dict(boxstyle="round,pad=0.4", fc="#dce9f5", ec="steelblue", lw=1.5))

    it = artist_bbox_transform(base)

    ax.annotate(
        "top-centre\n(unit: 0.5, 1.0)",
        xy=(0.5, 1.0), xycoords=it,
        xytext=(0, 18), textcoords="offset points",
        ha="center", fontsize=9, color="dimgray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="lightgray"),
    )
    ax.annotate(
        "bottom-right\n(unit: 1.0, 0.0)",
        xy=(1.0, 0.0), xycoords=it,
        xytext=(20, -20), textcoords="offset points",
        ha="left", fontsize=9, color="dimgray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="lightgray"),
    )

    plt.tight_layout()
    out = os.path.join(HERE, "demo1_annotate_relative.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Saved {out}")

def demo_blended_positioning():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Demo 2 — blended positioning (issue #22223 use case)", fontsize=11)

    artist_a = ax.text(2, 2, "Artist A", ha="center", va="center", fontsize=12,
                       color="darkgreen",
                       bbox=dict(boxstyle="round,pad=0.3", fc="#d8f0e0", ec="darkgreen"))
    artist_b = ax.text(8, 8, "Artist B", ha="center", va="center", fontsize=12,
                       color="darkorange",
                       bbox=dict(boxstyle="round,pad=0.3", fc="#fde8cc", ec="darkorange"))

    def _midpoint_transform() -> Transform:
        renderer = fig._get_renderer()
        bbox_a = artist_a.get_window_extent(renderer)
        bbox_b = artist_b.get_window_extent(renderer)
        mid_x = (bbox_a.x0 + bbox_a.x1) / 2
        mid_y = (bbox_b.y0 + bbox_b.y1) / 2
        mid_bbox = Bbox([[mid_x - 1, mid_y - 1], [mid_x + 1, mid_y + 1]])
        return BboxTransformTo(mid_bbox)

    mid_it = IndirectTransform(_midpoint_transform)
    ax.annotate(
        "Midpoint\n(x from A, y from B)",
        xy=(0.5, 0.5), xycoords=mid_it,
        ha="center", va="center", fontsize=9, color="purple",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f0e8fa", ec="purple", lw=1.2),
    )
    ax.plot([2, 8], [2, 8], "--", color="#cccccc", lw=1, zorder=0)

    plt.tight_layout()
    out = os.path.join(HERE, "demo2_blended.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Saved {out}")

def demo_bar_labels():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Demo 3 — bar labels via IndirectTransform", fontsize=11)

    values = [4, 7, 3, 6, 9]
    colors = ["#5b8db8", "#e07b54", "#6aab72", "#c479b2", "#e8c44a"]
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        it = artist_bbox_transform(bar)
        ax.annotate(
            str(val),
            xy=(0.5, 1.0), xycoords=it,
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="dimgray",
        )

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri"])
    ax.set_ylabel("Count")
    ax.set_ylim(0, 11)
    plt.tight_layout()
    out = os.path.join(HERE, "demo3_bar_labels.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Saved {out}")



if __name__ == "__main__":
    _run_unit_tests()
    print()
    demo_annotate_relative_to_text()
    demo_blended_positioning()
    demo_bar_labels()
    print(f"\n10/10 tests passed. 3 demo images saved to: {HERE}")
    print(f"Matplotlib version: {matplotlib.__version__}")