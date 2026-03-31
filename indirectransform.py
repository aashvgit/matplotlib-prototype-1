from __future__ import annotations
import os, inspect
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.transforms import (
    Transform, TransformNode, BboxTransformTo,
    blended_transform_factory, Bbox, BboxBase,
)
from matplotlib.artist import Artist
from typing import Callable, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))

class IndirectTransform(Transform):
    input_dims = 2
    output_dims = 2

    def __init__(self, func: Callable[[], Transform], **kwargs):
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func)!r}")
        super().__init__(**kwargs)
        self._func = func

    @property
    def _invalid(self):
        return TransformNode._INVALID_FULL

    @_invalid.setter
    def _invalid(self, value):
        pass  # absorb all parent invalidation writes

    def _resolve(self) -> Transform:
        t = self._func()
        if not isinstance(t, Transform):
            raise TypeError(f"func must return a Transform, got {type(t)!r}")
        return t

    def transform(self, values):          return self._resolve().transform(values)
    def transform_affine(self, values):   return self._resolve().transform_affine(values)
    def transform_non_affine(self, values): return self._resolve().transform_non_affine(values)
    def get_affine(self):                 return self._resolve().get_affine()
    def inverted(self):                   return self._resolve().inverted()
    def __repr__(self):                   return f"IndirectTransform({self._func!r})"


class BboxUnionTransform(IndirectTransform):
    def __init__(self, artists: Sequence[Artist], **kwargs):
        if not artists:
            raise ValueError("BboxUnionTransform requires at least one artist")
        self._artists = list(artists)

        def _union():
            renderer = self._artists[0].get_figure()._get_renderer()
            bboxes = [a.get_window_extent(renderer) for a in self._artists]
            return BboxTransformTo(Bbox.union(bboxes))

        super().__init__(_union, **kwargs)

    def __repr__(self): return f"BboxUnionTransform({self._artists!r})"


def artist_bbox_transform(artist: Artist) -> IndirectTransform:
    def _f():
        renderer = artist.get_figure()._get_renderer()
        return BboxTransformTo(artist.get_window_extent(renderer))
    return IndirectTransform(_f)

def show_refactor():
    import matplotlib.text as mtext
    src = inspect.getsource(mtext._AnnotationBase._get_xy_transform)
    lines = src.split('\n')
    print("=" * 65)
    print("  Exact _get_xy_transform refactor (lib/matplotlib/text.py)")
    print("=" * 65)
    print(f"\nReal source: {len(lines)} lines total\n")
    print("BRANCHES THAT CHANGE:")
    print()
    print("  BEFORE (Artist branch, line ~7 of method):")
    print("    elif isinstance(coords, Artist):")
    print("        bbox = coords.get_window_extent(renderer)  # eager")
    print("        return BboxTransformTo(bbox)")
    print()
    print("  AFTER:")
    print("    elif isinstance(coords, Artist):")
    print("        return artist_bbox_transform(coords)  # lazy IndirectTransform")
    print()
    print("  BEFORE (callable branch, line ~4 of method):")
    print("    elif callable(coords):")
    print("        tr = coords(renderer)  # eager — called immediately")
    print("        if isinstance(tr, BboxBase): return BboxTransformTo(tr)")
    print("        elif isinstance(tr, Transform): return tr")
    print()
    print("  AFTER:")
    print("    elif callable(coords):")
    print("        def _wrap():")
    print("            tr = coords(renderer)")
    print("            if isinstance(tr, BboxBase): return BboxTransformTo(tr)")
    print("            elif isinstance(tr, Transform): return tr")
    print("            else: raise TypeError(...)")
    print("        return IndirectTransform(_wrap)  # lazy")
    print()
    print("BRANCHES UNCHANGED:")
    print("  - isinstance(coords, tuple)  → blended_transform_factory already")
    print("    accepts IndirectTransform since it IS a Transform")
    print("  - isinstance(coords, BboxBase) → still returns BboxTransformTo")
    print("  - isinstance(coords, Transform) → returned directly (already lazy)")
    print("  - String dispatch → unchanged")
    print("=" * 65)


def run_tests():
    print("Running unit tests...")
    p = 0

    def ok(n, msg):
        nonlocal p; p += 1
        print(f"  [PASS] Test {n}: {msg}")

    t_scale3 = mtransforms.Affine2D().scale(3)
    it = IndirectTransform(lambda: t_scale3)

    assert np.allclose(it.transform(np.array([[1.,2.]])), [[3.,6.]])
    ok(1, "basic delegation")

    assert it._invalid == TransformNode._INVALID_FULL
    ok(2, "_invalid always _INVALID_FULL")

    it._invalid = TransformNode._VALID
    assert it._invalid == TransformNode._INVALID_FULL
    ok(3, "_invalid setter is no-op")

    try: IndirectTransform("x"); assert False
    except TypeError: ok(4, "TypeError on non-callable func")

    try: IndirectTransform(lambda: "x").transform(np.array([[0.,0.]])); assert False
    except TypeError: ok(5, "TypeError when func returns non-Transform")

    inv = IndirectTransform(lambda: mtransforms.Affine2D().scale(2)).inverted()
    assert np.allclose(inv.transform(np.array([[4.,6.]])), [[2.,3.]])
    ok(6, "inverted() delegates correctly")

    assert IndirectTransform(lambda: mtransforms.Affine2D().scale(2)).get_affine() is not None
    ok(7, "get_affine() returns value")

    tlist = [mtransforms.Affine2D().scale(1), mtransforms.Affine2D().scale(5)]
    idx = [0]
    def sw():
        t = tlist[idx[0]]; idx[0] = 1; return t
    it3 = IndirectTransform(sw)
    r1 = it3.transform(np.array([[1.,1.]])); r2 = it3.transform(np.array([[1.,1.]]))
    assert np.allclose(r1,[[1.,1.]]) and np.allclose(r2,[[5.,5.]])
    ok(8, "live updates on each call")

    assert it.input_dims == 2 and it.output_dims == 2
    ok(9, "input_dims=output_dims=2")

    composed = IndirectTransform(lambda: mtransforms.Affine2D().scale(2)) + \
               mtransforms.Affine2D().translate(10,20)
    assert np.allclose(composed.transform(np.array([[1.,1.]])), [[12.,22.]])
    ok(10, "composes with + operator")

    fig, ax = plt.subplots()
    b1 = ax.text(1,1,"A"); b2 = ax.text(3,3,"B"); fig.canvas.draw()
    ut = BboxUnionTransform([b1, b2])
    assert ut.transform(np.array([[0.5,0.5]])).shape == (1,2)
    plt.close()
    ok(11, "BboxUnionTransform returns display coords")

    tx = IndirectTransform(lambda: mtransforms.Affine2D().scale(2,1))
    ty = IndirectTransform(lambda: mtransforms.Affine2D().scale(1,3))
    bl = blended_transform_factory(tx, ty)
    assert np.allclose(bl.transform(np.array([[2.,2.]])), [[4.,6.]])
    ok(12, "tuple xycoords: blended IndirectTransforms compose correctly")

    print(f"\nAll {p}/12 tests passed on matplotlib {matplotlib.__version__}")


def demo1_four_corners():
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    ax.set_title("Demo 1 — Annotate any unit-square position on an artist", fontsize=11, pad=12)
    base = ax.text(5,5,"Base artist",ha="center",va="center",fontsize=13,color="#1B5FAA",
                   bbox=dict(boxstyle="round,pad=0.5",fc="#EEF4FB",ec="#1B5FAA",lw=1.8))
    it = artist_bbox_transform(base)
    for xy,lbl,off,ha in [
        ((0.5,1.0),"top-centre\n(0.5,1.0)",(0,16),"center"),
        ((1.0,0.5),"right-centre\n(1.0,0.5)",(14,0),"left"),
        ((0.5,0.0),"bottom-centre\n(0.5,0.0)",(0,-16),"center"),
        ((0.0,0.5),"left-centre\n(0.0,0.5)",(-14,0),"right"),
    ]:
        ax.annotate(lbl,xy=xy,xycoords=it,xytext=off,textcoords="offset points",
                    ha=ha,fontsize=8.5,color="dimgray",
                    arrowprops=dict(arrowstyle="->",color="#888888",lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.25",fc="#f8f8f8",ec="#cccccc"))
    plt.tight_layout()
    out = os.path.join(HERE,"demo1_four_corners.png")
    plt.savefig(out,dpi=150); plt.close(); print(f"Saved {out}")


def demo2_blended():
    fig,ax = plt.subplots(figsize=(8,5))
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    ax.set_title("Demo 2 — Blended: x from Artist A, y from Artist B (issue #22223 use case)",fontsize=11,pad=12)
    ta = ax.text(2,2,"Artist A",ha="center",va="center",fontsize=12,color="darkgreen",
                 bbox=dict(boxstyle="round,pad=0.35",fc="#d8f0e0",ec="darkgreen"))
    tb = ax.text(8,8,"Artist B",ha="center",va="center",fontsize=12,color="darkorange",
                 bbox=dict(boxstyle="round,pad=0.35",fc="#fde8cc",ec="darkorange"))
    def _bl():
        r = fig._get_renderer()
        ba = ta.get_window_extent(r); bb = tb.get_window_extent(r)
        mx = (ba.x0+ba.x1)/2; my = (bb.y0+bb.y1)/2
        return BboxTransformTo(Bbox([[mx-1,my-1],[mx+1,my+1]]))
    ax.annotate("Blended\nx←A, y←B",xy=(0.5,0.5),xycoords=IndirectTransform(_bl),
                ha="center",va="center",fontsize=9,color="#6A0D91",
                bbox=dict(boxstyle="round,pad=0.4",fc="#F5EAFB",ec="#6A0D91",lw=1.4))
    ax.annotate("",xy=(8,8),xytext=(2,2),
                arrowprops=dict(arrowstyle="-",color="#cccccc",linestyle="dashed",lw=1))
    plt.tight_layout()
    out = os.path.join(HERE,"demo2_blended.png")
    plt.savefig(out,dpi=150); plt.close(); print(f"Saved {out}")


def demo3_bbox_union():
    fig,ax = plt.subplots(figsize=(8,4))
    ax.set_title("Demo 3 — BboxUnionTransform: group label above N bars",fontsize=11,pad=12)
    groups = {"Group A":([0,1,2],[3,6,4],"#5b8db8"),
              "Group B":([4,5,6],[7,5,8],"#e07b54"),
              "Group C":([8,9,10],[2,4,3],"#6aab72")}
    for lbl,(xs,ys,col) in groups.items():
        bars = ax.bar(xs,ys,color=col,edgecolor="white",lw=1.2,width=0.8)
        ut = BboxUnionTransform(list(bars))
        ax.annotate(lbl,xy=(0.5,1.0),xycoords=ut,xytext=(0,8),textcoords="offset points",
                    ha="center",fontsize=10,fontweight="bold",color=col,
                    bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=col,lw=1.2))
    ax.set_xticks([]); ax.set_ylim(0,11)
    plt.tight_layout()
    out = os.path.join(HERE,"demo3_bbox_union.png")
    plt.savefig(out,dpi=150); plt.close(); print(f"Saved {out}")


def demo4_bar_labels():
    fig,ax = plt.subplots(figsize=(7,4))
    ax.set_title("Demo 4 — Per-bar labels via artist_bbox_transform",fontsize=11,pad=12)
    vals = [4,7,3,6,9,5,8]
    cols = ["#5b8db8","#e07b54","#6aab72","#c479b2","#e8c44a","#76b7b2","#ff9da7"]
    bars = ax.bar(range(len(vals)),vals,color=cols,edgecolor="white",lw=1.2)
    for bar,val in zip(bars,vals):
        ax.annotate(str(val),xy=(0.5,1.0),xycoords=artist_bbox_transform(bar),
                    xytext=(0,5),textcoords="offset points",
                    ha="center",va="bottom",fontsize=11,fontweight="bold",color="dimgray")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax.set_ylim(0,11)
    plt.tight_layout()
    out = os.path.join(HERE,"demo4_bar_labels.png")
    plt.savefig(out,dpi=150); plt.close(); print(f"Saved {out}")


if __name__ == "__main__":
    show_refactor()
    print()
    run_tests()
    print()
    demo1_four_corners()
    demo2_blended()
    demo3_bbox_union()
    demo4_bar_labels()
    print(f"\n12/12 tests. 4 demos. matplotlib {matplotlib.__version__}")
    print(f"Images saved to: {HERE}")
