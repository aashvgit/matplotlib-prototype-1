IndirectTransform — GSoC 2026 Prototype
========================================
Proposed addition to lib/matplotlib/transforms.py
Tested against matplotlib 3.10.8

Run with:
    python indirect_transform.py

Saves demo images to the same folder as this script.


# Core implementation


IndirectTransform(Transform):A Transform that lazily delegates to a callable at draw time.

    Parameters
    ----------
    func : Callable[[], Transform]
        Zero-argument callable returning a `.Transform`.  Called on every
        render pass so it always reflects the *current* state of whatever
        artist or object it closes over.

    Notes
    -----
    Stays permanently invalid (``_invalid = TransformNode._INVALID_FULL``)
    because the wrapped callable may return a *different* Transform object
    on every call — for example when an artist's bounding box changes after
    a resize.  Caching would produce stale coordinates.

    ``set_children`` is never called because there is no stable child node
    whose invalidation should propagate upward.

    Examples
    --------
    Annotate at the top-centre of a Text artist:

    >>> it = artist_bbox_transform(my_label)
    >>> ax.annotate("hi", xy=(0.5, 1.0), xycoords=it,
    ...             xytext=(0, 10), textcoords="offset points")


# Convenience factory
artist_bbox_transform():Return an IndirectTransform tracking *artist*'s window-extent bbox.

    Maps unit-square coordinates ``(x, y) ∈ [0,1]²`` to the artist's
    *current* bounding box in display (pixel) coordinates.  Re-evaluates
    on every render pass — correctly follows the artist after figure resizes.

    Parameters
    ----------
    artist : `.Artist`
        The artist whose bounding box to track.

    Returns
    -------
    IndirectTransform

    Examples
    --------
    >>> t = artist_bbox_transform(my_label)
    >>> ax.annotate("see this", xy=(0.5, 1.0), xycoords=t,
    ...             xytext=(0, 12), textcoords="offset points",
    ...             arrowprops=dict(arrowstyle="->"))


# how to run
use: MPLBACKEND=Agg python indirectransform.py
