"""Flat depth-shaded rendering of a C-alpha trace, with a pocket cloud."""
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def ca_trace(xyz, resid, resn=None, atom_name=None):
    """One representative point per residue (the C-alpha where available)."""
    order, seen = [], set()
    for i, r in enumerate(resid):
        if r not in seen:
            seen.add(r)
            order.append(i)
    return xyz[order], np.array([resid[i] for i in order]), np.array(order)


def spline(P, n_per=6):
    """Catmull-Rom through the points, so the backbone reads as a ribbon."""
    if len(P) < 4:
        return P
    Q = np.vstack([P[0], P, P[-1]])
    out = []
    t = np.linspace(0, 1, n_per, endpoint=False)[:, None]
    for i in range(len(Q) - 3):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(Q[-2][None, :])
    return np.vstack(out)


def view_matrix(pts, focus=None, roll=0.0):
    """Principal axes of the structure, optionally re-aimed at a focus point."""
    c = pts.mean(0)
    U, S, Vt = np.linalg.svd(pts - c, full_matrices=False)
    R = Vt
    if focus is not None:
        f = focus - c
        f = f / (np.linalg.norm(f) + 1e-9)
        # make the depth axis point away from the focus, so the pocket faces us
        z = -f
        x = np.cross(R[0], z)
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(R[1], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
    if roll:
        cs, sn = np.cos(roll), np.sin(roll)
        Rz = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
        R = Rz @ R
    return c, R


def project(pts, c, R):
    return (pts - c) @ R.T


def draw_backbone(ax, P3, values=None, cmap=None, lw=(1.1, 3.4),
                  base_color="#2a5a8a", shade=(0.30, 1.0), zorder=3,
                  n_per=6):
    """Depth-shaded backbone. P3 is already in view coordinates."""
    S = spline(P3, n_per=n_per)
    seg = np.stack([S[:-1, :2], S[1:, :2]], axis=1)
    z = (S[:-1, 2] + S[1:, 2]) / 2
    zr = (z - z.min()) / (np.ptp(z) + 1e-9)
    widths = lw[0] + (lw[1] - lw[0]) * zr
    if values is not None and cmap is not None:
        v = np.interp(np.linspace(0, len(values) - 1, len(seg)),
                      np.arange(len(values)), values)
        cols = cmap(v)
        cols[:, :3] *= (shade[0] + (shade[1] - shade[0]) * zr)[:, None]
    else:
        from matplotlib.colors import to_rgba
        base = np.array(to_rgba(base_color))
        cols = np.tile(base, (len(seg), 1))
        cols[:, :3] = 1 - (1 - cols[:, :3]) * (
            shade[0] + (shade[1] - shade[0]) * zr)[:, None]
    order = np.argsort(z)
    lc = LineCollection(seg[order], linewidths=widths[order],
                        colors=cols[order], capstyle="round",
                        joinstyle="round", zorder=zorder)
    ax.add_collection(lc)
    return S


def draw_cloud(ax, pts2, color="#b07d3a", alpha=0.16, size=14, hull=True,
               zorder=6):
    """A pocket, drawn as a soft cloud with an outline."""
    ax.scatter(pts2[:, 0], pts2[:, 1], s=size, color=color, alpha=alpha,
               edgecolors="none", zorder=zorder)
    if hull and len(pts2) >= 3:
        from scipy.spatial import ConvexHull
        try:
            h = ConvexHull(pts2)
            v = pts2[h.vertices]
            ax.add_patch(PathPatch(Path(np.vstack([v, v[:1]])),
                                   facecolor=color, alpha=0.13,
                                   edgecolor=color, lw=1.1, zorder=zorder - 1))
        except Exception:
            pass


def frame(ax, pts2, pad=0.06):
    lo, hi = pts2.min(0), pts2.max(0)
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * (1 + pad)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
