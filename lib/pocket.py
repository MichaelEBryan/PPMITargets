import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage

VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20, "SE": 1.90}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS",
               "TYR"}
DIRS = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]


def read_pdb(path):
    xyz, plddt, resid, resn, elem = [], [], [], [], []
    for ln in open(path):
        if not ln.startswith("ATOM"):
            continue
        xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        try:
            plddt.append(float(ln[60:66]))
        except ValueError:
            plddt.append(np.nan)
        resid.append(int(ln[22:26]))
        resn.append(ln[17:20].strip())
        e = ln[76:78].strip().upper()
        elem.append(e if e else ln[12:16].strip()[0].upper())
    return (np.array(xyz), np.array(plddt), np.array(resid),
            np.array(resn), np.array(elem))


def _psp(occ, spacing, reach):
    steps = max(1, int(round(reach / spacing)))
    events = np.zeros(occ.shape, np.int8)
    for d in DIRS:
        fwd = np.zeros(occ.shape, bool)
        bwd = np.zeros(occ.shape, bool)
        cur_f, cur_b = occ.copy(), occ.copy()
        nd = tuple(-v for v in d)
        for _ in range(steps):
            cur_f = np.roll(cur_f, d, axis=(0, 1, 2))
            cur_b = np.roll(cur_b, nd, axis=(0, 1, 2))
            fwd |= cur_f
            bwd |= cur_b
        events += (fwd & bwd).astype(np.int8)
    return events


def find_pockets(xyz, resn, resid, elem, spacing=0.8, probe=1.4, reach=10.0,
                 min_psp=5, min_volume=100.0, n_report=3,
                 max_grid=90_000_000):
    rad = np.array([VDW.get(e, 1.70) for e in elem]) + probe
    lo, hi = xyz.min(0) - 3.0, xyz.max(0) + 3.0
    gx = [np.arange(lo[i], hi[i] + spacing, spacing) for i in range(3)]
    shape = tuple(len(g) for g in gx)
    if int(np.prod(shape)) > max_grid:
        spacing *= 1.5
        gx = [np.arange(lo[i], hi[i] + spacing, spacing) for i in range(3)]
        shape = tuple(len(g) for g in gx)
        if int(np.prod(shape)) > max_grid:
            return []

    occ = np.zeros(shape, bool)
    for c, r in zip(xyz, rad):
        i0 = np.maximum(((c - r - lo) / spacing).astype(int), 0)
        i1 = np.minimum(((c + r - lo) / spacing).astype(int) + 1,
                        np.array(shape))
        if (i1 <= i0).any():
            continue
        sub = np.stack(np.meshgrid(
            *[np.arange(i0[k], i1[k]) for k in range(3)], indexing="ij"), -1)
        pts = lo + sub * spacing
        d2 = ((pts - c) ** 2).sum(-1)
        m = d2 < r * r
        idx = sub[m]
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    free = ~occ
    events = _psp(occ, spacing, reach)
    pocket = free & (events >= min_psp)
    if not pocket.any():
        return []

    lab, n = ndimage.label(pocket)
    sizes = ndimage.sum(pocket, lab, range(1, n + 1)) * spacing ** 3
    order = np.argsort(-sizes)[:n_report]
    surf = cKDTree(xyz)
    out = []
    for k in order:
        if sizes[k] < min_volume:
            continue
        sel = lab == (k + 1)
        idx = np.argwhere(sel)
        pts = lo + idx * spacing
        lining_sets = surf.query_ball_point(pts, 5.0)
        ai = sorted({a for L in lining_sets for a in L})
        lin = sorted(set(zip(resid[ai], resn[ai])))
        hyd = (np.mean([rn in HYDROPHOBIC for _, rn in lin])
               if lin else np.nan)
        d_surf, _ = surf.query(pts)
        centre = xyz.mean(0)
        rg = float(np.sqrt(((xyz - centre) ** 2).sum(1).mean()))
        depth = float(rg - np.linalg.norm(pts.mean(0) - centre))
        out.append(dict(
            volume=float(sizes[k]),
            enclosure=float(events[sel].mean() / 7.0),
            n_lining=len(lin),
            hydrophobic_fraction=float(hyd),
            mean_probe_clearance=float(d_surf.mean()),
            depth_below_surface=depth,
            centre=[float(v) for v in pts.mean(0)],
            lining=[f"{rn}{ri}" for ri, rn in lin],
            points=pts))
    return out
