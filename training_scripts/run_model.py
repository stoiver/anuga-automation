#!/usr/bin/env python
# coding: utf-8
"""
run_model.py

MPI-safe parallel ANUGA runner using YAML config file.

- If pts/tris .npy are not previously generated, create from triangle_shp (*_USM.shp) on rank 0
- Build serial domain on rank 0, tag boundaries using US/DS lines, then distribute
"""

import os
import sys
import math
import yaml
import shutil
import traceback
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd

import matplotlib

import fiona
import pyproj
from shapely.geometry import shape, LineString
from shapely.ops import transform as shp_transform

import anuga
from anuga.shallow_water.boundaries import Time_stage_zero_momentum_boundary, Reflective_boundary

from utils.anuga_tools import anuga_tools as at


# -----------------------------------------------------------------------------
# MPI identifiers
# -----------------------------------------------------------------------------
MYID, NUMPROCS = anuga.myid, anuga.numprocs
ISROOT = (MYID == 0)


# -----------------------------------------------------------------------------
# Helpers
# onfig-driven instead of hardcoded
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

#directory when saving plots,output SWW etc
def finish_plot(plt, path=None, dpi=150):
    if path:
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close()

#plotting in WSL/headless runs, saves the figure and immediately closes it
def read_first_linestring_with_crs(path):
    with fiona.open(path) as src:
        crs = src.crs_wkt or src.crs
        for ft in src:
            g = ft["geometry"]
            if not g:
                continue
            if g["type"] == "LineString":
                return LineString(g["coordinates"]), crs
            if g["type"] == "MultiLineString":
                parts = [LineString(c) for c in g["coordinates"]]
                coords = [xy for part in parts for xy in part.coords]
                return LineString(coords), crs
    raise RuntimeError(f"No LineString in {path}")

#read US/DS boundary line (and its CRS)
def reproject_linestring(ls, src_crs, dst_epsg):
    if not src_crs:
        raise RuntimeError("Source shapefile has no CRS; re-save it with CRS.")
    dst_crs = pyproj.CRS.from_epsg(dst_epsg)
    src_crs = pyproj.CRS.from_user_input(src_crs)
    proj = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return shp_transform(proj, ls)

#DS bbox doesn’t overlap mesh bbox or it is likely CRS mismatch
def overlaps(b1, b2):
    (x1min, y1min, x1max, y1max) = b1
    (x2min, y2min, x2max, y2max) = b2
    return (x1min <= x2max and x2min <= x1max and y1min <= y2max and y2min <= y1max)

#Converts ANUGA’s (triangle_id, edge_id) into the two vertex indices of that edge/side
def edge_nodes(v_idx, e):
    a, b, c = v_idx
    return [(b, c), (c, a), (a, b)][e]

#Builds a list of all exterior edges to create a temporary domain to get exterior edges
#compare each exterior edge distance to US/DS line → assign inlet/outlet/exterior.
def boundary_edges_segments(pts, tris):
    """Return list of ((tri_id, e), seg_line, seg_len, mid_xy) for exterior edges."""
    tmp = anuga.Domain(coordinates=pts, vertices=tris)
    out = []
    for (tri_id, e) in tmp.boundary.keys():
        v = tris[tri_id]
        i, j = edge_nodes(v, e)
        p1 = tuple(pts[i])
        p2 = tuple(pts[j])
        seg = LineString([p1, p2])
        L = float(seg.length)
        mid = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
        out.append(((tri_id, e), seg, L, mid))
    return out

#make trs contiguous int64 (M,3), remove negatives/degenerate/duplicates
def sanitize_tris(tris, npts):
    
    tris = np.asarray(tris)

    if tris.dtype == object and tris.ndim == 1:
        tris = np.vstack(tris)

    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"Triangles must be (M,3); got {tris.shape} dtype={tris.dtype}")

    if not np.issubdtype(tris.dtype, np.integer):
        tris = tris.astype(np.int64, copy=False)
    else:
        tris = tris.astype(np.int64, copy=False)

    neg = (tris < 0).any(axis=1)
    tris = tris[~neg]

    oob = (tris.max(axis=1) >= npts)
    tris = tris[~oob]

    deg = (tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2])
    tris = tris[~deg]

    if tris.size:
        tris = np.unique(tris, axis=0)

    return np.ascontiguousarray(tris, dtype=np.int64)

#change the any clockwise triangles to counter-clockwise
def force_ccw(pts, tris):
    
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    tris = np.ascontiguousarray(tris, dtype=np.int64)

    def signed_area2(p, t):
        a, b, c = p[t[0]], p[t[1]], p[t[2]]
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    flip = np.fromiter((signed_area2(pts, t) < 0 for t in tris), count=tris.shape[0], dtype=bool)
    if flip.any():
        tmp = tris[flip, 1].copy()
        tris[flip, 1] = tris[flip, 2]
        tris[flip, 2] = tmp
        if ISROOT:
            print(f"Flipped {int(flip.sum())} triangles to CCW")
    return pts, tris

#Makes your code work across different ANUGA versions
def build_time_stage_boundary(domain, stage_func):
    """Signature-safe Time_stage_zero_momentum_boundary across ANUGA versions."""
    sig = inspect.signature(Time_stage_zero_momentum_boundary.__init__)
    params = sig.parameters
    if "function" in params:
        bc = Time_stage_zero_momentum_boundary(domain, function=stage_func)
    elif "stage" in params:
        bc = Time_stage_zero_momentum_boundary(domain, stage=stage_func)
    else:
        bc = Time_stage_zero_momentum_boundary(domain, stage_func)

    if not hasattr(bc, "function"):
        bc.function = stage_func
    return bc

#Makes your make oducible .npy files froms *_USM.shp, mesh_*_pts.npy and mesh_*_tris.npy for reuse
def generate_pts_tris_from_triangle_shp(tri_shp: str, pts_npy: str, tris_npy: str, dedup_tol: float = 1e-6):
    
    def decimals_from_tol(tol: float) -> int:
        if tol <= 0:
            return 8
        return max(0, int(round(-math.log10(tol))))

    decimals = decimals_from_tol(dedup_tol)

    def keyfun(x, y):
        return (round(float(x), decimals), round(float(y), decimals))

    nodes = []
    node_index = {}
    tri_indices = []

    def add_point(x, y):
        k = keyfun(x, y)
        idx = node_index.get(k)
        if idx is None:
            idx = len(nodes)
            nodes.append([float(k[0]), float(k[1])])
            node_index[k] = idx
        return idx

    def extract_triangle_points(geom):
        tris = []
        if geom is None:
            return tris
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            return tris

        for poly in polys:
            coords = list(poly.exterior.coords)
            if len(coords) >= 2 and coords[0] == coords[-1]:
                coords = coords[:-1]

            uniq = []
            seen = set()
            for (x, y) in coords:
                p = (x, y)
                if p not in seen:
                    uniq.append(p)
                    seen.add(p)
                if len(uniq) == 3:
                    break
            if len(uniq) == 3:
                tris.append(uniq)

        return tris

    with fiona.open(tri_shp) as src:
        crs_info = src.crs_wkt or src.crs
        print("Mesh triangle shapefile CRS:", crs_info)

        n_feat = 0
        n_tri = 0
        for ft in src:
            n_feat += 1
            if not ft or not ft.get("geometry"):
                continue
            geom = shape(ft["geometry"])
            for tri in extract_triangle_points(geom):
                i = add_point(tri[0][0], tri[0][1])
                j = add_point(tri[1][0], tri[1][1])
                k = add_point(tri[2][0], tri[2][1])
                tri_indices.append([i, j, k])
                n_tri += 1

    pts = np.asarray(nodes, dtype=np.float64)
    tris = np.asarray(tri_indices, dtype=np.int32)

    if pts.size == 0 or tris.size == 0:
        raise RuntimeError("No points/triangles extracted. Check *_USM.shp contains triangle polygons.")

    os.makedirs(os.path.dirname(pts_npy), exist_ok=True)
    np.save(pts_npy, pts)
    np.save(tris_npy, tris)

    np.savetxt(os.path.splitext(pts_npy)[0] + ".csv", pts, delimiter=",", header="x,y", comments="")
    np.savetxt(os.path.splitext(tris_npy)[0] + ".csv", tris, delimiter=",", header="i,j,k", fmt="%d", comments="")

    print(f"Features read: {n_feat} | Triangles extracted: {n_tri} | Unique nodes: {len(nodes)}")
    print("Saved:")
    print(" -", pts_npy)
    print(" -", tris_npy)


# -----------------------------------------------------------------------------
# Main function
#pipeline: config → building mesh/domain → tagging boundaries → running evolve (spin-up + main) → merging
# -----------------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # --- headless ---
    headless = bool(cfg.get("runtime", {}).get("headless", True))
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    if headless:
        plt.show = lambda *a, **kw: None

    # --- paths ---
    workshop_dir = os.path.abspath(cfg["paths"]["workshop_dir"])
    data_dir = os.path.join(workshop_dir, cfg["paths"]["data_dir"])
    model_inputs_dir = os.path.join(workshop_dir, cfg["paths"]["model_inputs_dir"])
    model_outputs_dir = os.path.join(workshop_dir, cfg["paths"]["model_outputs_dir"])
    model_visuals_dir = os.path.join(workshop_dir, cfg["paths"]["model_visuals_dir"])
    model_validation_dir = os.path.join(workshop_dir, cfg["paths"]["model_validation_dir"])

    if ISROOT:
        for d in [model_inputs_dir, model_outputs_dir, model_visuals_dir, model_validation_dir]:
            ensure_dir(d)
    anuga.barrier()

    # --- DEM ---
    f_DEM_tif = os.path.join(workshop_dir, cfg["dem"]["tif"])
    if ISROOT:
        with rio.open(f_DEM_tif) as r:
            print("DEM CRS:", r.crs)

    # ------------------------------------------------------------------
    # Mesh (inside main)
    # ------------------------------------------------------------------
    mesh_tri_shp = os.path.join(workshop_dir, cfg["mesh"]["triangle_shp"])
    pts_npy = os.path.join(workshop_dir, cfg["mesh"]["pts_npy"])
    tris_npy = os.path.join(workshop_dir, cfg["mesh"]["tris_npy"])
    dedup_tol = float(cfg["mesh"].get("dedup_tol", 1e-6))

    if ISROOT:
        need_build = (not os.path.exists(pts_npy)) or (not os.path.exists(tris_npy))
        if need_build:
            if not os.path.exists(mesh_tri_shp):
                raise FileNotFoundError(f"Triangle shapefile not found: {mesh_tri_shp}")
            print("Mesh npy files not found. Generating from triangle shapefile:")
            print(" -", mesh_tri_shp)
            generate_pts_tris_from_triangle_shp(mesh_tri_shp, pts_npy, tris_npy, dedup_tol=dedup_tol)
        else:
            print("Using existing mesh npy files:")
            print(" -", pts_npy)
            print(" -", tris_npy)

    anuga.barrier()

    # Load on all ranks
    pts = np.load(pts_npy, allow_pickle=False)
    tris = np.load(tris_npy, allow_pickle=True)

    pts = np.ascontiguousarray(pts, dtype=np.float64)
    tris = sanitize_tris(tris, npts=len(pts))
    pts, tris = force_ccw(pts, tris)

    # --- build SERIAL domain on root ---
    domain = None
    Zv = None

    if ISROOT:
        domain = anuga.Domain(coordinates=pts, vertices=tris, verbose=False)

        with rio.open(f_DEM_tif) as r:
            L, B, R, T = r.bounds
            inside = (pts[:, 0] >= L) & (pts[:, 0] <= R) & (pts[:, 1] >= B) & (pts[:, 1] <= T)
            if not inside.all():
                raise RuntimeError("Mesh is outside DEM bounds. Reproject mesh or DEM to match.")

            Zv = np.array([v[0] for v in r.sample(pts)], dtype=float)
            if r.nodata is not None:
                Zv[Zv == r.nodata] = np.nan
            if np.isnan(Zv).any():
                Zv[np.isnan(Zv)] = np.nanmin(Zv)

        domain.quantities["elevation"].set_values(Zv, location="vertices")
        elev_v = domain.quantities["elevation"].vertex_values
        print("Domain elevation stats (m):", float(elev_v.min()), float(elev_v.max()))

        # optional mesh plot
        try:
            with rio.open(f_DEM_tif) as r:
                dem = r.read(1, masked=True)
                extent = (r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top)
            tri_mpl = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
            z_cell = domain.get_quantity("elevation").centroid_values
            fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
            ax.imshow(dem, extent=extent, origin="upper", alpha=0.5)
            pc = ax.tripcolor(tri_mpl, facecolors=z_cell, cmap="terrain", shading="flat")
            ax.triplot(tri_mpl, linewidth=0.15, alpha=0.35, color="k")
            plt.colorbar(pc, ax=ax, fraction=0.04, label="Elevation (m)")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("Interpolated Elevation on Mesh")
            plt.tight_layout()
            finish_plot(plt, os.path.join(model_visuals_dir, "DEM_meshed.png"))
        except Exception as e:
            print(f"(Skipping DEM_meshed plot: {e})")

    anuga.barrier()

    # --- simulation times ---
    sim_starttime = pd.Timestamp(cfg["simulation"]["starttime_utc"], tz="UTC")
    sim_endtime = pd.Timestamp(cfg["simulation"]["endtime_utc"], tz="UTC")
    spinup_endtime = pd.Timestamp(cfg["simulation"]["spinup_endtime_utc"], tz="UTC")

    timestep_sec = float(cfg["simulation"]["timestep_sec"])
    yieldstep_sec = float(cfg["simulation"]["yieldstep_sec"])
    outputstep_sec = float(cfg["simulation"]["outputstep_sec"])

    sim_total_duration = (sim_endtime - sim_starttime)
    sim_starttime_str = str(sim_starttime)[0:19].replace("-", "").replace(" ", "").replace(":", "")
    domain_name = cfg["model"]["domain_name"]
    model_name = f"{sim_starttime_str}_{domain_name}_{sim_total_duration.days}_days"

    if ISROOT:
        print("Simulation time from", sim_starttime, "to", sim_endtime, "| total duration:", sim_total_duration)

    discharge_csv = os.path.join(workshop_dir, cfg["gauges"]["discharge_csv"])
    level_csv = os.path.join(workshop_dir, cfg["gauges"]["level_csv"])

    t_start_py = sim_starttime.to_pydatetime()
    t_end_py = sim_endtime.to_pydatetime()

    #these functions are inside utils folder
    discharge_function = at.GenerateHydrograph(
        filename=discharge_csv,
        smoothing=False,
        t_start=t_start_py,
        t_end=t_end_py,
        t_step=timestep_sec,
        progressive=False,
    )

    level_function = at.GenerateTideGauge(
        filename=level_csv,
        t_start=t_start_py,
        t_end=t_end_py,
        t_step=timestep_sec,
        offset=0,
        smoothing=False,
        smoothing_span=0.1,
    )

    discharge_transect_shp = os.path.join(workshop_dir, cfg["gauges"]["discharge_transect_shp"])
    discharge_transect_gdf = gpd.read_file(discharge_transect_shp)
    discharge_loc = np.asarray(discharge_transect_gdf.geometry.iloc[0].xy).T.tolist()

    # --- boundary tagging on root ---
    if ISROOT:
        bcfg = cfg["boundaries"]
        mesh_epsg = int(bcfg["mesh_epsg_for_tagging"])
        base_tol_m = float(bcfg["base_tol_m"])
        us_line_shp = os.path.join(workshop_dir, bcfg["us_line_shp"])
        ds_line_shp = os.path.join(workshop_dir, bcfg["ds_line_shp"])

        US_raw, US_crs = read_first_linestring_with_crs(us_line_shp)
        DS_raw, DS_crs = read_first_linestring_with_crs(ds_line_shp)

        US = reproject_linestring(US_raw, US_crs, mesh_epsg)
        DS = reproject_linestring(DS_raw, DS_crs, mesh_epsg)

        items = boundary_edges_segments(pts, tris)
        edge_lengths = np.array([L for _, _, L, _ in items])
        auto_tol = (3.0 * np.percentile(edge_lengths, 10)) if len(edge_lengths) else 5.0
        tol = min(base_tol_m, max(auto_tol, 1.0))

        dists = np.array([seg.distance(DS) for _, seg, _, _ in items])
        print(f"Min dist DS->any exterior edge: {dists.min():.3f} m (tagging tol = {tol:.3f} m)")

        boundary_map = {}
        n_inlet = 0
        n_outlet = 0

        for ((tri_id, e), seg, _, _) in items:
            dUS = seg.distance(US)
            dDS = seg.distance(DS)
            if dUS <= tol:
                boundary_map[(tri_id, e)] = "inlet"; n_inlet += 1
            elif dDS <= tol:
                boundary_map[(tri_id, e)] = "outlet"; n_outlet += 1
            else:
                boundary_map[(tri_id, e)] = "exterior"

        print(f"Tagged edges -> inlet: {n_inlet}, outlet: {n_outlet}, exterior: {len(items)-n_inlet-n_outlet}")

        if n_outlet == 0:
            tol2 = max(tol, min(base_tol_m, dists.min() * 1.25 + 1.0))
            print(f"Retry with larger tol = {tol2:.2f} m")
            boundary_map.clear()
            n_inlet = n_outlet = 0
            for ((tri_id, e), seg, _, _) in items:
                dUS = seg.distance(US)
                dDS = seg.distance(DS)
                if dUS <= tol2:
                    boundary_map[(tri_id, e)] = "inlet"; n_inlet += 1
                elif dDS <= tol2:
                    boundary_map[(tri_id, e)] = "outlet"; n_outlet += 1
                else:
                    boundary_map[(tri_id, e)] = "exterior"
            if n_outlet == 0:
                raise RuntimeError("no 'outlet' tagged: DS line not touching domain; extend DS line to cross boundary.")

        for (tri_id,e), tag in boundary_map.items():
            domain.boundary[(tri_id, e)] = tag

        print(f"Boundary tags assigned: {domain.get_boundary_tags()}")
        
        boundary_tags = domain.get_boundary_tags()
        domain.set_boundary({tag: None for tag in boundary_tags})

        domain.set_quantity("elevation", Zv, location="vertices")
        domain.set_quantity("stage", expression="elevation")
        domain.set_quantity("friction", float(cfg["model"]["friction"]), location="centroids")
        domain.set_starttime(0.0)
        domain.set_flow_algorithm(cfg["model"]["flow_algorithm"])
        domain.set_name(model_name)
        domain.set_low_froude(int(cfg["model"]["low_froude"]))
        domain.set_minimum_allowed_height(float(cfg["model"]["minimum_allowed_height"]))

        domain.print_statistics()
        
    else:
        domain = None

    # --- distribute the domain generated in rank 0
    if NUMPROCS > 1:
        if ISROOT:
            print(f"Distributing domain across {NUMPROCS} processes...")
        domain = anuga.distribute(domain)

    boundary_tags = domain.get_boundary_tags()
    print(f"Rank {MYID} boundary tags: {boundary_tags}")

    def stage_func(t):
        return float(level_function(t))

    Bc = {}
    if "outlet" in boundary_tags:
        Bc["outlet"] = build_time_stage_boundary(domain, stage_func)
    for tag in boundary_tags:
        if tag not in Bc:
            Bc[tag] = Reflective_boundary(domain)

    domain.set_boundary(Bc)

    inlet_ATC = anuga.Inlet_operator(domain, discharge_loc, Q=discharge_function(0.0))

    spinup_duration_sec = (spinup_endtime - sim_starttime).total_seconds()
    total_duration_sec = (sim_endtime - sim_starttime).total_seconds()

    anuga.barrier()


    try:
        for t in domain.evolve(yieldstep=yieldstep_sec, outputstep=outputstep_sec, finaltime=spinup_duration_sec):
            if inlet_ATC is not None:
                inlet_ATC.Q = discharge_function(t)
            if ISROOT:
                domain.print_timestepping_statistics()
            domain.report_water_volume_statistics()

        for t in domain.evolve(yieldstep=yieldstep_sec, outputstep=outputstep_sec, finaltime=total_duration_sec):
            if inlet_ATC is not None:
                inlet_ATC.Q = discharge_function(t)
            if ISROOT:
                domain.print_timestepping_statistics()
            domain.report_water_volume_statistics()

    except Exception:
        if ISROOT:
            traceback.print_exc()
        raise

    if ISROOT:
        print("Merging parallel sww files...")
    domain.sww_merge(delete_old=True)

    if ISROOT:
        merged_sww = domain.get_name() + ".sww"
        f_anuga_output_in = os.path.join(workshop_dir, merged_sww)
        f_anuga_output_out = os.path.join(model_outputs_dir, merged_sww)
        shutil.copy2(f_anuga_output_in, f_anuga_output_out)

    anuga.finalize()


if __name__ == "__main__":
    main()
