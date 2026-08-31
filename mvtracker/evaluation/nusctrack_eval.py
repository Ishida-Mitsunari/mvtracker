"""NuscTrack / xTAP3D metrics (same protocol as BEVTracker tap_eval.py).

Ego 3D, fixed-meter thresholds, no TAPVid-3D scale rescale.
Clip records are concatenated and micro-averaged over tracks (All / Inst. / BG).
"""
from __future__ import annotations

import numpy as np

THRESHOLDS = (0.01, 0.04, 0.16, 0.64, 2.56)
PIXEL_THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0)
CAMERAS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
_SPLIT_SUFFIX = {
    "all": "",
    "instance": "_instance",
    "background": "_background",
}


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_leading_batch(*arrays):
    out = []
    for x in arrays:
        if x is None:
            out.append(None)
            continue
        x = _to_numpy(x)
        if x.ndim >= 1 and x.shape[0] == 1 and x.ndim >= 2:
            # (1, T, N, ...) or (1, N)
            if x.ndim >= 3 or (x.ndim == 2 and x.shape[0] == 1):
                x = x[0]
        out.append(x)
    return out


def _vis_from_pred(pred_v, vis_thresh, occ_as_visible):
    pred_v = np.asarray(pred_v)
    pred_v_fin = np.isfinite(pred_v)
    if occ_as_visible:
        vis = (pred_v < vis_thresh) & pred_v_fin
    else:
        vis = (pred_v >= vis_thresh) & pred_v_fin
    return vis, pred_v_fin


def compute_track_eval(
    pred_tracks,
    pred_vis,
    gt_tracks,
    gt_vis,
    valid,
    query_frames,
    is_background=None,
    empty_source=False,
    vis_thresh=0.5,
    occ_as_visible=False,
):
    """Per-track 3D metrics for one clip.

    pred_tracks: (T, N, 3) ego xyz. Do **not** nan_to_num beforehand.
    pred_vis: (T, N) any-view/query-cam vis, or (T, N, K) per-camera vis.
    gt_tracks: (T, N, 3)
    gt_vis: (T, N, K)
    valid: (T, N)
    query_frames: (N,)
    """
    pred_xyz, pred_v, gt_xyz, gt_v, valid, qf, is_bg = _squeeze_leading_batch(
        pred_tracks, pred_vis, gt_tracks, gt_vis, valid, query_frames, is_background
    )
    if pred_xyz.ndim == 4:
        pred_xyz = pred_xyz[0]
        pred_v = pred_v[0]
        gt_xyz = gt_xyz[0]
        gt_v = gt_v[0]
        valid = valid[0]
        if qf.ndim == 2:
            qf = qf[0]
        if is_bg is not None and is_bg.ndim == 2:
            is_bg = is_bg[0]
    qf = np.asarray(qf).reshape(-1).astype(np.int64)
    T, N, _ = pred_xyz.shape
    if gt_v.ndim == 2:
        gt_v = gt_v[..., None]
    K = gt_v.shape[-1]

    t_idx = np.arange(T)[:, None]
    after_query = t_idx > qf[None, :]
    valid = valid.astype(bool) & after_query
    if empty_source:
        valid = np.zeros_like(valid, dtype=bool)

    gt_fin = np.isfinite(gt_xyz).all(axis=-1)
    pred_fin = np.isfinite(pred_xyz).all(axis=-1)
    valid_gt = valid & gt_fin

    vis_gt = gt_v >= 0.5
    vis_gt_any = vis_gt.any(axis=-1)

    vis_pred_raw, pred_v_fin = _vis_from_pred(pred_v, vis_thresh, occ_as_visible)
    has_per_cam_pred = vis_pred_raw.ndim == 3 and vis_pred_raw.shape[-1] == K
    if vis_pred_raw.ndim == 2:
        vis_pred_any = vis_pred_raw
        vis_pred = None
    elif vis_pred_raw.ndim == 3 and vis_pred_raw.shape[-1] == 1:
        vis_pred_any = vis_pred_raw[..., 0]
        vis_pred = None
        has_per_cam_pred = False
    else:
        vis_pred = vis_pred_raw
        vis_pred_any = vis_pred.any(axis=-1)

    delta = pred_xyz - gt_xyz
    err = np.linalg.norm(
        np.where(pred_fin[..., None] & gt_fin[..., None], delta, np.nan),
        axis=-1,
    )

    taus = np.asarray(THRESHOLDS, dtype=np.float64)
    within = pred_fin[..., None] & (err[..., None] < taus[None, None, :])

    vis_loc = valid_gt & vis_gt_any
    n_vis = vis_loc.sum(axis=0).astype(np.float64)
    has_eval = valid_gt.any(axis=0)
    has_vis = n_vis > 0

    aj_tau = []
    apd_tau = []
    for i, _tau in enumerate(taus):
        w = within[..., i]
        tp = valid_gt & vis_pred_any & vis_gt_any & w
        fp = valid_gt & vis_pred_any & ((~vis_gt_any) | (~w))
        fn = valid_gt & vis_gt_any & ((~vis_pred_any) | (~w))
        denom = tp.sum(axis=0) + fp.sum(axis=0) + fn.sum(axis=0)
        a = np.divide(
            tp.sum(axis=0),
            denom,
            out=np.full(N, np.nan, dtype=np.float64),
            where=denom > 0,
        )
        aj_tau.append(a)
        hit = (w & vis_loc).sum(axis=0).astype(np.float64)
        d = np.divide(
            hit,
            n_vis,
            out=np.full(N, np.nan, dtype=np.float64),
            where=n_vis > 0,
        )
        apd_tau.append(d)
    aj = np.nanmean(np.stack(aj_tau, axis=0), axis=0)
    apd = np.nanmean(np.stack(apd_tau, axis=0), axis=0)

    mte = np.full(N, np.nan, dtype=np.float64)
    loc_mask = vis_loc & pred_fin
    for n in range(N):
        e_n = err[loc_mask[:, n], n]
        e_n = e_n[np.isfinite(e_n)]
        if e_n.size:
            mte[n] = float(np.median(e_n))

    n_valid = valid_gt.sum(axis=0).astype(np.float64)
    oa_ok = (vis_pred_any == vis_gt_any) & valid_gt
    oa_any = np.divide(
        oa_ok.sum(axis=0),
        n_valid,
        out=np.full(N, np.nan, dtype=np.float64),
        where=n_valid > 0,
    )

    oa_cam = np.full((N, K), np.nan, dtype=np.float64)
    cam_iou = np.full(N, np.nan, dtype=np.float64)
    if has_per_cam_pred and vis_pred is not None:
        for k in range(K):
            ok = (vis_pred[..., k] == vis_gt[..., k]) & valid_gt
            oa_cam[:, k] = np.divide(
                ok.sum(axis=0),
                n_valid,
                out=np.full(N, np.nan, dtype=np.float64),
                where=n_valid > 0,
            )
        inter = (vis_pred & vis_gt).sum(axis=-1).astype(np.float64)
        union = (vis_pred | vis_gt).sum(axis=-1).astype(np.float64)
        iou_t = np.divide(
            inter,
            union,
            out=np.zeros((T, N), dtype=np.float64),
            where=union > 0,
        )
        cam_iou_mask = valid_gt & vis_gt_any
        n_iou = cam_iou_mask.sum(axis=0)
        for n in range(N):
            if n_iou[n] > 0:
                cam_iou[n] = float(iou_t[cam_iou_mask[:, n], n].mean())

    n_nf = (valid_gt & vis_gt_any & (~pred_fin)).sum(axis=0).astype(np.float64)
    pct_nf = np.divide(n_nf, n_vis, out=np.zeros(N, dtype=np.float64), where=n_vis > 0)

    if is_bg is None:
        is_bg = np.zeros(N, dtype=bool)
    else:
        is_bg = np.asarray(is_bg).reshape(-1).astype(bool)
        if is_bg.size != N:
            is_bg = np.zeros(N, dtype=bool)

    return dict(
        aj=aj,
        apd=apd,
        mte=mte,
        oa_any=oa_any,
        cam_iou=cam_iou,
        oa_cam=oa_cam,
        is_background=is_bg,
        has_eval=has_eval,
        has_vis=has_vis,
        pct_nonfinite=pct_nf,
        n_vis=n_vis,
    )


def compute_query_cam_2d_eval(
    pred_xy,
    pred_vis,
    gt_xy,
    gt_vis_querycam,
    valid,
    query_frames,
    is_background=None,
    image_hw=(432, 768),
    vis_thresh=0.5,
    occ_as_visible=False,
    empty_source=False,
):
    """Query-camera 2D TAP (protocol §6.1). Coords scaled to 256x256."""
    pred_xy, pred_v, gt_xy, gt_v, valid, qf, is_bg = _squeeze_leading_batch(
        pred_xy, pred_vis, gt_xy, gt_vis_querycam, valid, query_frames, is_background
    )
    qf = np.asarray(qf).reshape(-1).astype(np.int64)
    if pred_xy.ndim == 4:
        pred_xy = pred_xy[0]
    if pred_v.ndim == 3 and pred_v.shape[0] == 1:
        pred_v = pred_v[0]
    if gt_xy.ndim == 4:
        gt_xy = gt_xy[0]
    if gt_v.ndim == 3 and gt_v.shape[0] == 1:
        gt_v = gt_v[0]
    if valid.ndim == 3:
        valid = valid[0]
    T, N, _ = pred_xy.shape
    h, w = float(image_hw[0]), float(image_hw[1])
    scale = np.array([256.0 / w, 256.0 / h], dtype=np.float64)
    pred_s = pred_xy.astype(np.float64) * scale
    gt_s = gt_xy.astype(np.float64) * scale

    t_idx = np.arange(T)[:, None]
    after_query = t_idx > qf[None, :]
    valid = valid.astype(bool) & after_query
    if empty_source:
        valid = np.zeros_like(valid, dtype=bool)

    gt_fin = np.isfinite(gt_s).all(axis=-1)
    pred_fin = np.isfinite(pred_s).all(axis=-1)
    valid_gt = valid & gt_fin
    vis_gt = np.asarray(gt_v) >= 0.5
    if vis_gt.ndim == 3:
        vis_gt = vis_gt.any(axis=-1)
    vis_pred, _ = _vis_from_pred(pred_v, vis_thresh, occ_as_visible)
    if vis_pred.ndim == 3:
        vis_pred = vis_pred.any(axis=-1)

    err = np.linalg.norm(
        np.where(pred_fin[..., None] & gt_fin[..., None], pred_s - gt_s, np.nan),
        axis=-1,
    )
    taus = np.asarray(PIXEL_THRESHOLDS, dtype=np.float64)
    within = pred_fin[..., None] & (err[..., None] < taus[None, None, :])
    vis_loc = valid_gt & vis_gt
    n_vis = vis_loc.sum(axis=0).astype(np.float64)
    has_eval = valid_gt.any(axis=0)
    has_vis = n_vis > 0

    aj_tau, apd_tau = [], []
    for i, _tau in enumerate(taus):
        w = within[..., i]
        tp = valid_gt & vis_pred & vis_gt & w
        fp = valid_gt & vis_pred & ((~vis_gt) | (~w))
        fn = valid_gt & vis_gt & ((~vis_pred) | (~w))
        denom = tp.sum(axis=0) + fp.sum(axis=0) + fn.sum(axis=0)
        aj_tau.append(
            np.divide(
                tp.sum(axis=0),
                denom,
                out=np.full(N, np.nan, dtype=np.float64),
                where=denom > 0,
            )
        )
        hit = (w & vis_loc).sum(axis=0).astype(np.float64)
        apd_tau.append(
            np.divide(hit, n_vis, out=np.full(N, np.nan, dtype=np.float64), where=n_vis > 0)
        )
    aj = np.nanmean(np.stack(aj_tau, axis=0), axis=0)
    apd = np.nanmean(np.stack(apd_tau, axis=0), axis=0)
    n_valid = valid_gt.sum(axis=0).astype(np.float64)
    oa = np.divide(
        ((vis_pred == vis_gt) & valid_gt).sum(axis=0),
        n_valid,
        out=np.full(N, np.nan, dtype=np.float64),
        where=n_valid > 0,
    )
    if is_bg is None:
        is_bg = np.zeros(N, dtype=bool)
    else:
        is_bg = np.asarray(is_bg).reshape(-1).astype(bool)
        if is_bg.size != N:
            is_bg = np.zeros(N, dtype=bool)
    return dict(
        aj_2d_querycam=aj,
        apd_2d_querycam=apd,
        oa_2d_querycam=oa,
        has_eval_2d=has_eval,
        has_vis_2d=has_vis,
        is_background=is_bg,
    )


def _mean_finite(x, mask):
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    if not m.any():
        return float("nan")
    return float(x[m].mean())


def summarize_track_eval(track_eval, prefix=""):
    return aggregate_track_evals([track_eval], prefix=prefix)


def aggregate_track_evals(records, prefix=""):
    if not records:
        return {}
    keys = (
        "aj",
        "apd",
        "mte",
        "oa_any",
        "cam_iou",
        "pct_nonfinite",
        "is_background",
        "has_eval",
        "has_vis",
        "n_vis",
    )
    cat = {}
    oa_cams = []
    for rec in records:
        oa_cams.append(np.asarray(rec["oa_cam"]))
    cat["oa_cam"] = np.concatenate(oa_cams, axis=0)
    for k in keys:
        cat[k] = np.concatenate([np.asarray(r[k]) for r in records], axis=0)

    has_2d = all("aj_2d_querycam" in r for r in records)
    if has_2d:
        for k in ("aj_2d_querycam", "apd_2d_querycam", "oa_2d_querycam", "has_eval_2d", "has_vis_2d"):
            cat[k] = np.concatenate([np.asarray(r[k]) for r in records], axis=0)

    out = {}
    splits = (
        ("all", cat["has_eval"]),
        ("instance", cat["has_eval"] & (~cat["is_background"])),
        ("background", cat["has_eval"] & cat["is_background"]),
    )
    for split, mask in splits:
        suf = _SPLIT_SUFFIX[split]
        n = int(mask.sum())
        out["n_tracks{}".format(suf)] = n
        out["n_tracks_vis{}".format(suf)] = int((mask & cat["has_vis"]).sum())
        out["average_jaccard{}".format(suf)] = _mean_finite(cat["aj"], mask)
        out["average_pts_within_thresh{}".format(suf)] = _mean_finite(cat["apd"], mask)
        out["MTE{}".format(suf)] = _mean_finite(cat["mte"], mask & cat["has_vis"])
        out["OA{}".format(suf)] = _mean_finite(cat["oa_any"], mask)
        out["camera_iou{}".format(suf)] = _mean_finite(cat["cam_iou"], mask & cat["has_vis"])
        vis_m = mask & cat["has_vis"]
        out["pct_nonfinite_pred{}".format(suf)] = _mean_finite(cat["pct_nonfinite"], vis_m)
        if has_2d:
            m2 = cat["has_eval_2d"] & (
                np.ones_like(mask) if split == "all" else
                (~cat["is_background"] if split == "instance" else cat["is_background"])
            )
            if split == "all":
                m2 = cat["has_eval_2d"]
            elif split == "instance":
                m2 = cat["has_eval_2d"] & (~cat["is_background"])
            else:
                m2 = cat["has_eval_2d"] & cat["is_background"]
            out["average_jaccard_2d_querycam{}".format(suf)] = _mean_finite(cat["aj_2d_querycam"], m2)
            out["average_pts_within_thresh_2d_querycam{}".format(suf)] = _mean_finite(
                cat["apd_2d_querycam"], m2
            )
            out["OA_2d_querycam{}".format(suf)] = _mean_finite(cat["oa_2d_querycam"], m2)
        if split == "all":
            for k, name in enumerate(CAMERAS):
                if k < cat["oa_cam"].shape[1]:
                    out["{}_OA".format(name)] = _mean_finite(cat["oa_cam"][:, k], mask)

    out["3D-AJ"] = out["average_jaccard"]
    out["APD"] = out["average_pts_within_thresh"]
    out["3D-AJ_instance"] = out["average_jaccard_instance"]
    out["APD_instance"] = out["average_pts_within_thresh_instance"]
    out["3D-AJ_background"] = out["average_jaccard_background"]
    out["APD_background"] = out["average_pts_within_thresh_background"]
    if has_2d:
        out["2D-AJ_querycam"] = out["average_jaccard_2d_querycam"]
        out["2D-APD_querycam"] = out["average_pts_within_thresh_2d_querycam"]
        out["2D-AJ_querycam_instance"] = out["average_jaccard_2d_querycam_instance"]
        out["2D-APD_querycam_instance"] = out["average_pts_within_thresh_2d_querycam_instance"]
        out["2D-AJ_querycam_background"] = out["average_jaccard_2d_querycam_background"]
        out["2D-APD_querycam_background"] = out["average_pts_within_thresh_2d_querycam_background"]
    if prefix:
        out = {prefix + k: v for k, v in out.items()}
    return out


def format_tap_metrics(metrics):
    lines = []
    n_all = metrics.get("n_tracks", 0)
    n_i = metrics.get("n_tracks_instance", 0)
    n_b = metrics.get("n_tracks_background", 0)
    lines.append("n_tracks All/Inst/BG: {} / {} / {}".format(n_all, n_i, n_b))

    def row(title, k):
        a = metrics.get(k, float("nan"))
        i = metrics.get(k + "_instance", float("nan"))
        b = metrics.get(k + "_background", float("nan"))
        lines.append("{} All/Inst/BG: {:.4f} / {:.4f} / {:.4f}".format(title, a, i, b))

    row("3D-AJ", "average_jaccard")
    row("APD", "average_pts_within_thresh")
    row("OA (any-view)", "OA")
    row("MTE", "MTE")
    row("camera-IoU", "camera_iou")
    row("pct_nonfinite_pred", "pct_nonfinite_pred")
    if "average_jaccard_2d_querycam" in metrics:
        row("2D-AJ (query-cam)", "average_jaccard_2d_querycam")
        row("2D-APD (query-cam)", "average_pts_within_thresh_2d_querycam")
        row("2D-OA (query-cam)", "OA_2d_querycam")
    oa_cam = ", ".join(
        "{}={:.4f}".format(name, metrics.get("{}_OA".format(name), float("nan")))
        for name in CAMERAS
    )
    lines.append("per-view OA (All): {}".format(oa_cam))
    return "\n".join(lines)


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return None
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        value = value.item()
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if np.isnan(value):
            return None
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def dump_tap_metrics(metrics, path_prefix, header_lines=None):
    import json
    import os

    path_prefix = os.path.splitext(os.path.abspath(path_prefix))[0]
    parent = os.path.dirname(path_prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)
    txt_path = path_prefix + ".txt"
    json_path = path_prefix + ".json"
    summary = format_tap_metrics(metrics)
    lines = []
    if header_lines:
        lines.extend(header_lines)
        lines.append("")
    lines.append(summary)
    lines.extend(["", "all scalars"])
    jsonable = {}
    for key in sorted(metrics.keys()):
        val = metrics[key]
        if isinstance(val, dict):
            continue
        lines.append("{}: {}".format(key, val))
        jsonable[key] = _jsonable(val)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w") as f:
        json.dump(jsonable, f, indent=2, sort_keys=True)
        f.write("\n")
    return txt_path, json_path
