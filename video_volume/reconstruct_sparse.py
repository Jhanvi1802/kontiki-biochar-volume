"""Video -> Volume pipeline, STEP 2 (local, CPU): sparse 3-D reconstruction.

Runs COLMAP Structure-from-Motion (via pycolmap) on the extracted frames to recover
camera poses + a sparse 3-D point cloud. Tuned for kiln videos:
  - SINGLE shared camera (all frames come from one phone) -> more constrained, robust.
  - Boosted SIFT (2x upscale + affine-invariant shape) -> more features on low-texture
    biochar and low-resolution video.

Usage:
    python reconstruct_sparse.py frames_dir work_dir [exhaustive|sequential]
"""
import sys, shutil
from pathlib import Path
import pycolmap


def run(image_dir, work_dir, matcher="exhaustive"):
    image_dir = Path(image_dir); work = Path(work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    db = work / "database.db"
    sparse = work / "sparse"; sparse.mkdir()
    n = len(list(image_dir.glob("*.jpg")))
    print(f"images: {n}")

    feo = pycolmap.FeatureExtractionOptions()
    applied = {}
    for k, v in [("max_num_features", 16384), ("first_octave", -1),
                 ("estimate_affine_shape", True), ("domain_size_pooling", False),
                 ("edge_threshold", 12.0), ("peak_threshold", 0.004)]:
        try:
            setattr(feo.sift, k, v); applied[k] = v
        except Exception:
            pass
    print("sift boosted:", applied)

    print("1/3 extract_features (SINGLE camera)...")
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              extraction_options=feo)
    print(f"2/3 matching: {matcher}...")
    if matcher == "sequential":
        po = pycolmap.SequentialPairingOptions()
        try: po.overlap = 20
        except Exception: pass
        pycolmap.match_sequential(db, pairing_options=po)
    else:
        pycolmap.match_exhaustive(db)
    print("3/3 incremental_mapping...")
    maps = pycolmap.incremental_mapping(db, image_dir, sparse)

    recs = list(maps.values()) if isinstance(maps, dict) else list(maps)
    if not recs:
        print("RECONSTRUCTION FAILED: no model built.")
        return None
    rec = max(recs, key=lambda r: r.num_reg_images())
    print(f"models built     : {len(recs)}")
    print(f"registered images: {rec.num_reg_images()} / {n}")
    print(f"3D points        : {rec.num_points3D()}")
    try:
        print(f"mean reproj error: {rec.compute_mean_reprojection_error():.3f} px")
    except Exception as e:
        print("reproj error n/a:", e)

    ply = work / "sparse.ply"
    rec.export_PLY(str(ply))
    rec.write(str(sparse))
    print("wrote", ply)
    return str(ply)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    matcher = sys.argv[3] if len(sys.argv) > 3 else "exhaustive"
    run(sys.argv[1], sys.argv[2], matcher)
