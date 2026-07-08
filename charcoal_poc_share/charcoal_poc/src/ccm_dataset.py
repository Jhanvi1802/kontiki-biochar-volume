#!/usr/bin/env python
"""
Loader / index for the real benchmark track (CORSMAL C-CCM).

What C-CCM actually provides as ground truth (verified against the files):
  * container region mask  (soft Mask R-CNN map, 0..254; threshold to binarise)
  * filling TYPE   : empty / pasta / rice / water        (categorical)
  * filling LEVEL  : empty(0%) / half(50%) / full(90%)   (categorical, nominal %)
  * container ID + name + material + transparency
  * bbox, view, occlusion, scenario

What C-CCM does NOT provide (important, documented honestly):
  * container CAPACITY (mL)  -> so absolute volume in litres is unknown
  * filling MASS (g)         -> so absolute weight in kg is unknown
Therefore Track A scores SEGMENTATION (IoU vs container mask) and RELATIVE
filling-LEVEL estimation (against the nominal 0/50/90% labels). Absolute
volume/weight accuracy is carried by the synthetic track (Track B).

Granular fillings (pasta, rice) are the charcoal stand-ins.
"""
import os, json, cv2, numpy as np

FILL_TYPE = {0: "empty", 1: "pasta", 2: "rice", 3: "water"}
GRANULAR_TYPES = {1, 2}                       # pasta, rice = charcoal proxies
LEVEL_NAME = {0: "empty", 1: "half", 2: "full"}
LEVEL_PCT = {0: 0.0, 1: 50.0, 2: 90.0}       # nominal % of capacity (C-CCM def)
VIEW = {1: "left_side", 2: "right_side", 3: "robot"}
TRANSP = {0: "transparent", 1: "semi", 2: "opaque"}
MASK_THRESH = 128                            # binarise the soft container map


class CCM:
    def __init__(self, root):
        self.root = root
        d = json.load(open(os.path.join(root, "c_ccm_annotations.json")))
        self.containers = {c["id"]: c for c in d["containers"]}
        self.ann = {a["id"]: a for a in d["annotations"]}
        # which RGB frames are actually present locally (we hold one shard)
        rgb_dir = os.path.join(root, "rgb")
        self.available = sorted(int(f[:-4]) for f in os.listdir(rgb_dir)
                                if f.endswith(".png"))
        self.available = [i for i in self.available if i in self.ann]

    # ---- paths
    def rgb_path(self, i):  return os.path.join(self.root, "rgb",  f"{i:06d}.png")
    def mask_path(self, i): return os.path.join(self.root, "masks", f"{i:06d}.png")

    # ---- decoded record
    def record(self, i):
        a = self.ann[i]
        c = self.containers[a["containerID"]]
        return dict(
            id=i,
            container_id=a["containerID"], container=c["name"],
            material=c["material"], transparency=TRANSP[a["transparency"]],
            filling_type=FILL_TYPE[a["filling_type"]], filling_type_id=a["filling_type"],
            filling_level=LEVEL_NAME[a["filling_level"]], filling_level_id=a["filling_level"],
            filling_level_pct=LEVEL_PCT[a["filling_level"]],
            is_granular=a["filling_type"] in GRANULAR_TYPES,
            view=VIEW.get(a["view"], str(a["view"])),
            occlusion=bool(a["occlusion"]),
            bbox=[int(v) for v in a["bbox"]],
        )

    def container_mask(self, i):
        """Binary container mask (uint8 0/255) from the soft Mask R-CNN map."""
        m = cv2.imread(self.mask_path(i), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        bw = (m >= MASK_THRESH).astype(np.uint8) * 255
        # keep the largest connected component (drop speckle from soft edges)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
        if n > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            bw = (lab == big).astype(np.uint8) * 255
        return bw

    def granular(self):
        return [i for i in self.available if self.ann[i]["filling_type"] in GRANULAR_TYPES]


def make_splits(ccm, seed=0):
    """Container-disjoint split (evaluate on UNSEEN containers, the meaningful
    CORSMAL-style protocol). Only uses locally-available frames."""
    # cups: 1,2,3,7 ; glasses: 4,5,6,8 -> spread across splits
    train_c = {1, 2, 4, 5}
    val_c = {3, 6}
    test_c = {7, 8}
    out = {"train": [], "val": [], "test": []}
    for i in ccm.available:
        c = ccm.ann[i]["containerID"]
        if c in train_c: out["train"].append(i)
        elif c in val_c: out["val"].append(i)
        elif c in test_c: out["test"].append(i)
    out["_meta"] = dict(protocol="container-disjoint",
                        train_containers=sorted(train_c),
                        val_containers=sorted(val_c),
                        test_containers=sorted(test_c))
    return out


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/ccm"
    ds = CCM(root)
    print(f"available RGB frames: {len(ds.available)}")
    print(f"granular (rice/pasta): {len(ds.granular())}")
    print("sample record:", json.dumps(ds.record(ds.available[0]), indent=1))
