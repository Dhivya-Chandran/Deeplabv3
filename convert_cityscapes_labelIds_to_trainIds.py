import cv2
import numpy as np
from pathlib import Path

# Cityscapes labelId -> trainId mapping (19-class)
ID2TRAIN = {
    -1: 255,
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,    8: 1,    9: 255, 10: 255,
    11: 2,   12: 3,   13: 4,  14: 255, 15: 255, 16: 255,
    17: 5,   18: 255, 19: 6,  20: 7,  21: 8,  22: 9,  23: 10,
    24: 11,  25: 12,  26: 13, 27: 14, 28: 15, 29: 255,
    30: 255, 31: 16,  32: 17, 33: 18
}

def convert_one(src: Path, dst: Path):
    mask = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if mask is None:
        print("Failed to read:", src)
        return
    out = np.full(mask.shape, 255, dtype=np.uint8)
    for k, v in ID2TRAIN.items():
        out[mask == k] = v
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out)

def main():
    src_root = Path(r".\datasets\Cityscapes\gtFine")
    dst_root = Path(r".\datasets\Cityscapes\gtFine_trainIds")

    files = list(src_root.rglob("*_gtFine_labelIds.png"))
    if not files:
        print("No *_gtFine_labelIds.png found.")
        return

    for f in files:
        rel = f.relative_to(src_root)
        out = dst_root / rel
        out = Path(str(out).replace("_labelIds.png", "_trainIds.png"))
        convert_one(f, out)

    print("Done. Converted masks saved to:", dst_root)

if __name__ == "__main__":
    main()
