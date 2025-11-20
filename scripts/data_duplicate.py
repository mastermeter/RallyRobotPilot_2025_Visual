import os
import glob
import numpy as np

def flip_one_npz(in_path: str, out_path: str):
    with np.load(in_path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}

        if 'images' in d:
            imgs = d['images']
            if imgs.size > 0:
                # axe W inversé
                out['images'] = imgs[:, :, ::-1, :]
                print(f"  images: {imgs.shape} -> flipped")

        if 'controls' in d:
            ctr = d['controls']
            if ctr.ndim == 2 and ctr.shape[1] == 4:
                ctr = ctr.copy()
                ctr[:, [2, 3]] = ctr[:, [3, 2]]
                out['controls'] = ctr
                print("  controls: swapped LEFT<->RIGHT")

        if 'rays' in d:
            rays = d['rays']
            if rays.ndim == 2 and rays.shape[1] >= 1:
                out['rays'] = rays[:, ::-1]
                print(f"  rays: reversed per-sample ({rays.shape})")

    np.savez_compressed(out_path, **out)

def main():
    pattern = "record_*.npz" #record_signs_*.npz
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Aucun fichier trouvé pour: {pattern}")
        return

    print(f"Trouvé {len(files)} fichier(s).")
    for fp in files:
        base, ext = os.path.splitext(fp)
        out_fp = f"{base}_fliped{ext}" 
        if os.path.exists(out_fp):
            print(f"[skip] {out_fp} existe déjà.")
            continue
        try:
            print(f"\nProcessing: {fp}")
            flip_one_npz(fp, out_fp)
            print(f"[OK] Écrit: {out_fp}")
        except Exception as e:
            print(f"[ERREUR] {fp}: {e}")

if __name__ == "__main__":
    main()
