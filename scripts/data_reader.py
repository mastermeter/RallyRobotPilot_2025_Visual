

if __name__ == "__main__":
    import numpy as np
    filePath = "record_signs_5.npz"

    d = np.load(filePath, allow_pickle=False)
    print(f"Keys in {filePath}: {list(d.keys())}")
    print(f"Raycast distances shape: {d['rays']}")