import os
# One BLAS/OpenMP thread per process; parallelism comes from --num_workers
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import argparse
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm


def is_broken(path):
    # Reading every array forces a full decompress, which catches truncated or corrupted files
    try:
        with np.load(path) as data:
            for key in data.files:
                data[key]
        return False
    except Exception:
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find and delete broken .npz files')
    parser.add_argument('--save_path', default='./cache', type=str, help='path to processed data')
    parser.add_argument('--num_workers', type=int, help='number of parallel processes', default=len(os.sched_getaffinity(0)))
    parser.add_argument('--dry_run', action='store_true', help='only list broken files without deleting them')
    args = parser.parse_args()

    file_names = os.listdir(args.save_path)
    npz_files = [os.path.join(args.save_path, f) for f in file_names if f.endswith('.npz')]
    # Leftover temp files from interrupted writes (see DataProcessor.save_to_disk)
    tmp_files = [os.path.join(args.save_path, f) for f in file_names if '.npz.tmp' in f]
    print(f"Checking {len(npz_files)} .npz files in {args.save_path}")

    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(is_broken, npz_files, chunksize=16), total=len(npz_files)))
    broken_files = [f for f, broken in zip(npz_files, results) if broken]

    for f in broken_files + tmp_files:
        print(f)
    print(f"Broken .npz files: {len(broken_files)}, leftover temp files: {len(tmp_files)}")

    if args.dry_run:
        print("Dry run: nothing deleted")
    else:
        for f in broken_files + tmp_files:
            os.remove(f)
        print(f"Deleted {len(broken_files) + len(tmp_files)} files")
