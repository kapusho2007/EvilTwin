"""Stage 1 driver: raw CSVs -> unified frame tables under data_unified/<dataset>/.
Usage:  python run_ingest.py awid3            (whole dataset)
        python run_ingest.py rids 3 10        (piece 3 of 10; pieces are independent and restartable)
Needs only pandas."""
import sys
import time

from pipeline import ingestion

if __name__ == "__main__":
    ds = sys.argv[1]
    k, n = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    t0 = time.time()
    parts = ingestion.ingest(ds, k, n)
    print(f"[{ds}] piece {k}/{n}: {len(parts)} parts in {time.time() - t0:.0f}s", flush=True)
