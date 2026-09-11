"""
Downloads (or refreshes) the provincial DEM cached at ./data/cache/argendem_neuquen.tif.

Same code path as the sidebar "Actualizar DEM provincial" button, but can
optionally block and poll a Drive export until it finishes:

    python update_dem.py                    # start download/export, don't wait
    python update_dem.py --wait             # poll the Drive export until COMPLETED
    python update_dem.py --force --source SRTM
"""

import argparse
import logging
import time

from modules.ee_client import (
    EarthEngineError,
    check_provincial_dem_export,
    describe_dem,
    update_provincial_dem,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["AW3D30", "SRTM"], default="AW3D30")
    parser.add_argument("--scale", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="Re-download even if a cached DEM exists.")
    parser.add_argument("--wait", action="store_true", help="Poll a Drive export until it completes.")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.monotonic()

    try:
        result = update_provincial_dem(source=args.source, scale=args.scale, force=args.force)
        print(f"update_provincial_dem -> {result}")

        while args.wait and result["state"] not in ("CACHED", "COMPLETED"):
            time.sleep(args.poll_seconds)
            result = check_provincial_dem_export()
            elapsed = time.monotonic() - started
            print(f"[{elapsed:7.0f}s] export state: {result['state']}")
    except EarthEngineError as ee_error:
        print(f"ERROR: {ee_error}")
        return 1

    if result["state"] in ("CACHED", "COMPLETED"):
        print(f"Total time: {time.monotonic() - started:.0f}s")
        for key, value in describe_dem().items():
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
