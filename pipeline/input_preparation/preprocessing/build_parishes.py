"""Build parish boundaries using the shared CAOP geometry helpers."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipeline.utils.geometry import (  # noqa: E402
    dump_plot_ready_geojson_if_missing,
    load_mainland_and_islands,
)
from pipeline.utils.paths import load_paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Override pipeline/config/external_paths.json")
    parser.add_argument("--map-dir", type=Path, help="Override configured map_dir")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--internal-paths-config", type=Path, help="Override internal_paths.json")
    args = parser.parse_args()
    paths = load_paths(args.config, internal_config_path=args.internal_paths_config)
    args.out = args.out if args.out is not None else paths.internal("parish_geometry")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}. Choose a new path.")

    map_dir = args.map_dir if args.map_dir is not None else paths.external("caop_shapefiles")
    mainland, islands = load_mainland_and_islands(
        map_dir, "dicofre", "Cont_AAD_CAOP2016.shp"
    )
    all_ids = []
    for name, frame in [("Mainland", mainland), *islands]:
        if frame.crs is None:
            raise ValueError(f"Missing CRS: {name}")
        if not frame["dicofre"].str.fullmatch(r"[0-9A-Z]{6}").all():
            raise ValueError(f"Invalid parish IDs: {name}")
        # The raster aggregation pipeline expects this exact field name.
        frame["ID"] = frame["dicofre"]
        all_ids.extend(frame["ID"])
        print(f"{name}: {len(frame):,} parishes")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate parish IDs across regions")

    dump_plot_ready_geojson_if_missing(mainland, islands, args.out)


if __name__ == "__main__":
    main()
