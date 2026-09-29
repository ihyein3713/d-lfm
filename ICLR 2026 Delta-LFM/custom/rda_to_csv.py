#!/usr/bin/env python
"""Convert the ADNIMERGE2 R package datasets (.rda) to CSV.

Readers are tried in order: pyreadr (librdata), then rdata (pure python). The two cover
different corners of the R serialization format, and a few ADNIMERGE2 tables use features
neither supports, so the run ends with a per-file report instead of failing on the first error.

Usage:
    python custom/rda_to_csv.py                          # default dirs, see below
    python custom/rda_to_csv.py --src DIR --out DIR
    python custom/rda_to_csv.py --tarball PKG.tar.gz     # extract every .rda first
    python custom/rda_to_csv.py --only ADSL DXSUM        # a subset, by dataset name
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

DEFAULT_SRC = Path("/mnt/aix22308/data/ADNI_delta_lfm/meta/ADNIMERGE2/data")
DEFAULT_OUT = Path("/mnt/aix22308/data/ADNI_delta_lfm/meta/ADNIMERGE2/csv")


def read_with_pyreadr(path: Path):
    import pyreadr
    result = pyreadr.read_r(str(path))
    return {k: v for k, v in result.items()}


def read_with_rdata(path: Path):
    import warnings
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # R classes without a python constructor
        return rdata.read_rda(str(path))


READERS = (("pyreadr", read_with_pyreadr), ("rdata", read_with_rdata))


def extract_tarball(tarball: Path, dest: Path) -> int:
    """Extract every data/*.rda of an R package tarball into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tarball) as tf:
        for member in tf.getmembers():
            if member.isfile() and member.name.endswith(".rda") and "/data/" in member.name:
                member.name = Path(member.name).name       # flatten
                tf.extract(member, dest)
                n += 1
    return n


def convert(path: Path, out_dir: Path) -> tuple[str, str]:
    """-> (status, detail). status is 'ok', 'skipped' or 'failed'."""
    errors = []
    for name, reader in READERS:
        try:
            tables = reader(path)
        except Exception as e:                              # noqa: BLE001 - report, do not abort
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:90]}")
            continue
        written = []
        for key, df in tables.items():
            if not hasattr(df, "to_csv"):                   # not a data.frame (lists, functions)
                errors.append(f"{name}: {key} is {type(df).__name__}, not a table")
                continue
            # column names arrive as numpy str_ with some readers; plain str keeps the header clean
            df.columns = [str(c) for c in df.columns]
            dst = out_dir / f"{key if key else path.stem}.csv"
            df.to_csv(dst, index=False)
            written.append(f"{dst.name} ({df.shape[0]}x{df.shape[1]})")
        if written:
            return "ok", f"{name} -> " + ", ".join(written)
    return "failed", " | ".join(errors) if errors else "no table found"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="directory holding the .rda files")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for the CSV files")
    ap.add_argument("--tarball", type=Path, default=None, help="R package tarball to extract .rda files from first")
    ap.add_argument("--only", nargs="*", default=None, help="dataset names to convert (without .rda)")
    ap.add_argument("--overwrite", action="store_true", help="rewrite CSVs that already exist")
    args = ap.parse_args()

    if args.tarball:
        n = extract_tarball(args.tarball, args.src)
        print(f"extracted {n} .rda files into {args.src}")

    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("*.rda"))
    if args.only:
        wanted = {n.removesuffix(".rda") for n in args.only}
        files = [f for f in files if f.stem in wanted]
    if not files:
        sys.exit(f"no .rda files in {args.src}")

    ok, failed, skipped = [], [], []
    for f in files:
        if not args.overwrite and (args.out / f"{f.stem}.csv").exists():
            skipped.append(f.stem)
            print(f"{f.stem:34s} SKIP (csv exists)")
            continue
        status, detail = convert(f, args.out)
        print(f"{f.stem:34s} {status.upper():7s} {detail}")
        (ok if status == "ok" else failed).append(f.stem)

    print(f"\nconverted {len(ok)}, failed {len(failed)}, skipped {len(skipped)} -> {args.out}")
    if failed:
        print("failed:", ", ".join(failed))
        print("these use R serialization features neither reader supports; converting them needs R:")
        print("  Rscript -e 'load(\"X.rda\"); write.csv(get(ls()[1]), \"X.csv\", row.names=FALSE)'")


if __name__ == "__main__":
    main()
