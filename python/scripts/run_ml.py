#!/usr/bin/env python
"""Run reusable ML workflows on parameter HDF5 datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.io.hdf5 import read_dataset  # noqa: E402
from analysis.ml import (  # noqa: E402
    feature_matrix,
    ledoit_wolf_covariance,
    run_gmm,
    run_hdbscan,
    run_pca,
    run_umap,
)
from analysis.parallel import add_parallel_arguments, config_from_args  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UMAP/HDBSCAN/GMM/PCA/LedoitWolf parameter analysis.")
    parser.add_argument("h5_file", help="Parameter HDF5 file.")
    parser.add_argument("--features", nargs="+", required=True, help="Dataset names used as feature columns.")
    parser.add_argument("--method", choices=("pca", "umap", "hdbscan", "gmm", "ledoitwolf"), default="umap")
    parser.add_argument("--output", help="Output HDF5 path. Defaults to <input>_<method>.h5.")
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--gmm-components", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--no-standardize", action="store_true")
    add_parallel_arguments(parser, include_chunk_size=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    src = Path(args.h5_file)
    out = Path(args.output) if args.output else src.with_name(f"{src.stem}_{args.method}.h5")
    columns = [read_dataset(src, key) for key in args.features]
    x = feature_matrix(columns)
    finite_mask = np.all(np.isfinite(x), axis=1)
    x_fit = x[finite_mask]
    cfg = config_from_args(args)

    with h5py.File(out, "w") as handle:
        handle.attrs["source_file"] = str(src.resolve())
        handle.attrs["features"] = ",".join(args.features)
        handle.attrs["method"] = args.method
        handle.create_dataset("finite_mask", data=finite_mask.astype(np.uint8))
        if args.method == "pca":
            embedding = run_pca(x_fit, n_components=args.n_components, standardize=not args.no_standardize)
            handle.create_dataset("embedding", data=embedding)
        elif args.method == "umap":
            embedding = run_umap(
                x_fit,
                n_components=args.n_components,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
                random_state=args.random_state,
                standardize=not args.no_standardize,
                config=cfg,
            )
            handle.create_dataset("embedding", data=embedding)
        elif args.method == "hdbscan":
            labels = run_hdbscan(x_fit, min_cluster_size=args.min_cluster_size)
            handle.create_dataset("labels", data=labels.astype(np.int32))
        elif args.method == "gmm":
            labels = run_gmm(x_fit, n_components=args.gmm_components)
            handle.create_dataset("labels", data=labels.astype(np.int32))
        else:
            covariance, shrinkage = ledoit_wolf_covariance(x_fit)
            handle.create_dataset("covariance", data=covariance)
            handle.attrs["shrinkage"] = shrinkage
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
