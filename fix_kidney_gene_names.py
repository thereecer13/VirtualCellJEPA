"""
fix_kidney_gene_names.py
========================
Re-downloads a tiny slice of kidney data from CELLxGENE Census to recover
the correct gene symbol names (feature_name), then overwrites the gene names
JSON files saved by pretrain_kidney.py / pretrain_kidney_sigreg.py.

Run this ONCE after the kidney pretraining checkpoints already exist — no
retraining needed, only the JSON is updated.

Usage (Colab):
    python3 fix_kidney_gene_names.py \
        --jepa_genes   /content/drive/.../kidney_pretrain/kidney_gene_names.json \
        --sigreg_genes /content/drive/.../kidney_sigreg/kidney_sigreg_gene_names.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import scanpy as sc


def fetch_symbol_names(n_cells: int = 500) -> list[str]:
    """
    Download n_cells kidney cells from Census, run HVG selection, and
    return the 2000 HVG gene symbols (feature_name).
    """
    try:
        import cellxgene_census
    except ImportError:
        raise ImportError("pip install cellxgene-census")

    print(f"Fetching {n_cells} kidney cells from Census to recover gene symbols...")
    census = cellxgene_census.open_soma()
    obs_df = census["census_data"]["homo_sapiens"]["obs"].read(
        value_filter="tissue_general == 'kidney' and is_primary_data == True",
        column_names=["soma_joinid"],
    ).concat().to_pandas()
    census.close()

    sampled_ids = obs_df["soma_joinid"].sample(n=min(n_cells, len(obs_df)), random_state=42).tolist()

    census = cellxgene_census.open_soma()
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_coords=sampled_ids,
    )
    census.close()
    print(f"  Downloaded {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  var columns: {list(adata.var.columns)}")
    print(f"  Sample var_names (Ensembl): {list(adata.var_names[:3])}")

    import scipy.sparse as sp
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    adata = adata[:, adata.var.highly_variable].copy()

    if "feature_name" in adata.var.columns:
        gene_names = list(adata.var["feature_name"].astype(str))
        print(f"  Recovered {len(gene_names)} gene symbols via feature_name")
        print(f"  Sample symbols: {gene_names[:5]}")
    else:
        gene_names = list(adata.var_names)
        print("  WARNING: feature_name not found, using var_names (may still be Ensembl IDs)")

    return gene_names


def main():
    p = argparse.ArgumentParser(description="Fix kidney gene name JSONs to use gene symbols")
    p.add_argument("--jepa_genes", default=None,
                   help="Path to kidney_gene_names.json to overwrite")
    p.add_argument("--sigreg_genes", default=None,
                   help="Path to kidney_sigreg_gene_names.json to overwrite")
    p.add_argument("--n_cells", type=int, default=500,
                   help="Cells to fetch for HVG selection (default 500 — fast)")
    args = p.parse_args()

    if not args.jepa_genes and not args.sigreg_genes:
        print("Provide at least one of --jepa_genes or --sigreg_genes")
        return

    gene_names = fetch_symbol_names(args.n_cells)

    for path in [args.jepa_genes, args.sigreg_genes]:
        if path:
            with open(path, "w") as f:
                json.dump(gene_names, f)
            print(f"  Wrote {len(gene_names)} gene symbols to {path}")

    print("\nDone. Re-run run_transfer.py — gene alignment should now find matches.")


if __name__ == "__main__":
    main()
