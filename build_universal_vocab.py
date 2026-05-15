"""
build_universal_vocab.py — Universal Gene Vocabulary Builder
=============================================================
Builds a shared gene vocabulary from protein-coding human genes, intersected
with the union of gene symbols observed across all datasets used in CellJEPA
experiments (kidney, PBMC-68K, PBMC-3K).

Filtering to protein-coding genes (~19-20k) rather than taking the raw union
of all dataset gene symbols (~73k) avoids inflating the vocabulary with
pseudogenes, non-coding RNAs, and annotation artifacts from different genome
references. This keeps the embedding table to ~10M params (vs ~37M for 73k
genes) and allows pre-training on more cells within the same RAM budget.

The protein-coding gene list is fetched from HGNC (Human Genome Nomenclature
Committee), the authoritative source for human gene symbols.

Token ID convention (same as original CellJEPA):
    0 = <cls>
    1 = <pad>
    2..N+1 = gene symbols in alphabetical order

Output:
    universal_gene_names.json  — ordered list of N gene symbols
    (vocab_size = N + 2)

Usage:
    # Full run (fetches HGNC list + collects dataset genes):
    python build_universal_vocab.py \\
        --tar_path /path/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz \\
        --drive_dir /content/drive/MyDrive/CellJEPA_results/universal_vocab/

    # Smoke test (uses scanpy built-ins, no Census/tarball needed):
    python build_universal_vocab.py --smoke_test

    # Skip sources you don't have:
    python build_universal_vocab.py --skip_kidney --skip_pbmc68k
"""

from __future__ import annotations

import argparse
import io
import json
import os

import numpy as np
import scanpy as sc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build universal gene vocabulary")
    p.add_argument("--drive_dir",
                   default="/content/drive/MyDrive/CellJEPA_results/universal_vocab/",
                   help="Output directory for universal_gene_names.json")
    p.add_argument("--tar_path", default=None,
                   help="Path to fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz "
                        "(optional; includes PBMC-68K genes in vocab)")
    p.add_argument("--cache_dir", default="/content",
                   help="Directory to extract PBMC-68K tarball")
    p.add_argument("--smoke_test", action="store_true",
                   help="Use built-in scanpy datasets only (no Census/tarball)")
    p.add_argument("--skip_kidney",  action="store_true")
    p.add_argument("--skip_pbmc68k", action="store_true")
    p.add_argument("--skip_pbmc3k",  action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Protein-coding gene filter (HGNC)
# ---------------------------------------------------------------------------

HGNC_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/"
    "hgnc_complete_set.txt"
)

def get_protein_coding_genes() -> set[str]:
    """
    Fetch the HGNC complete gene set and return all approved protein-coding
    gene symbols. Falls back to None if the download fails (caller should
    then skip filtering).
    """
    import urllib.request

    print("Fetching protein-coding gene list from HGNC …")
    try:
        with urllib.request.urlopen(HGNC_URL, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  WARNING: could not fetch HGNC list ({e}) — skipping protein-coding filter")
        return None

    import csv
    coding_genes: set[str] = set()
    reader = csv.DictReader(io.StringIO(content), delimiter="\t")
    for row in reader:
        if row.get("locus_type") == "gene with protein product" and row.get("symbol"):
            coding_genes.add(row["symbol"])

    print(f"  {len(coding_genes):,} protein-coding genes from HGNC")
    return coding_genes


# ---------------------------------------------------------------------------
# Gene set collectors
# ---------------------------------------------------------------------------

def get_pbmc3k_genes() -> set[str]:
    print("Collecting PBMC-3K genes …")
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    genes = set(adata.var_names.tolist())
    print(f"  {len(genes):,} genes")
    return genes


def get_pbmc68k_genes_smoke() -> set[str]:
    print("Collecting PBMC-68K genes (reduced built-in) …")
    adata = sc.datasets.pbmc68k_reduced()
    if adata.raw is not None:
        adata = adata.raw.to_adata()
    adata.var_names_make_unique()
    genes = set(adata.var_names.tolist())
    print(f"  {len(genes):,} genes (reduced dataset)")
    return genes


def get_pbmc68k_genes_full(tar_path: str, cache_dir: str) -> set[str]:
    import tarfile

    print("Collecting PBMC-68K genes (full tarball) …")
    out_dir    = os.path.join(cache_dir, "pbmc68k")
    matrix_dir = os.path.join(out_dir, "filtered_matrices_mex", "hg19")

    if not os.path.exists(matrix_dir):
        print(f"  Extracting {tar_path} …")
        with tarfile.open(tar_path) as tar:
            tar.extractall(out_dir)

    adata = sc.read_10x_mtx(matrix_dir, var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()
    genes = set(adata.var_names.tolist())
    print(f"  {len(genes):,} genes")
    return genes


def get_kidney_genes_smoke() -> set[str]:
    print("Collecting kidney genes (PBMC-68K reduced as proxy for smoke test) …")
    return get_pbmc68k_genes_smoke()


def get_kidney_genes_full() -> set[str]:
    print("Collecting kidney genes from CELLxGENE Census …")
    try:
        import cellxgene_census
    except ImportError:
        raise ImportError("pip install cellxgene-census")

    census = cellxgene_census.open_soma()
    obs_df = census["census_data"]["homo_sapiens"]["obs"].read(
        value_filter="tissue_general == 'kidney' and is_primary_data == True",
        column_names=["soma_joinid"],
    ).concat().to_pandas()
    census.close()

    sampled = obs_df["soma_joinid"].sample(n=min(500, len(obs_df)), random_state=42).tolist()

    census = cellxgene_census.open_soma()
    adata = cellxgene_census.get_anndata(
        census=census, organism="Homo sapiens", obs_coords=sampled,
    )
    census.close()

    if "feature_name" in adata.var.columns:
        genes = set(adata.var["feature_name"].astype(str).tolist())
        print(f"  {len(genes):,} genes (gene symbols from feature_name)")
    else:
        genes = set(adata.var_names.tolist())
        print(f"  {len(genes):,} genes (var_names — may be Ensembl IDs)")
    return genes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.drive_dir, exist_ok=True)

    all_genes: set[str] = set()

    if not args.skip_pbmc3k:
        all_genes |= get_pbmc3k_genes()

    if not args.skip_pbmc68k:
        if args.smoke_test or args.tar_path is None:
            all_genes |= get_pbmc68k_genes_smoke()
        else:
            all_genes |= get_pbmc68k_genes_full(args.tar_path, args.cache_dir)

    if not args.skip_kidney:
        if args.smoke_test:
            all_genes |= get_kidney_genes_smoke()
        else:
            all_genes |= get_kidney_genes_full()

    # Remove empty strings and obvious non-gene entries
    all_genes = {g for g in all_genes if g and not g.startswith("__")}

    # Filter to protein-coding genes only
    if not args.smoke_test:
        coding_genes = get_protein_coding_genes()
        if coding_genes is not None:
            before = len(all_genes)
            all_genes = all_genes & coding_genes
            print(f"  Protein-coding filter: {before:,} → {len(all_genes):,} genes")
        else:
            print("  Skipping protein-coding filter (HGNC download failed)")
    else:
        print("  [smoke test] skipping protein-coding filter")

    # Sort alphabetically for reproducibility
    gene_list = sorted(all_genes)
    vocab_size = len(gene_list) + 2  # 0=<cls>, 1=<pad>

    out_path = os.path.join(args.drive_dir, "universal_gene_names.json")
    with open(out_path, "w") as f:
        json.dump(gene_list, f)

    print(f"\n{'='*60}")
    print(f"  Universal vocabulary built")
    print(f"  Total genes: {len(gene_list):,}")
    print(f"  Vocab size (incl. <cls>/<pad>): {vocab_size:,}")
    print(f"  Saved to: {out_path}")
    print(f"{'='*60}")
    print(f"\nNext step: pretrain_universal.py with --vocab_file {out_path}")


if __name__ == "__main__":
    main()
