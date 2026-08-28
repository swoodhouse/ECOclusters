# ECOclusters

This repository contains the code used to construct **ECOclusters**, as described in the [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.03.26.583354v3) "Immune signatures of common exposures through co-occurrence of T-cell receptors in tens of thousands of donors".

An ECOcluster is a group of T-cell receptors (TCRs) that tend to co-occur in the same people, mined from more than 30,000 TCR repertoires. Each ECOcluster is a putative signature of humanity's collective, shared T-cell response to a shared exposure — a virus, bacterium, or other common antigen. Each ECOcluster comprises one or more HLA-COclusters: clusters associated with a single HLA allele, clustered by occurrence in the repertoires of donors inferred to have the allele.

## What's in this repo

This repo provides the algorithm implementations used to build ECOclusters from TCR repertoire data:

- [`taccs.py`](taccs.py) — construction of TCR Antigen-specific Correlated Clusters (TACCs), the building blocks that ECOclusters are assembled from, via biclustering and dimensionality reduction (SVD, UMAP, HDBSCAN) of TCR co-occurrence across repertoires.
- [`ecoclusters.py`](ecoclusters.py) — the core ECOcluster construction pipeline: TACC co-occurrence and correlation matrices, hierarchical clustering of TACCs into a clustering tree, and the elbow-cutting procedure used to call ECOclusters from that tree.
- [`ecoclusters_extras.py`](ecoclusters_extras.py) — supporting utilities for the pipeline above, including masked correlation, TACC masking, and cluster reporting.
- [`exact_linkage.py`](exact_linkage.py) / [`exact_linkage_cython.pyx`](exact_linkage_cython.pyx) — exact nearest-neighbor-chain hierarchical linkage used in ECOcluster construction, implemented in Cython for speed and built on first import via `pyximport`.

Together, these implement all of the algorithms used to construct ECOclusters as described in the manuscript.

## Data and code availability

This code was developed and is run inside Adaptive Biotechnologies' internal research computing environment, and depends on internal libraries, infrastructure (e.g., Apache Spark), and proprietary repertoire datasets that are not included here and cannot be released. As a result, this repo cannot be run standalone — it is provided so that the algorithms underlying ECOclusters can be easily inspected and reimplemented.

## Questions

For questions about the methods or manuscript, please contact the corresponding author(s) listed in the [preprint](https://www.biorxiv.org/content/10.1101/2024.03.26.583354v3).
