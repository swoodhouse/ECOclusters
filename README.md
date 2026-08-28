# ECOclusters

This repository contains the code used to construct **ECOclusters**, as described in the [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.03.26.583354v3) "Immune signatures of common exposures through co-occurrence of T-cell receptors in tens of thousands of donors".

An ECOcluster is a group of T-cell receptors (TCRs) that tend to co-occur in the same people, mined from more than 30,000 TCR repertoires. Each ECOcluster is a putative signature of humanity's collective, shared T-cell response to a shared exposure — a virus, bacterium, or other common antigen. Each ECOcluster comprises one or more HLA-COclusters: clusters associated with a single HLA allele, clustered by occurrence in the repertoires of donors inferred to have the allele.

## What's in this repo

This repo provides the algorithm implementations used to build ECOclusters from TCR repertoire data:

- [`hlacoclusters.py`](hlacoclusters.py) — construction of HLA-COclusters, the building blocks that ECOclusters are assembled from. Biclustering, dimensionality reduction and HDBSCAN clustering of the co-occurrence matrix of TCRs associated with an HLA allele across the repertoires of donors inferred to carry the allele.
- [`ecoclusters.py`](ecoclusters.py) — construction of ECOclusters by clustering of HLA-COclusters: HLA-COcluster co-occurrence and correlation matrices, hierarchical clustering of HLA-COclusters into a clustering tree, and the elbow-cutting procedure used to call ECOclusters from that tree.
- [`ecoclusters_extras.py`](ecoclusters_extras.py) — supporting utilities for the pipeline above, including masked correlation, HLA-COcluster masking, and cluster reporting.
- [`exact_linkage.py`](exact_linkage.py) / [`exact_linkage_cython.pyx`](exact_linkage_cython.pyx) — exact nearest-neighbor-chain hierarchical linkage used in ECOcluster construction, implemented in Cython for speed and built on first import via `pyximport`.

Together, these implement the algorithms used to construct ECOclusters as described in the manuscript.

## Data and code availability

This code was developed and is run inside Adaptive Biotechnologies' internal research computing environment, and depends on internal libraries, infrastructure (e.g., Apache Spark), and proprietary repertoire datasets that are not publicly available. As a result, this repo cannot be run standalone; it is provided to enable understanding and reimplementation of the ECOclusters algorithms.

## Questions

For questions about the methods or manuscript, please contact the corresponding author(s) listed in the [preprint](https://www.biorxiv.org/content/10.1101/2024.03.26.583354v3).
