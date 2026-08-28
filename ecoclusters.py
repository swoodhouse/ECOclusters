from typing import List, Tuple, Set, Callable, Any, Optional, Dict
import logging
import pandas as pd
import numpy as np
import numpy.typing as npt
from hdbscan import HDBSCAN
import umap
from collections import defaultdict

from scipy.sparse import hstack, coo_matrix, csr_matrix, identity, triu
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import (
    LongType,
    StringType,
    StructField,
    StructType,
    ArrayType,
    FloatType,
    DoubleType,
)


from immunopipeline.ecocluster._ecocluster import (
    EcoclusterConstructor,  ##MUSTHAVE
    sparse_map_nonzero, ##MUSTHAVE
    masked_correlation, ##MUSTHAVE
    build_hlacocluster_mask # can't import from contrib  ##MUSTHAVE
)

from scipy.stats import percentileofscore

from immunopipeline.ecocluster.exact_linkage import exact_linkage  ##MUSTHAVE

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import itertools


DEFAULT_MIN_MEMBERS: int = 1
# this is just a meaninglessly high number. It's a count of HLA-COclusters, not TCRs.
DEFAULT_MAX_MEMBERS: int = 20000

logger = logging.getLogger(__name__)

V5_HLA_ALLELES_WITH_HLACOCLUSTERS = [
    x for x in V5_HLA_ALLELES if
    x not in ["B*35:02", "B*39:10", "B*44:05", "B*58:02", "DPA1*01:03+DPB1*01:01", "DPA1*02:01+DPB1*03:01", "DPA1*02:01+DPB1*11:01", "DQA1*01:03+DQB1*06:09", "DRB1*16:01", "DRB1*16:02"]
]

class HlaCoclusterNames(ISparkDatasetJob):
    description = """
    Gives the ordered list of HLA-COclusters, for indexing the correlation matrix.
    """
    author = DatasetEmail("swoodhouse@adaptivebiotech.com")

    schema = StructType(
        [
            StructField(name="order_index", dataType=LongType(), nullable=False),
            StructField(name="hlacocluster", dataType=StringType(), nullable=False),
        ]
    )

    def __init__(self, version: str) -> None:
        self.version = version

        if version == "V8":
            self.ALLELES = V8_HLA_ALLELES_WITH_HLACOCLUSTERS
        elif version == "V5":
            self.ALLELES = V5_HLA_ALLELES_WITH_HLACOCLUSTERS

        super().__init__()

    def inputs(self) -> DatasetInputRequirements:
        if self.version == "V8":
            return DatasetInputRequirements(
                tuple(
                    [
                        InputRequirement(
                            name=f"{clean_allele(allele)}_hlacoclusters",
                            dataset=f"taccs.hladb-v8-{clean_allele(allele)}-fdr-{1e-3}.parquet",
                            path="/parquet/*/",
                            strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                        )
                        for allele in V8_HLA_ALLELES_WITH_HLACOCLUSTERS
                    ]
                )
            )
        elif self.version == "V5":
            return DatasetInputRequirements(
                tuple(
                    [
                        InputRequirement(
                            name=f"{clean_allele(allele)}_hlacoclusters",
                            dataset=f"taccs.hladb-v5-{clean_allele(allele)}.parquet",
                            path="/parquet/*/",
                            strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                        )
                        for allele in V5_HLA_ALLELES_WITH_HLACOCLUSTERS
                    ]
                )
            )
        else:
            raise ValueError("Version not implemented")

    def output(self) -> str:
        if self.version == "V8":
            return f"ecoclusters.tacc_names.hladb-v8-fdr-{1e-3}.parquet"
        elif self.version == "V5":
            return "ecoclusters.tacc_names.hladb-v5.parquet"
        else:
            raise ValueError("Version not implemented")

    def transform(self, request: SparkDatasetJobRequest) -> DataFrame:
        spark = df2spark(request.get(f"{clean_allele(self.ALLELES[0])}_hlacoclusters"))

        hlacoclusters = []
        for allele in self.ALLELES:
            hla_hlacoclusters = (
                request.get(f"{clean_allele(allele)}_hlacoclusters")
                .select('hlacocluster')
                .drop_duplicates()
                .toPandas()['hlacocluster'].tolist()
            )
            hlacoclusters = hlacoclusters + hla_hlacoclusters

        hlacoclusters.sort()

        return spark.createDataFrame(
            schema=self.schema,
            data=pd.DataFrame.from_dict(  # type: ignore
                {"order_index": [i for i in range(len(hlacoclusters))], "hlacocluster": hlacoclusters}
            )
        )


class CountHlaCoclusters(ISparkDatasetJob):
    description = """
    Sparse samples x HLA-COcluster count matrix.
    """
    author = DatasetEmail("swoodhouse@adaptivebiotech.com")

    schema = StructType(
        [
            StructField(name="data", dataType=LongType(), nullable=False),
            StructField(name="coords0", dataType=LongType(), nullable=False),
            StructField(name="coords1", dataType=LongType(), nullable=False),
        ]
    )

    def __init__(self, version: str) -> None:
        self.version = version

        if version == "V8":
            self.ALLELES = V8_HLA_ALLELES_WITH_HLACOCLUSTERS
        elif version == "V5":
            self.ALLELES = V5_HLA_ALLELES_WITH_HLACOCLUSTERS
        super().__init__()

    def inputs(self) -> DatasetInputRequirements:
        if self.version == "V8":
            return DatasetInputRequirements(
                tuple(
                    [
                        InputRequirement(
                            name="hlas",
                            dataset="repertoire.samples.taccs-hladb-v8.hlas.parquet",
                            path="/parquet/*/",
                            strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                        )
                    ] +
                    [
                        InputRequirement(
                            name=f"{clean_allele(allele)}_hlacoclusters",
                            dataset=f"taccs.hladb-v8-{clean_allele(allele)}-fdr-{1e-3}.parquet",
                            path="/parquet/*/",
                            strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                        )
                        for allele in V8_HLA_ALLELES_WITH_HLACOCLUSTERS
                    ] +
                    [
                        InputRequirement(
                            name=f"{clean_allele(allele)}_filteredseqs",
                            dataset=f"repertoire.sequences.hladb-v8-filteredseqs-{clean_allele(allele)}-fdr-{1e-3}.parquet",
                            path="/parquet/*/",
                            strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                        )
                        for allele in V8_HLA_ALLELES_WITH_HLACOCLUSTERS
                    ]
                )
            )
        elif self.version == "V5":
            return DatasetInputRequirements(
                tuple(
                    [
                        InputRequirement(
                            name="hlas",
                            dataset="repertoire.samples.taccs-hladb-v5.hlas.parquet",
                            path="/parquet/*/",
                            strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                        )
                    ] +
                    [
                        InputRequirement(
                            name=f"{clean_allele(allele)}_hlacoclusters",
                            dataset=f"taccs.hladb-v5-{clean_allele(allele)}.parquet",
                            path="/parquet/*/",
                            strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                        )
                        for allele in V5_HLA_ALLELES_WITH_HLACOCLUSTERS
                    ] +
                    [
                        InputRequirement(
                            name=f"{clean_allele(allele)}_filteredseqs",
                            dataset=f"repertoire.sequences.hladb-v5-filteredseqs-{clean_allele(allele)}.parquet",
                            path="/parquet/*/",
                            strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                        )
                        for allele in V5_HLA_ALLELES_WITH_HLACOCLUSTERS
                    ]
                )
            )
        else:
            raise ValueError("Version not implemented")

    def output(self) -> str:
        if self.version == "V8":
            return f"ecoclusters.count_matrix.hladb-v8-lowestfdr-{1e-3}.parquet"
        elif self.version == "V5":
            return "ecoclusters.count_matrix.hladb-v5.parquet"
        else:
            raise ValueError("Version not implemented")

    def get_sample_names(self, request: SparkDatasetJobRequest) -> List[str]:
        return sorted(
            request.get("hlas")
            .select('name')
            .toPandas()["name"]
            .tolist()
        )

    def get_hlacocluster_tcrs(self, request: SparkDatasetJobRequest, allele: str) -> DataFrame:
        hlacoclusters_tcrs = (
            request.get(f"{clean_allele(allele)}_hlacoclusters")
            .withColumn("bioIdentity", F.explode(F.col('members')))
            .drop('members', 'hla')
        )

        return hlacoclusters_tcrs

    def count_hlacocluster_tcrs(self, seqs: DataFrame, hlacocluster_tcrs: DataFrame, sample_names: List[str]) -> coo_matrix:
        hlacocluster_tcrs = hlacocluster_tcrs.cache()

        seqs = (
            seqs.join(hlacocluster_tcrs, on='bioIdentity')
            .select('name', 'hlacocluster', 'bioIdentity')
            .where(F.col('name').isin(sample_names))
            .groupBy('name', 'hlacocluster')
            .agg(F.expr("count(bioIdentity) as cr_count"))
        )

        hlacocluster_counts = (
            seqs.groupby('name')
            .pivot("hlacocluster")
            .sum("cr_count")
            .toPandas()
            .fillna(0)
        )

        missing_samples = set(sample_names).difference(hlacocluster_counts['name'].tolist())
        zeroes = pd.DataFrame(0,
                              index=list(missing_samples),
                              columns=[x for x in hlacocluster_counts.columns.tolist() if x is not "name"])

        zeroes['name'] = zeroes.index
        zeroes = zeroes.reset_index(drop=True)
        hlacocluster_counts = pd.concat([hlacocluster_counts, zeroes])

        hlacocluster_counts = hlacocluster_counts.sort_values(by='name')
        hlacocluster_counts.sort_index(axis=1, inplace=True)

        assert hlacocluster_counts.columns.tolist()[:-1] == sorted(hlacocluster_tcrs.select('hlacocluster').drop_duplicates().toPandas()['hlacocluster'].tolist())
        assert hlacocluster_counts["name"].tolist() == sorted(sample_names)

        count_matrix = coo_matrix(
            hlacocluster_counts.drop('name', axis=1).to_numpy(),
            dtype=np.dtype(int),
            shape=(len(sample_names), len(hlacocluster_counts.columns.tolist()[:-1]))
        )

        return count_matrix

    def transform(self, request: SparkDatasetJobRequest) -> DataFrame:
        spark = df2spark(request.get(f"{clean_allele(self.ALLELES[0])}_hlacoclusters"))
        sample_names = self.get_sample_names(request)
        columns = []

        for allele in sorted(self.ALLELES):
            seqs = (
                request.get(f"{clean_allele(allele)}_filteredseqs")
                .select('bioIdentity', 'sample.name')
            )
            hlacocluster_tcrs = self.get_hlacocluster_tcrs(request, allele)
            columns.append(self.count_hlacocluster_tcrs(seqs, hlacocluster_tcrs, sample_names))

        count_matrix = hstack(columns, format="coo", dtype=np.dtype(int))
        coords = [x.tolist() for x in list(count_matrix.coords)]

        return spark.createDataFrame(
            schema=self.schema,
            data=pd.DataFrame({"data": count_matrix.data.tolist(),
                               "coords0": coords[0],
                               "coords1": coords[1]})
        )


def clustering_rcd_inputs(version: str):
    if version == "V8":
        return DatasetInputRequirements(
            tuple(
                [
                    InputRequirement(
                        name="hlas",
                        dataset="repertoire.samples.taccs-hladb-v8.hlas.parquet",
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    ),
                    InputRequirement(
                        name="count_matrix",
                        dataset=f"ecoclusters.count_matrix.hladb-v8-lowestfdr-{1e-3}.parquet",                        
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    ),
                    InputRequirement(
                        name="hlacocluster_mask",
                        dataset=f"ecoclusters.hladb-v8.tacc-mask.parquet",                        
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    ),
                    InputRequirement(
                        name="correlation_matrix_min_samples_0",
                        dataset=f"ecoclusters.hladb-v8.correlation-matrix.min-samples_0.parquet",  # move this  
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    ),
                    InputRequirement(
                        name="correlation_matrix_min_samples_10",
                        dataset=f"ecoclusters.hladb-v8.correlation-matrix.min-samples_10.parquet",                        
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    ),
                    InputRequirement(
                        name="correlation_matrix_with_nans_min_samples_20",
                        dataset=f"ecoclusters.hladb-v8.correlation-matrix.with-nans.min-samples_20.parquet",                    
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    ),
                    InputRequirement(
                        name="hlacocluster_names",
                        dataset=f"ecoclusters.tacc_names.hladb-v8-fdr-{1e-3}.parquet",
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    )
                ]
            )
        )
    elif version == "V5":
        return DatasetInputRequirements(
            tuple(
                [
                    InputRequirement(
                        name="hlas",
                        dataset="repertoire.samples.hladb-v5-hlas.parquet",  # WE NEED TO SWITCH THIS OUT FOR THESE SAME SAMPLES BUT WITH V5 ALLELES!
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    ),
                    InputRequirement(
                        name="count_matrix",
                        dataset="ecoclusters.count_matrix.hladb-v5.parquet",
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    ),
                    InputRequirement(
                        name="hlacocluster_names",
                        dataset="ecoclusters.tacc_names.hladb-v5.parquet",
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    )
                ]
            )
        )
    else:
        raise ValueError("Version not implemented")


def get_metas(request: SparkDatasetJobRequest) -> pd.DataFrame:
    return (
        request.get('hlas')
        .toPandas()
        .sort_values(by='name')
    )


def get_hlacocluster_names(request: SparkDatasetJobRequest) -> List[str]:
    return (
        request.get("hlacocluster_names")
        .toPandas()
        .sort_values(by="order_index")["hlacocluster"]
        .tolist()
    )


# refactor out this duplication
def get_count_matrix(request: SparkDatasetJobRequest, n_samples: int, n_hlacoclusters: int) -> csr_matrix:
    pdf = (
        request.get('count_matrix')
        .toPandas()
    )

    matrix = (
        coo_matrix((pdf['data'].tolist(), (pdf["coords0"].tolist(), pdf["coords1"].tolist())),
                   dtype=np.float32,
                   shape=(n_samples, n_hlacoclusters))
        .tocsr()
    )

    return matrix

def get_hlacocluster_mask(request: SparkDatasetJobRequest, n_samples: int, n_hlacoclusters: int) -> csr_matrix:
    pdf = (
        request.get('hlacocluster_mask')
        .toPandas()
    )

    matrix = (
        coo_matrix((pdf['data'].tolist(), (pdf["coords0"].tolist(), pdf["coords1"].tolist())),
                   dtype=np.float32,
                   shape=(n_samples, n_hlacoclusters))
        .tocsr()
    )

    return matrix

def get_correlation_matrix(request: SparkDatasetJobRequest, n_hlacoclusters: int, n_min_samples: int) -> csr_matrix:
    pdf = (
        request.get(f'correlation_matrix_min_samples_{n_min_samples}')
        .toPandas()
    )

    matrix = (
        coo_matrix((pdf['data'].tolist(), (pdf["coords0"].tolist(), pdf["coords1"].tolist())),
                   dtype=np.float32,
                   shape=(n_hlacoclusters, n_hlacoclusters))
        .tocsr()
    )

    return matrix


def get_correlation_matrix_with_nans(request: SparkDatasetJobRequest, n_hlacoclusters: int, n_min_samples: int) -> csr_matrix:
    pdf = (
        request.get(f'correlation_matrix_with_nans_min_samples_{n_min_samples}')
        .toPandas()
    )

    matrix = (
        coo_matrix((pdf['data'].tolist(), (pdf["coords0"].tolist(), pdf["coords1"].tolist())),
                   dtype=np.float32,
                   shape=(n_hlacoclusters, n_hlacoclusters))
        .tocsr()
    )

    return matrix

class HlaCoclusterMask(ISparkDatasetJob):
    description ="""
    Builds sample x HLA-COcluster sparse binary matrix where 1 means
    this sample has the HLA corresponding to this HLA-COcluster, 0
    means it does not.
    """
    author = DatasetEmail("swoodhouse@adaptivebiotech.com")

    version = "V8"

    schema = StructType(
        [
            StructField(name="data", dataType=LongType(), nullable=False),
            StructField(name="coords0", dataType=LongType(), nullable=False),
            StructField(name="coords1", dataType=LongType(), nullable=False),
        ]
    )

    def __init__(self) -> None:
        super().__init__()

    def inputs(self) -> DatasetInputRequirements:
        return clustering_rcd_inputs(self.version)   

    def output(self) -> str:
        return f"ecoclusters.hladb-v8.tacc-mask.parquet"

    def transform(self, request: SparkDatasetJobRequest) -> DataFrame:
        metas = get_metas(request)
        hlacocluster_names = get_hlacocluster_names(request)
        spark = df2spark(request.get('hlas'))

        mask = build_hlacocluster_mask(metas, hlacocluster_names).tocoo()

        coords = [x.tolist() for x in list(mask.coords)]

        return spark.createDataFrame(
            schema=self.schema,
            data=pd.DataFrame({"data": mask.data.tolist(),
                               "coords0": coords[0],
                               "coords1": coords[1]})
        )


class CorrelationMatrix(ISparkDatasetJob):
    description ="""
    Builds HLA-COcluster x HLA-COcluster sparse HLA-masked correlation matrix.
    """
    author = DatasetEmail("swoodhouse@adaptivebiotech.com")

    version = "V8"
    min_shared_samples = 10

    schema = StructType(
        [
            StructField(name="data", dataType=FloatType(), nullable=True),
            StructField(name="coords0", dataType=LongType(), nullable=True),
            StructField(name="coords1", dataType=LongType(), nullable=True),
        ]
    )

    def __init__(self) -> None:
        super().__init__()

    def inputs(self) -> DatasetInputRequirements:
        return clustering_rcd_inputs(self.version)   

    def output(self) -> str:
        return f"ecoclusters.hladb-v8.correlation-matrix.min-samples_{self.min_shared_samples}.parquet"

    def transform(self, request: SparkDatasetJobRequest) -> DataFrame:
        metas = get_metas(request)
        hlacocluster_names = get_hlacocluster_names(request)
        mask = get_hlacocluster_mask(request, len(metas), len(hlacocluster_names))
        count_matrix = get_count_matrix(request, len(metas), len(hlacocluster_names))
        spark = df2spark(request.get('hlas'))

        # if it is the full correlation matrix we compute it here
        if self.min_shared_samples == 0:
            corr = masked_correlation(count_matrix, mask).tocoo()

        # else, we assume the full one has been computed and manipulate that
        else:
            corr = get_correlation_matrix(request, len(hlacocluster_names), 0)

            min_shared_samples = sparse_map_nonzero(
                mask.T @ mask, lambda x: (x >= self.min_shared_samples).astype(np.float32)
            )

            corr = triu(corr.multiply(min_shared_samples).tocoo())
        
        coords = [x.tolist() for x in list(corr.coords)]
        out = pd.DataFrame({"data": corr.data.tolist(),
                            "coords0": coords[0],
                            "coords1": coords[1]})
        return spark.createDataFrame(
            data=out
        )


class CorrelationMatrixWithNaNs(ISparkDatasetJob):
    description ="""
    Builds HLA-COcluster x HLA-COcluster sparse HLA-masked correlation matrix, with NaNs
    where a pair of HLA-COclusters do not share enough HLA-matched donors, instead of 0.
    """
    author = DatasetEmail("swoodhouse@adaptivebiotech.com")

    version = "V8"
    min_shared_samples = 0

    schema = StructType(
        [
            StructField(name="data", dataType=DoubleType(), nullable=True),
            StructField(name="coords0", dataType=LongType(), nullable=True),
            StructField(name="coords1", dataType=LongType(), nullable=True),
        ]
    )

    def __init__(self) -> None:
        super().__init__()

    def inputs(self) -> DatasetInputRequirements:
        return clustering_rcd_inputs(self.version)   

    def output(self) -> str:
        return f"ecoclusters.hladb-v8.correlation-matrix.with-nans.min-samples_{self.min_shared_samples}.parquet"

    def transform(self, request: SparkDatasetJobRequest) -> DataFrame:
        metas = get_metas(request)
        hlacocluster_names = get_hlacocluster_names(request)
        mask = get_hlacocluster_mask(request, len(metas), len(hlacocluster_names))
        count_matrix = get_count_matrix(request, len(metas), len(hlacocluster_names))
        spark = df2spark(request.get('hlas'))

        logger.info("Loading correlation matrix")

        corr = get_correlation_matrix(request, len(hlacocluster_names), 0)

        logger.info("Masking to NaNs, 1")

        min_shared_samples = sparse_map_nonzero(
            mask.T @ mask, lambda x: np.where(x < self.min_shared_samples, np.nan, 1.0).astype(np.float32)
        )

        corr = triu(corr.multiply(min_shared_samples), format="csr")

        # we also need nans where the mask = 0
        logger.info("Masking to NaNs, 2")

        inverted_masked = 1 - (mask.T @ mask).todense()  # this is not really very sparse anyway
        inverted_masked = triu(csr_matrix(inverted_masked), format="csr")  # this is very sparse
        inverted_masked = sparse_map_nonzero(
            inverted_masked, lambda x: np.full(x.shape, np.nan, dtype=np.float32)
        )

        logger.info("Writing")
        corr = corr.tocoo()
        inverted_masked = inverted_masked.tocoo()

        coords = [x.tolist() for x in list(corr.coords)] + [x.tolist() for x in list(inverted_masked.coords)]
        out = pd.DataFrame({"data": corr.data.tolist(),
                            "coords0": coords[0],
                            "coords1": coords[1]})

        return spark.createDataFrame(
            data=out
        )


class ECOclusteringTree(ISparkDatasetJob):
    description = """
    Build ECOClusters from V8 or V5 HLAdb HLA-COclusters. Clustering is done on driver node.
    No executors are required, but a large amount of driver RAM is.
    """
    author = DatasetEmail("swoodhouse@adaptivebiotech.com")

    min_shared_samples_correlation = 10

    schema = StructType(
        [
            StructField(name="row_index", dataType=LongType(), nullable=False),
            StructField(name="child_1", dataType=LongType(), nullable=False),
            StructField(name="child_2", dataType=LongType(), nullable=False),
            StructField(name="distance", dataType=DoubleType(), nullable=False),
            StructField(name="n_members", dataType=LongType(), nullable=False),
        ]
    )

    def __init__(self, version:str, mask_on_v28_tcrs:bool = False) -> None:
        if mask_on_v28_tcrs:
            raise ValueError("mask_on_v28_tcrs not implemented")

        self.version = version

        super().__init__()

    def inputs(self) -> DatasetInputRequirements:
        return clustering_rcd_inputs(self.version)

    def output(self) -> str:
        if self.version == "V8":
            return f"ecoclusters.clustering-tree.hladb-v8-fdr-{1e-3}-min_shared_samples-{self.min_shared_samples_correlation}.parquet"
        if self.version == "V5":
            return "ecoclusters.clustering-tree.hladb-v5.parquet"
        else:
            raise ValueError("Version not implemented")

    def transform(self, request: SparkDatasetJobRequest) -> DataFrame:
        hlacocluster_names = get_hlacocluster_names(request)
        spark = df2spark(request.get('hlas'))

        corr = get_correlation_matrix(request, len(hlacocluster_names), self.min_shared_samples_correlation)

        # replace any nans with 0
        nan_to_zero = sparse_map_nonzero(
            corr, lambda x: np.where(np.isnan(x), 0.0, 1.0).astype(np.float32)
        )

        corr = triu(corr.multiply(nan_to_zero).tocoo())

        distance = sp.distance.squareform(1 - (corr + corr.T + identity(corr.shape[0])).toarray())

        linkage = hc.linkage(
            # go from upper triangular matrix to expanded matrix, and from corr to distance
            # why are we making this dense and non triangular
            distance,  # REWRITE THIS. this should directly take upper tri
            method="average"
        )

        node_connectivity = pd.DataFrame(
            linkage,
            columns=["child_1", "child_2", "distance", "n_members"]
        )
        node_connectivity["row_index"] = list(range(len(node_connectivity)))
        node_connectivity = node_connectivity[["row_index", "child_1", "child_2", "distance", "n_members"]]

        return spark.createDataFrame(node_connectivity, self.schema)


class ECOclusteringExactTree(ISparkDatasetJob):
    description = """
    Build ECOClustering agglomerative clustering tree from V8 or V5 HLAdb HLA-COclusters. Clustering is done on driver node.
    No executors are required, but a large amount of driver RAM is.
    This RCD computes the exact average correlation distances at each level of the tree rather
    than using average linkage.
    """
    author = DatasetEmail("swoodhouse@adaptivebiotech.com")

    min_shared_samples_correlation = 20

    schema = StructType(
        [
            StructField(name="row_index", dataType=LongType(), nullable=False),
            StructField(name="child_1", dataType=LongType(), nullable=False),
            StructField(name="child_2", dataType=LongType(), nullable=False),
            StructField(name="distance", dataType=DoubleType(), nullable=False),
            StructField(name="n_members", dataType=LongType(), nullable=False),
        ]
    )

    def __init__(self, version:str, mask_on_v28_tcrs:bool = False) -> None:
        if mask_on_v28_tcrs:
            raise ValueError("mask_on_v28_tcrs not implemented")

        self.version = version

        super().__init__()

    def inputs(self) -> DatasetInputRequirements:
        return clustering_rcd_inputs(self.version)

    def output(self) -> str:
        if self.version == "V8":
            return f"ecoclusters.exact-clustering-tree.hladb-v8-fdr-{1e-3}-min_shared_samples-{self.min_shared_samples_correlation}.parquet"
        else:
            raise ValueError("Version not implemented")

    def transform(self, request: SparkDatasetJobRequest) -> DataFrame:
        hlacocluster_names = get_hlacocluster_names(request)
        spark = df2spark(request.get('hlas'))

        corr = get_correlation_matrix_with_nans(request, len(hlacocluster_names), self.min_shared_samples_correlation)

        corr = (corr + corr.T + identity(corr.shape[0])).toarray()

        distance = sp.distance.squareform(1 - np.nan_to_num(corr))  # retain nans in corr itself, but not distance

        linkage = exact_linkage(
            distance,
            corr.shape[0],
            corr,
        )

        node_connectivity = pd.DataFrame(
            linkage,
            columns=["child_1", "child_2", "distance", "n_members"]
        )
        node_connectivity["row_index"] = list(range(len(node_connectivity)))
        node_connectivity = node_connectivity[["row_index", "child_1", "child_2", "distance", "n_members"]]

        return spark.createDataFrame(node_connectivity, self.schema)

# move this. can't import from contib

def report_clusters_entiretree(pdf_hlacocluster_node_connectivity: pd.DataFrame,
                               hlacocluster_names_in_order: List[str],
                               pdf_hlacocluster_tcrs: pd.DataFrame) -> pd.DataFrame:
    """
    Walk up the agglomerativeclustering tree and report the clusters at every
    node.

    Args:
        pdf_hlacocluster_node_connectivity (pd.DataFrame): dataframe describing the clustering tree
        pdf_hlacocluster_tcrs: 
        hlacocluster_names_in_order (List(str)): list of HLA-COcluster names in order

    Returns:
        pd.DataFrame: clusters all through the tree
    """
    
    cluster_distance_threshold = None
            
    # combine all the child nodes together and figure out which ones are bioids and samples
    n_leaf_nodes = len(hlacocluster_names_in_order)
    
    # retain the full connectivity dataframe in _all
    pdf_hlacocluster_node_connectivity_all = pdf_hlacocluster_node_connectivity

    # count members in each cluster
    #cluster_nbioids_map = dict(zip(*[pdf_hlacocluster_tcrs.cluster, pdf_hlacocluster_tcrs.n_bioids]))
    cluster_membercounts = pdf_hlacocluster_tcrs.cluster.value_counts()
    cluster_membercount_map = dict(zip(*[cluster_membercounts.index, cluster_membercounts]))

    # initialize the lists that will become result columns.
    # seed with n_leaf_node dummy rows so that indexes match up
    clusters_at_idxs = [[hlacocluster_name] for hlacocluster_name in hlacocluster_names_in_order]
    bioid_counts = [cluster_membercount_map[cluster] for cluster in hlacocluster_names_in_order]
    distances = [0] * n_leaf_nodes
    child_1_idxs = [-1] * n_leaf_nodes
    child_2_idxs = [-1] * n_leaf_nodes
    
    for _, row in pdf_hlacocluster_node_connectivity.iterrows():
        # lists holding all the _1 and _2 children, respectively,
        # because I write those out
        distance = row["distance"]
        if cluster_distance_threshold is not None and distance > cluster_distance_threshold:
            break
        cluster_hlacocluster_lists_bothcols = []    
        n_bioids = 0 
        for col in ["child_1", "child_2"]:
            cluster_idx = int(row[col])
            if cluster_idx > len(clusters_at_idxs) - 1:
                logger.warning(f"Fail! cluster_idx={cluster_idx}, clusters: {len(clusters_at_idxs)}")
            this_bioid_list = clusters_at_idxs[cluster_idx]
            cluster_hlacocluster_lists_bothcols.append(this_bioid_list)
            if col == "child_1":
                child_1_idxs.append(cluster_idx)
            else:
                child_2_idxs.append(cluster_idx)
            n_bioids += bioid_counts[cluster_idx]
            
        cluster_child1_hlacoclusters, cluster_child2_hlacoclusters = cluster_hlacocluster_lists_bothcols
        
        # hold all the HLA-COclusters in this cluster
        this_cluster_hlacoclusters = cluster_child1_hlacoclusters + cluster_child2_hlacoclusters
        
        bioid_counts.append(n_bioids)
        distances.append(distance)
        clusters_at_idxs.append(this_cluster_hlacoclusters)

    # make the clusters dataframe with everything
    pdf_clusters_distances = pd.DataFrame({
        "cluster_members": clusters_at_idxs,
        "n_hlacoclusters": [len(x) for x in clusters_at_idxs],
        "n_bioids": bioid_counts,
        "distance": distances,
        "tree_idx": range(len(clusters_at_idxs)),
        "child_1_idx": child_1_idxs,
        "child_2_idx": child_2_idxs,
    })

    # throw out everything over the distance threshold
    if cluster_distance_threshold is not None:
        pdf_clusters_distances = pdf_clusters_distances[pdf_clusters_distances.distance < cluster_distance_threshold]
        logger.debug(f"    distance threshold: {cluster_distance_threshold}, tree size: {len(pdf_clusters_distances)}")

    # Calculate all distances as percentiles of the full distance distribution
    pdf_clusters_distances["distance_percentile"] = percentileofscore(
        pdf_hlacocluster_node_connectivity_all.distance, pdf_clusters_distances.distance)

    return pdf_clusters_distances

class EcoclusterConstructor:
    def __init__(self,
                 min_members: int = DEFAULT_MIN_MEMBERS,
                 max_members: int = DEFAULT_MAX_MEMBERS,
                 max_distance: Optional[float] = None) -> None:
        """
        Args:
            min_members (int, optional): Minimum HLA-COclusters to combine. Defaults to DEFAULT_MIN_MEMBERS.
            max_members (int, optional): Maximum HLA-COclusters to combine. Defaults to DEFAULT_MAX_MEMBERS.
            max_distance (float, optional): Maximum distance in the tree to combine. Defaults to None.
            max_mean_hlacoclusters_per_hla (float, optional): Maximum mean HLA-COclusters per HLA. Defaults to DEFAULT_MAX_MEAN_HLACOCLUSTERS_PER_HLA.
        """
        self.min_members = min_members
        self.max_members = max_members
        self.max_distance = max_distance

    def cluster_ecoclusters(self, pdf_node_connectivity: pd.DataFrame,
                            hlacocluster_names_in_order: List[str],
                            pdf_hlacocluster_tcrs: pd.DataFrame) -> pd.DataFrame:
        """
        Construct ECOClusters, in a cross-HLA clustering-of-clusters setting. Steps:

        1. Walk up the tree accumulating clusters
        2. Walk down the tree in descending-sorted distance order. Return the first cluster
        encountered, for each TCR, that satisfies min_members (minimum number of HLA-COclusters
        subsumed), max_members (ditto but maximum) max_distance (maximum distance) and
        max_mean_hlacoclusters_per_hla (maximum mean # HLA-COclusters per HLA).
        3. Annotate each cluster with its set of HLAs

        This class is capable of splitting ECOclusters at the first level in the tree at which
        the mean number of HLAs per HLA-COcluster exceeds a threshold, but by default this is set to an
        impossibly high number so that no splitting is done.

        Args:
            pdf_node_connectivity (PandasDataFrame): dataframe containing the AgglomerativeClustering result
            hlacocluster_names_in_order: list of the HLA-COcluster names in index order as referred to in above arg
            pdf_hlacocluster_tcrs: HLA-COcluster dataframe with one row per bioIdentity+HLA-COcluster
        Raises:
            ValueError: raised if the clustering fails because the tree wasn't arranged properly.

        Returns:
            PandasDataFrame: The defined cross-HLA clustering. Similar to the within-HLA
                clustering, but no "hla" column
        """
        pdf_cluster_tree = report_clusters_entiretree(
            pdf_node_connectivity, hlacocluster_names_in_order, pdf_hlacocluster_tcrs)

        all_members = set().union(*[set(x) for x in pdf_cluster_tree.cluster_members])
        pdf_cluster_tree["order"] = range(len(pdf_cluster_tree))
        pdf_cluster_tree_distance_desc = pdf_cluster_tree.sort_values("order", ascending=False)
        
        if self.max_distance is not None:
            pdf_cluster_tree_distance_desc = pdf_cluster_tree_distance_desc[
                pdf_cluster_tree_distance_desc.distance <= self.max_distance]
        
        hlacoclusters_in_clusters: Set[str] = set()

        # make for easy lookup of HLA-COcluster HLAs
        pdf_hlacocluster_hlas = pdf_hlacocluster_tcrs.drop_duplicates("cluster")
        hlacocluster_hla_map = dict(zip(*[pdf_hlacocluster_hlas.cluster,
                                pdf_hlacocluster_hlas.hla])) 
        def lookup_hlacoclusterlist_hlas(hlacocluster_list: List[str]) -> Set[str]:
            return set([hlacocluster_hla_map[hlacocluster] for hlacocluster in hlacocluster_list])

        pdf_cluster_tree_distance_desc["n_hlacoclusters"] = pdf_cluster_tree_distance_desc.cluster_members.apply(len)

        # Calculate mean HLAs per HLA-COcluster at each node
        pdf_cluster_tree_distance_desc["hlas"] = pdf_cluster_tree_distance_desc.cluster_members.apply(
            lookup_hlacoclusterlist_hlas)
        pdf_cluster_tree_distance_desc["n_hlas"] = pdf_cluster_tree_distance_desc.hlas.apply(len)
        pdf_cluster_tree_distance_desc["mean_hlas_per_hlacocluster"] = (
            pdf_cluster_tree_distance_desc.n_hlacoclusters / pdf_cluster_tree_distance_desc.n_hlas)

        # filter the tree on distance and mean HLAs per HLA-COcluster
        pdf_cluster_tree_distance_desc = pdf_cluster_tree_distance_desc[
            #(pdf_cluster_tree_distance_desc.mean_hlas_per_hlacocluster <= self.max_mean_hlacoclusters_per_hla) &
            (pdf_cluster_tree_distance_desc.n_hlacoclusters <= self.max_members) &
            (pdf_cluster_tree_distance_desc.n_hlacoclusters >= self.min_members)]
        
        cluster_hlacocluster_members = []
        cluster_distances = []
        for _, row in pdf_cluster_tree_distance_desc.iterrows():
            cluster_hlacoclusters_set = set(row["cluster_members"])
            clusterhlacoclusters_inters_already = cluster_hlacoclusters_set.intersection(hlacoclusters_in_clusters)
            if len(clusterhlacoclusters_inters_already) == 0:
                # no overlap with existing clusters
                # new cluster, add it
                hlacoclusters_in_clusters.update(cluster_hlacoclusters_set)
                cluster_hlacocluster_members.append(row["cluster_members"])
                cluster_distances.append(row["distance"])
            else:
                # overlap with existing clusters. It had better be the whole thing.
                if not len(clusterhlacoclusters_inters_already) == len(cluster_hlacoclusters_set):
                    # we should never see a cluster with some members already accounted for and others not
                    raise ValueError(
                        f"ERROR! cluster HLA-COclusters already clustered: {clusterhlacoclusters_inters_already}, " +
                        "total in set: {cluster_hlacoclusters_set}")
        logger.debug(f"    Clusters: {len(cluster_hlacocluster_members)}. Retained {len(hlacoclusters_in_clusters)} " +
                    f"of {len(all_members)} HLA-COclusters")

        # I originally added an array of all component-cluster HLAs as an "hla" column here.
        # But that drove spark insane, made the dataset impossible to work with, I don't
        # know why.

        pdf_ecoclustering = pd.DataFrame({
            "cluster": [f"e-{i}" for i in range(len(cluster_hlacocluster_members))],
            "members": cluster_hlacocluster_members,
            "ASR_in_cluster": [0] * len(cluster_hlacocluster_members), # dummy value
            "cluster_distance": cluster_distances,
        })
        pdf_ecoclustering["n_members"] = [len(x) for x in pdf_ecoclustering.members]
        pdf_ecoclustering["member_frequencies"] = [[0] * n for n in pdf_ecoclustering.n_members] # dummy values
        
        # reorder cluster tight to loose, just for fun
        if len(pdf_ecoclustering) > 0:
            pdf_ecoclustering = pdf_ecoclustering.sort_values("cluster_distance", ascending=False)
        pdf_ecocluster_tcrs = self.reformat_ecoclusters_one_row_per_tcr(
            pdf_ecoclustering, pdf_hlacocluster_tcrs
        )
        return pdf_ecocluster_tcrs

    def reformat_ecoclusters_one_row_per_tcr(self,
            pdf_ecoclustering: pd.DataFrame,
            pdf_hlacocluster_tcrs: pd.DataFrame) -> pd.DataFrame:
        pdf_hlacocluster_ecocluster_map = pdf_ecoclustering.explode("members")[  # type: ignore
            ["cluster", "members"]].rename(columns={
             "cluster": "ecocluster",
             "members": "cluster"
        })
        pdf_ecocluster_tcrs = pdf_hlacocluster_tcrs.merge(pdf_hlacocluster_ecocluster_map, on="cluster")
        # reorder columns, putting ecocluster first
        pdf_ecocluster_tcrs = pdf_ecocluster_tcrs[["ecocluster"] +
                                                  [c for c in pdf_ecocluster_tcrs.columns
                                                   if c != "ecocluster"]]
        return pdf_ecocluster_tcrs


class ECOclusteringAgglomerative(ISparkDatasetJob):
    description = """
    ...
    """
    author = DatasetEmail("swoodhouse@adaptivebiotech.com")

    max_distance = 0.4 # 0.4-0.8
    min_shared_samples_correlation = 10

    schema = StructType(
        [
            StructField(name="ecocluster", dataType=StringType(), nullable=False),
            StructField(name="hlacocluster", dataType=StringType(), nullable=False),
            StructField(name="hla", dataType=StringType(), nullable=False),
            StructField(name="bioIdentity", dataType=StringType(), nullable=False),
        ]
    )

    def __init__(self, version: str, mask_on_v28_tcrs: bool = False) -> None:
        if mask_on_v28_tcrs:
            raise ValueError("mask_on_v28_tcrs not implemented")

        self.version = version

        super().__init__()

    def inputs(self) -> DatasetInputRequirements:
        if self.version == "V8":
            return DatasetInputRequirements(
                tuple(
                    [
                        InputRequirement(
                            name="clustering_tree",
                            dataset=f"ecoclusters.clustering-tree.hladb-v8-fdr-{1e-3}-min_shared_samples-{self.min_shared_samples_correlation}.parquet",
                            path="/parquet/*/",
                            strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                        ),
                        InputRequirement(
                            name="hlacocluster_names",
                            dataset=f"ecoclusters.tacc_names.hladb-v8-fdr-{1e-3}.parquet",
                            path="/parquet/*/",
                            strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                        )
                    ] +
                    [
                        InputRequirement(
                            name=f"{clean_allele(allele)}_hlacoclusters",
                            dataset=f"taccs.hladb-v8-{clean_allele(allele)}-fdr-{1e-3}.parquet",
                            path="/parquet/*/",
                            strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                        )
                        for allele in V8_HLA_ALLELES_WITH_HLACOCLUSTERS
                    ]
                )
            )
        else:
            raise ValueError("not implemented")

    def output(self) -> str:
        if self.version == "V8":
            return f"ecoclusters.agglomerative-clustering.hladb-v8-lowestfdr-{1e-3}-min_shared_samples-{self.min_shared_samples_correlation}-maxdistance-{self.max_distance}.parquet"
        if self.version == "V5":
            return f"ecoclusters.agglomerative-clustering.hladb-v5-maxdistance-{self.max_distance}.parquet"
        else:
            raise ValueError("Version not implemented")

    # duplicated. refactor
    def get_hlacocluster_tcrs(self, request: SparkDatasetJobRequest) -> pd.DataFrame:
        if self.version == "V8":
            alleles = V8_HLA_ALLELES_WITH_HLACOCLUSTERS
        elif self.version == "V5":
            alleles = V5_HLA_ALLELES_WITH_HLACOCLUSTERS
        else:
            raise ValueError("Version not implemented")

        pdfs = []

        for allele in alleles:
            pdfs.append(
                request.get(f"{clean_allele(allele)}_hlacoclusters")
                .withColumn("bioIdentity", F.explode(F.col('members')))
                .drop('members')
                .withColumn('hla', F.lit(allele))
                .toPandas()
            )

        return pd.concat(pdfs)

    def transform(self, request: SparkDatasetJobRequest) -> DataFrame:
        spark = df2spark(request.get("clustering_tree"))
        node_connectivity = request.get("clustering_tree").toPandas()
        node_connectivity = node_connectivity.sort_values(by='row_index')

        hlacocluster_names_in_order = get_hlacocluster_names(request)
        hlacocluster_tcrs = (
            self.get_hlacocluster_tcrs(request)
            .rename(columns={"hlacocluster": "cluster"})
        )

        ecocluster_tcrs = (
            EcoclusterConstructor(max_distance=self.max_distance)
            .cluster_ecoclusters(node_connectivity, hlacocluster_names_in_order, hlacocluster_tcrs)
            .rename(columns={"cluster": "hlacocluster"})
        )

        return spark.createDataFrame(
            schema=self.schema,
            data=ecocluster_tcrs,
        )


def parent_map(node_connectivity: pd.DataFrame, n_hlacoclusters: int):
    parent = dict()
    for row in node_connectivity.itertuples():
        if row.child_1 >= n_hlacoclusters: 
            parent[row.child_1 - n_hlacoclusters] = row.row_index
        if row.child_2 >= n_hlacoclusters:
            parent[row.child_2 - n_hlacoclusters] = row.row_index
    return parent

def compressed_parent_map(parents, cuts):
    compressed_parents = {}

    for n1 in cuts:
        cur = n1
        while cur in parents:
            cur = parents[cur]
            if cur in cuts:
                compressed_parents[n1] = cur
                break

    return compressed_parents


class ECOclusteringElbows(ISparkDatasetJob):
    description = """
    Cut the agglomerative clustering tree at local "elbow" points, rather than at a uniform fixed distance.
    """
    author = DatasetEmail("swoodhouse@adaptivebiotech.com")

    min_shared_samples_correlation = 20
    min_second_deriv = 0.05  # vary this

    schema = StructType(
        [
            StructField(name="ecocluster", dataType=StringType(), nullable=False),
            StructField(name="cut", dataType=LongType(), nullable=True),
            StructField(name="parent_cut", dataType=LongType(), nullable=True),
            StructField(name="depth", dataType=LongType(), nullable=False),
            StructField(name="hlacocluster", dataType=StringType(), nullable=False),
            StructField(name="hla", dataType=StringType(), nullable=False),
            StructField(name="bioIdentity", dataType=StringType(), nullable=False),
        ]
    )

    def __init__(self, version: str) -> None:
        if version != "V8":
            raise ValueError("Not implemented")

        super().__init__()

    def output(self) -> str:
        return f"ecoclusters.elbow-clustering-nested.hladb-v8-lowestfdr-{1e-3}-min_shared_samples-{self.min_shared_samples_correlation}-min_second_deriv-{self.min_second_deriv}.parquet"

    def inputs(self) -> DatasetInputRequirements:
        return DatasetInputRequirements(
            tuple(
                [
                    InputRequirement(
                        name="clustering_tree",
                        dataset=f"ecoclusters.exact-clustering-tree.hladb-v8-fdr-{1e-3}-min_shared_samples-{self.min_shared_samples_correlation}.parquet",
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    ),
                    InputRequirement(
                        name="hlacocluster_names",
                        dataset=f"ecoclusters.tacc_names.hladb-v8-fdr-{1e-3}.parquet",
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    )
                ] +
                [
                    InputRequirement(
                        name=f"{clean_allele(allele)}_hlacoclusters",
                        dataset=f"taccs.hladb-v8-{clean_allele(allele)}-fdr-{1e-3}.parquet",
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    )
                    for allele in V8_HLA_ALLELES_WITH_HLACOCLUSTERS
                ]
            )
        )

    def get_hlacocluster_tcrs(self, request: SparkDatasetJobRequest) -> pd.DataFrame:
        pdfs = []

        for allele in V8_HLA_ALLELES_WITH_HLACOCLUSTERS:
            pdfs.append(
                request.get(f"{clean_allele(allele)}_hlacoclusters")
                .withColumn("bioIdentity", F.explode(F.col('members')))
                .drop('members')
                .withColumn('hla', F.lit(allele))
                .toPandas()
            )

        return pd.concat(pdfs)

    def walk_tree(self, node_connectivity: pd.DataFrame, hlacocluster_names: List[str]) -> Tuple[Dict[str, List[float]], Dict[str, List[int]]]:
        n_leaf_nodes = len(hlacocluster_names)
        clusters_at_idxs = [[t] for t in hlacocluster_names]

        hlacocluster_correlations = defaultdict(list)
        hlacocluster_paths = defaultdict(list)

        # walk up the tree, left to right at each level
        for _, row in node_connectivity.iterrows():
            cluster_hlacocluster_lists_bothcols = []

            # step down one level
            for col in ["child_1", "child_2"]:
                cluster_idx = int(row[col])
                if cluster_idx == -1:
                    this_hlacocluster_list = []
                else:
                    this_hlacocluster_list = clusters_at_idxs[cluster_idx]
                cluster_hlacocluster_lists_bothcols.append(this_hlacocluster_list)

            cluster_child1_hlacoclusters, cluster_child2_hlacoclusters = cluster_hlacocluster_lists_bothcols

            # hold all the HLA-COclusters in this cluster
            clusters_at_idxs.append(cluster_child1_hlacoclusters + cluster_child2_hlacoclusters)

            for hlacocluster in cluster_child1_hlacoclusters + cluster_child2_hlacoclusters:
                hlacocluster_correlations[hlacocluster] = [1 - row["distance"]] + hlacocluster_correlations[hlacocluster]
                hlacocluster_paths[hlacocluster] = [int(row.row_index)] + hlacocluster_paths[hlacocluster]

        return hlacocluster_correlations, hlacocluster_paths

    def cut_at_max_2nd_deriv(self, hlacocluster_correlations: Dict[str, List[float]], hlacocluster_paths:Dict[str, List[int]], min_2nd_deriv:Optional[float]=None) -> Dict[str, int]:
        cuts = dict()

        for hlacocluster, corrs in hlacocluster_correlations.items():
            path = hlacocluster_paths[hlacocluster]
            second = np.diff(np.diff(corrs))
            idx = np.argmax(second)
            max_second = second[idx]
            if min_2nd_deriv and max_second < min_2nd_deriv:
                continue
            cut = path[idx + 2]  # align with center of second derivative
            cuts[hlacocluster] = cut

        return cuts

    def cuts_to_clusters(self, clustering_tree: pd.DataFrame, cuts: List[int], hlacocluster_names: List[str]):
        clusters_at_idxs = [[t] for t in hlacocluster_names]

        for _, row in clustering_tree.iterrows():
            cluster_hlacocluster_lists_bothcols = []

            # step down one level
            for col in ["child_1", "child_2"]:
                cluster_idx = int(row[col])
                this_hlacocluster_list = clusters_at_idxs[cluster_idx]
                cluster_hlacocluster_lists_bothcols.append(this_hlacocluster_list)

            cluster_child1_hlacoclusters, cluster_child2_hlacoclusters = cluster_hlacocluster_lists_bothcols

            # hold all the HLA-COclusters in this cluster
            clusters_at_idxs.append(cluster_child1_hlacoclusters + cluster_child2_hlacoclusters)

        clusters = pd.DataFrame({"cut":cuts, "hlacoclusters":[clusters_at_idxs[i + len(hlacocluster_names)] for i in cuts]})
        return clusters

    def cut_elbows(self, clustering_tree: pd.DataFrame, hlacocluster_names: List[str]) -> pd.DataFrame:
        compressed_hlacocluster_correlations, compressed_hlacocluster_paths = self.walk_tree(
            clustering_tree, hlacocluster_names
        )

        cuts = self.cut_at_max_2nd_deriv(
            compressed_hlacocluster_correlations, compressed_hlacocluster_paths, self.min_second_deriv
        )

        clusters = self.cuts_to_clusters(clustering_tree, list(cuts.values()), hlacocluster_names)

        clusters["len_hlacoclusters"] = [len(t) for t in clusters["hlacoclusters"]]
        return clusters

    def name_clusters(self, ecoclusters: pd.DataFrame, clustering_tree: pd.DataFrame, n_hlacoclusters: int) -> pd.DataFrame:
        cuts = frozenset(ecoclusters['cut'].tolist())
        parents = compressed_parent_map(parent_map(clustering_tree, n_hlacoclusters), cuts)
        names = dict()

        for row in ecoclusters.sort_values(by='len_hlacoclusters', ascending=False).itertuples():
            node = row.cut
            if node in names:
                continue 
            if not node in parents:
                names[node] = f"e-{node}"
            else:
                node2 = node
                parent_chain = []
                while node2 in parents:
                    p = parents[node2]
                    if p in names:
                        names[node2] = f"{names[p]}-{node2}"
                        parent_chain.append(p)
                        break
                    else:
                        parent_chain.append(p)
                    node2 = p
                
                if len(parent_chain) > 0:
                    parent_chain.reverse()

                    for i in range(len(parent_chain)-1):
                        node2 = parent_chain[i+1]
                        names[node2] = f"{names[parent_chain[i]]}-{node2}"

        names_pdf = pd.DataFrame({"cut": names.keys(), "ecocluster": names.values()})
        ecoclusters = ecoclusters.merge(names_pdf, on="cut")
        ecoclusters["parent_cut"] = [parents[x] if x in parents else None for x in ecoclusters["cut"]]
        ecoclusters["depth"] = [x.count('-')-1 for x in ecoclusters["ecocluster"]]

        return ecoclusters

    def get_singletons(self, hlacocluster_names_in_order: List[str], ecoclusters: pd.DataFrame) -> pd.DataFrame:
        clustered_hlacoclusters = set(itertools.chain.from_iterable(ecoclusters.hlacoclusters.tolist()))
        unclustered_hlacoclusters = [t for t in hlacocluster_names_in_order if t not in clustered_hlacoclusters]
        names = [f"e-{t}" for t in unclustered_hlacoclusters]

        singletons = pd.DataFrame({"ecocluster":names, "cut":None, "parent_cut":None, "depth":0, "hlacoclusters": [[t] for t in unclustered_hlacoclusters]})
        return singletons

    def transform(self, request: SparkDatasetJobRequest) -> DataFrame:
        logger.info("Loading data")
        spark = df2spark(request.get("clustering_tree"))
        node_connectivity = request.get("clustering_tree").toPandas()
        node_connectivity = node_connectivity.sort_values(by='row_index')

        hlacocluster_names_in_order = get_hlacocluster_names(request)
        hlacocluster_tcrs = self.get_hlacocluster_tcrs(request)

        logger.info("Cutting elbows")
        ecoclusters = self.cut_elbows(node_connectivity, hlacocluster_names_in_order)

        logger.info("Naming clusters")
        ecoclusters = self.name_clusters(ecoclusters, node_connectivity, len(hlacocluster_names_in_order))

        logger.info("Adding singletons HLA-COclusters")
        singletons = self.get_singletons(hlacocluster_names_in_order, ecoclusters)
        ecoclusters = pd.concat([ecoclusters, singletons])

        ecocluster_tcrs = (
            ecoclusters.explode('hlacoclusters')
            .rename(columns={'hlacoclusters':'hlacocluster'})
            .merge(hlacocluster_tcrs, on="hlacocluster")[["ecocluster", "cut", "parent_cut", "depth", "hlacocluster", "hla", "bioIdentity"]]
            .drop_duplicates()
        )

        return spark.createDataFrame(
            schema=self.schema,
            data=ecocluster_tcrs
        )


class ECOclusteringTopElbows(ISparkDatasetJob):
    description = """
    Drop all nested elbows, keeping only the top ones as our "ecoclusters"
    """

    author = DatasetEmail("swoodhouse@adaptivebiotech.com")

    min_shared_samples_correlation = 20
    min_second_deriv = 0.05

    schema = StructType(
        [
            StructField(name="ecocluster", dataType=StringType(), nullable=False),
            StructField(name="hlacocluster", dataType=StringType(), nullable=False),
            StructField(name="hla", dataType=StringType(), nullable=False),
            StructField(name="bioIdentity", dataType=StringType(), nullable=False),
        ]
    )

    def __init__(self, version: str) -> None:
        if version != "V8":
            raise ValueError("Not implemented")

        super().__init__()

    def output(self) -> str:
        return f"ecoclusters.elbow-clustering-notnested.hladb-v8-lowestfdr-{1e-3}-min_shared_samples-{self.min_shared_samples_correlation}-min_second_deriv-{self.min_second_deriv}.parquet"        

    def inputs(self) -> DatasetInputRequirements:
        return DatasetInputRequirements(
            tuple(
                [
                    InputRequirement(
                        name="nested_clusters",
                        dataset=f"ecoclusters.elbow-clustering-nested.hladb-v8-lowestfdr-{1e-3}-min_shared_samples-{self.min_shared_samples_correlation}-min_second_deriv-{self.min_second_deriv}.parquet",
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    ),
                ]
            )
        )

    def transform(self, request: SparkDatasetJobRequest) -> DataFrame:
        logger.info("Loading data")
        nested = request.get("nested_clusters")

        not_nested = (
            nested.where(F.col("depth") == 0)
            .select("ecocluster", "hlacocluster", "hla", "bioIdentity")
        )

        return not_nested
