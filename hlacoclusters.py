import logging
from typing import List, Tuple, Union
from functools import reduce
import pandas as pd
import numpy as np
from sklearn.cluster._bicluster import _scale_normalize
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import assert_all_finite
from hdbscan import HDBSCAN
import umap
from pyspark.sql import DataFrame
import scipy.sparse as sparse

from pyspark.sql.types import (
    StringType,
    StructField,
    StructType,
    ArrayType,
)
from ecocluster_extras import clean_allele

logger = logging.getLogger(__name__)

V8_HLA_ALLELES_WITH_HLACOCLUSTERS = [
    'A*01:01',
    'A*02:01',
    'A*02:02',
    'A*02:03',
    'A*02:05',
    'A*02:06',
    'A*02:07',
    'A*03:01',
    'A*11:01',
    'A*23:01P',
    'A*24:02',
    'A*25:01',
    'A*26:01',
    'A*29:02',
    'A*30:01',
    'A*30:02',
    'A*31:01',
    'A*32:01',
    'A*33:01',
    'A*33:03',
    'A*34:02',
    'A*68:01',
    'A*68:02',
    'A*74:01',
    'B*07:02',
    'B*07:05P',
    'B*08:01',
    'B*13:01',
    'B*14:02',
    'B*15:01',
    'B*15:02',
    'B*15:03',
    'B*15:16',
    'B*15:17',
    'B*15:18',
    'B*18:01',
    'B*27:04',
    'B*27:05',
    'B*35:01',
    'B*35:03',
    'B*35:05',
    'B*35:08',
    'B*37:01',
    'B*38:01',
    'B*38:02',
    'B*39:01',
    'B*39:06',
    'B*40:01',
    'B*40:02',
    'B*40:06',
    'B*41:01',
    'B*42:01',
    'B*44:02',
    'B*44:03',
    'B*44:05',
    'B*45:01',
    'B*46:01',
    'B*47:01',
    'B*48:01',
    'B*49:01',
    'B*50:01',
    'B*51:01',
    'B*52:01',
    'B*55:01',
    'B*57:01',
    'B*57:03',
    'B*58:01',
    'C*01:02',
    'C*02:02P',
    'C*03:04',
    'C*04:01',
    'C*05:01',
    'C*06:02',
    'C*07:01P',
    'C*07:02',
    'C*08:02',
    'C*12:02',
    'C*12:03',
    'C*16:01',
    'DPA1*01:03+DPB1*02:01',
    'DPA1*01:03+DPB1*02:02',
    'DPA1*01:03+DPB1*03:01',
    'DPA1*01:03+DPB1*04:01',
    'DPA1*01:03+DPB1*04:02',
    'DPA1*01:03+DPB1*06:01',
    'DPA1*01:03+DPB1*104:01',
    'DPA1*01:03+DPB1*13:01',
    'DPA1*01:03+DPB1*15:01',
    'DPA1*01:03+DPB1*18:01',
    'DPA1*02:01+DPB1*01:01',
    'DPA1*02:01+DPB1*09:01',
    'DPA1*02:01+DPB1*131:01',
    'DPA1*02:01+DPB1*13:01',
    'DPA1*02:01+DPB1*14:01',
    'DPA1*02:01+DPB1*17:01',
    'DPA1*02:02+DPB1*01:01',
    'DPA1*02:02+DPB1*05:01',
    'DPA1*02:02+DPB1*135:01',
    'DPA1*02:06+DPB1*05:01',
    'DPA1*02:07+DPB1*19:01',
    'DQA1*01:01+DQB1*02:01',
    'DQA1*01:01+DQB1*02:02',
    'DQA1*01:01+DQB1*05:01',
    'DQA1*01:01+DQB1*05:03',
    'DQA1*01:01+DQB1*06:02',
    'DQA1*01:01+DQB1*06:03',
    'DQA1*01:02+DQB1*03:01',
    'DQA1*01:02+DQB1*05:02',
    'DQA1*01:02+DQB1*06:01',
    'DQA1*01:02+DQB1*06:02',
    'DQA1*01:02+DQB1*06:04',
    'DQA1*01:02+DQB1*06:09',
    'DQA1*01:03+DQB1*02:01',
    'DQA1*01:03+DQB1*05:01',
    'DQA1*01:03+DQB1*05:03',
    'DQA1*01:03+DQB1*06:01',
    'DQA1*01:03+DQB1*06:02',
    'DQA1*01:03+DQB1*06:03',
    'DQA1*01:04+DQB1*05:01',
    'DQA1*01:04+DQB1*05:03',
    'DQA1*01:04+DQB1*06:02',
    'DQA1*01:05+DQB1*05:01',
    'DQA1*02:01+DQB1*02:01',
    'DQA1*02:01+DQB1*02:02',
    'DQA1*02:01+DQB1*04:02',
    'DQA1*02:01+DQB1*05:01',
    'DQA1*02:01+DQB1*06:02',
    'DQA1*02:01+DQB1*06:03',
    'DQA1*03:01+DQB1*02:02',
    'DQA1*03:01+DQB1*03:02',
    'DQA1*03:01+DQB1*05:01',
    'DQA1*03:02+DQB1*03:03',
    'DQA1*03:03+DQB1*02:01',
    'DQA1*03:03+DQB1*02:02',
    'DQA1*03:03+DQB1*03:01',
    'DQA1*04:01+DQB1*03:19',
    'DQA1*04:01+DQB1*04:02',
    'DQA1*05:01+DQB1*02:01',
    'DQA1*05:01+DQB1*02:02',
    'DQA1*05:01+DQB1*03:01',
    'DQA1*05:01+DQB1*05:01',
    'DQA1*05:01+DQB1*05:03',
    'DQA1*05:01+DQB1*06:02',
    'DQA1*05:01+DQB1*06:03',
    'DQA1*05:03+DQB1*03:01',
    'DQA1*05:05+DQB1*02:01',
    'DQA1*05:05+DQB1*02:02',
    'DQA1*05:05+DQB1*03:01',
    'DQA1*05:05+DQB1*03:19',
    'DQA1*05:05+DQB1*05:01',
    'DQA1*05:09+DQB1*03:01',
    'DQA1*06:01+DQB1*03:01',
    'DRB1*01:01',
    'DRB1*01:02',
    'DRB1*01:03',
    'DRB1*03:01',
    'DRB1*03:02',
    'DRB1*04:01',
    'DRB1*04:02',
    'DRB1*04:03',
    'DRB1*04:04',
    'DRB1*04:05',
    'DRB1*04:07',
    'DRB1*04:08',
    'DRB1*04:11',
    'DRB1*07:01',
    'DRB1*08:01',
    'DRB1*08:02',
    'DRB1*08:03',
    'DRB1*08:04',
    'DRB1*09:01',
    'DRB1*10:01',
    'DRB1*11:01',
    'DRB1*11:02',
    'DRB1*11:03',
    'DRB1*11:04',
    'DRB1*12:01',
    'DRB1*12:02',
    'DRB1*13:01',
    'DRB1*13:02',
    'DRB1*13:03',
    'DRB1*13:04',
    'DRB1*14:01P',
    'DRB1*14:04',
    'DRB1*14:06',
    'DRB1*15:01',
    'DRB1*15:02',
    'DRB1*15:03',
    'DRB1*16:02P',
    'DRB3*01:01P',
    'DRB3*02:02',
    'DRB3*03:01',
    'DRB4*01:01P',
    'DRB5*01:01',
    'DRB5*01:02',
    'DRB5*02:02P'
]


def svd(array, n_components, n_discard):
    """Returns first `n_components` left and right singular
    vectors u and v, discarding the first `n_discard`.
    """
    u, _, vt = randomized_svd(
        array, n_components,
        random_state=None,
    )

    assert_all_finite(u)
    assert_all_finite(vt)
    u = u[:, n_discard:]
    vt = vt[n_discard:]
    return u, vt.T


class ClusterHlaCoclusters(ISparkDatasetJob):  ## YES!
    description = """
    For a specific allele, build HLA-COclusters from HLAdb v8. Clustering is done in driver node,
    so large number of executors is not required.
    """

    n_svd_components = 150
    n_umap_components = 15
    hdbscan_min_samples = 10
    hdbscan_min_cluster_size = 10
    min_tcrs_per_cluster = 5
    fdr_cutoff = 1e-3

    schema = StructType([
        StructField('hlacocluster', StringType(), True),
        StructField('hla', StringType(), True),
        StructField('members', ArrayType(StringType()), True),
    ])

    def __init__(self, allele: str) -> None:
        self.allele = allele

        super().__init__()

    def inputs(self) -> DatasetInputRequirements:
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
                        name="bioids",
                        dataset=f"repertoire.sequences.hladb-v8-bioids-{clean_allele(self.allele)}-fdr-{self.fdr_cutoff}.parquet",
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    ),
                    InputRequirement(
                        name="seqs",
                        dataset=f"repertoire.sequences.hladb-v8-filteredseqs-{clean_allele(self.allele)}-fdr-{self.fdr_cutoff}.parquet",
                        path="/parquet/*/",
                        strategy=DatasetResolverStrategy.MATCH_OUTPUT,
                    ),

                ]
            )
        )

    def output(self) -> str:
        return f"taccs.hladb-v8-{clean_allele(self.allele)}-fdr-{self.fdr_cutoff}.parquet"

    def get_counts(
        self,
        request: SparkDatasetJobRequest
    ) -> Tuple[sparse.csr_matrix, np.ndarray, int, List[str], List[str]]:
        pdf_sample_hlas = request.get("hlas").toPandas()
        pdf_sample_hlas.index = pdf_sample_hlas["name"]

        bioids = [
            b[0] for b in request.get("bioids").orderBy("order_index").select("bioIdentity").collect()
        ]

        seqs = request.get("seqs")

        X_count_matrix, hla_counted_sample_names = _get_dataframe_counts(
            seqs,
            len(bioids),
            return_dense=False,

        )
        X_count_matrix = X_count_matrix.tocsr()  # type:ignore

        samples_to_cluster = list(
            set(hla_counted_sample_names).intersection(set(pdf_sample_hlas.name))
        )
        logger.info(
            f"    Samples w/ TCR counts for this HLA: {len(hla_counted_sample_names)}. "
            f"Intersection with known sample HLAs: {len(samples_to_cluster)}."
        )

        hhh = pdf_sample_hlas.loc[samples_to_cluster]
        X_hla, hla_index_map = get_imputed_HLA_indicator_vector(
            hhh.x_inferred_hla_class_i.tolist(), hhh.x_inferred_hla_class_ii.tolist(), HLA_list=V8_HLA_ALLELES_WITH_HLACOCLUSTERS,
        )
        hla_index = hla_index_map[self.allele]

        index_X = np.asarray(
            [hla_counted_sample_names.index(ni) for ni in samples_to_cluster]
        )
        hla_counted_sample_names = list(np.array(hla_counted_sample_names)[index_X])
        X_count_matrix = X_count_matrix[index_X]

        return X_count_matrix, X_hla, hla_index, bioids, hla_counted_sample_names

    def hla_restrict_countmatrix_bioids(
        self,
        X_count_matrix: sparse.csr_matrix,
        X_hla: np.ndarray,
        bioids: List[str],
        samples: List[str]
    ) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
        """Restrict X_count_matrix to just the people with the HLA as specified
        in X_hla, and to just the bioids that those people have.

        Args:
            X_count_matrix (ndarray): Count matrix
            X_hla (ndarray): Matrix indicating who has the HLA of interest
            bioid_list (List[str]): list of bioids, in order

        Returns:
            Tuple[np.ndarray, np.ndarray]: HLA-restricted X_count_matrix,
                1d array containing just the bioids the people with the HLA have,
                and 1d array containing the sample names
        """
        bioids = np.asarray(bioids)  # type:ignore
        samples_to_cluster = np.asarray(samples)  # type:ignore
        sample_depths = X_count_matrix[:, [-1]].toarray()

        has_hla = (X_hla == 1).reshape((sample_depths.shape[0],))

        repertoires_have_hla = np.where(has_hla)[0]
        X_mayhave_noseq_reps = X_count_matrix[repertoires_have_hla, :bioids.shape[0]]
        samples_to_cluster = samples_to_cluster[repertoires_have_hla]

        logger.info(f"X_mayhave_noseq_reps: {X_count_matrix.shape}")

        has_any_topk_seqs = (X_mayhave_noseq_reps.sum(axis=1) > 0)
        if len(has_any_topk_seqs.shape) > 1:
            logger.info(f"len(has_any_topk_seqs.shape): {len(has_any_topk_seqs.shape)}")
            # this happens with sparse arrays, and it's a devil to deal with.
            # I'm sure there's a better way, but this is what I've found.
            has_any_topk_seqs = has_any_topk_seqs.tolist()
            if type(has_any_topk_seqs[0]) == list:
                has_any_topk_seqs = [x[0] for x in has_any_topk_seqs]
            has_any_topk_seqs = np.array(has_any_topk_seqs)

        X_count_matrix = X_mayhave_noseq_reps[has_any_topk_seqs, :]
        samples_to_cluster = samples_to_cluster[has_any_topk_seqs]
        logger.info(f"X_count_matrix.shape: {X_count_matrix.shape}")

        # Restrict to the bioids that occur in at least one sample
        # Without this check, clustering can fail.
        where_index_bioid_has_any_samples = np.where(X_count_matrix.sum(axis=0) > 0)  # can't use numpy's "any" here for sparse
        index_bioid_has_any_samples = where_index_bioid_has_any_samples[len(where_index_bioid_has_any_samples) - 1]

        bioids = bioids[index_bioid_has_any_samples]
        X_count_matrix = X_count_matrix[:, index_bioid_has_any_samples]

        return X_count_matrix, bioids, samples_to_cluster

    def cluster_hlacoclusters(
        self,
        X_count_matrix: sparse.csr_matrix,
        bioids: List[str],
    ) -> pd.DataFrame:
        logger.info(f"Running SVD with {self.n_svd_components} - 1 components")
        normalized_data, row_diag, col_diag = _scale_normalize(X_count_matrix)
        u, v = svd(normalized_data, self.n_svd_components, n_discard=1)
        z = np.vstack((row_diag[:, np.newaxis] * u, col_diag[:, np.newaxis] * v))

        logger.info("Running UMAP")
        embedding = umap.UMAP(n_components=self.n_umap_components).fit_transform(z)

        logger.info("Running HDBSCAN")
        labels = HDBSCAN(
            min_samples=self.hdbscan_min_samples,
            min_cluster_size=self.hdbscan_min_cluster_size,
        ).fit_predict(embedding)

        n_rows = X_count_matrix.shape[0]
        n_clusters = np.nanmax(labels) + 1
        logger.info(f"Found {n_clusters} clusters")

        logger.info(f"Dropping samples from clusters, keeping bioids where cluster has >= {self.min_tcrs_per_cluster} bioids")
        column_labels = labels[n_rows:]
        columns = np.vstack([column_labels == c for c in range(n_clusters)])

        bioid_clusters = []
        for i in range(columns.shape[0]):
            iii = np.where(columns[i, :])[0]
            if len(iii) >= self.min_tcrs_per_cluster:
                bioid_clusters.append([str(esi) for esi in bioids[iii]])

        logger.info("cluster_hlacoclusters() complete")
        return bioid_clusters

    def name_cluster(self, i: int) -> str:
        return f"h-{clean_allele(self.allele)}-{i}"

    def transform(self, request: SparkDatasetJobRequest) -> DataFrame:
        spark = df2spark(request.get("bioids"))

        logger.info("Calling get_counts...")
        X_count_matrix, X_hla, hla_index, bioids, hla_counted_sample_names = self.get_counts(request)

        logger.info("Calling get_counts...")
        X_count_matrix, bioids, hla_counted_sample_names = self.hla_restrict_countmatrix_bioids(
            X_count_matrix, X_hla[:, hla_index], bioids, hla_counted_sample_names)

        logger.info("Calling cluster_hlacoclusters...")
        bioid_clusters = self.cluster_hlacoclusters(
            X_count_matrix,
            bioids,
        )

        logger.info("Constructing output DataFrame...")
        cluster_names = [
            self.name_cluster(i)
            for i in range(len(bioid_clusters))
        ]

        clusters = spark.createDataFrame(
            pd.DataFrame.from_dict({  # type: ignore
                "hlacocluster": cluster_names,
                "hla": self.allele,
                "members": bioid_clusters,
            })
        )

        logger.info("Returning...")

        return clusters

def get_imputed_HLA_indicator_vector(
    class_i_list: List[str], class_ii_list: List[str], HLA_list: Union[None, List[str]]
) -> Tuple[np.ndarray, Dict[str, int]]:  # type:ignore[type-arg]
    """
    For a given list of imputed HLAs, generate an indicator vector for each sample

    :param class_i_list: A list of where each element is expected to be a comma separated string
            of all class I HLAs
    :param class_ii_list: A list of where each element is expected to be a comma separated string
            of all class I HLAs
    :param HLA_list: if you have a list of HLAs for which you want the indicator vector
    :return: X_hla is the indicator vector, hla_dict is the dictionary where each HLA is the key
            and the index in X_hla is the value
    """
    hla_dict = {hla_i: i for i, hla_i in enumerate(HLA_list)}
    HLA_indicator_out = []
    for ci, cii in zip(class_i_list, class_ii_list):
        hla_i = np.zeros(len(hla_dict), dtype=int)
        if (type(ci) == str) and (type(cii) == str):
            if len(ci) > 0:
                for hi in ci.split(","):
                    if hi in hla_dict:
                        hla_i[hla_dict[hi]] = 1
            if len(cii) > 0:
                for hii in cii.split(","):
                    if hii in hla_dict:
                        hla_i[hla_dict[hii]] = 1
        HLA_indicator_out.append(list(hla_i))

    return np.asarray(HLA_indicator_out), hla_dict