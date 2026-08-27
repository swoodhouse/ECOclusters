def report_clusters_entiretree(
    pdf_tacc_node_connectivity: pd.DataFrame,
    tacc_names_in_order: ty.List[str],
    pdf_tacc_tcrs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Walk up the agglomerativeclustering tree and report the clusters at every
    node.

    Args:
        pdf_tacc_node_connectivity (pd.DataFrame): dataframe describing the clustering tree
        pdf_tacc_tcrs:
        tacc_names_in_order (List(str)): list of tacc names in order

    Returns:
        pd.DataFrame: clusters all through the tree
    """

    cluster_distance_threshold = None

    # combine all the child nodes together and figure out which ones are bioids and samples
    n_leaf_nodes = len(tacc_names_in_order)

    # retain the full connectivity dataframe in _all
    pdf_tacc_node_connectivity_all = pdf_tacc_node_connectivity

    # count members in each cluster
    # cluster_nbioids_map = dict(zip(*[pdf_tacc_tcrs.cluster, pdf_tacc_tcrs.n_bioids]))
    col_to_count = "tacc" if "tacc" in pdf_tacc_tcrs.columns else "cluster"
    cluster_membercounts = pdf_tacc_tcrs[col_to_count].value_counts()
    cluster_membercount_map = dict(zip(*[cluster_membercounts.index, cluster_membercounts]))

    # initialize the lists that will become result columns.
    # seed with n_leaf_node dummy rows so that indexes match up
    clusters_at_idxs = [[tacc_name] for tacc_name in tacc_names_in_order]
    bioid_counts = [cluster_membercount_map[cluster] for cluster in tacc_names_in_order]
    distances = [0] * n_leaf_nodes
    child_1_idxs = [-1] * n_leaf_nodes
    child_2_idxs = [-1] * n_leaf_nodes

    for _, row in pdf_tacc_node_connectivity.iterrows():
        # lists holding all the _1 and _2 children, respectively,
        # because I write those out
        distance = row["distance"]
        if cluster_distance_threshold is not None and distance > cluster_distance_threshold:
            break
        cluster_tacc_lists_bothcols = []
        n_bioids = 0
        for col in ["child_1", "child_2"]:
            cluster_idx = int(row[col])
            if cluster_idx > len(clusters_at_idxs) - 1:
                logger.warning(
                    f"Fail! cluster_idx={cluster_idx}, clusters: {len(clusters_at_idxs)}"
                )
            this_bioid_list = clusters_at_idxs[cluster_idx]
            cluster_tacc_lists_bothcols.append(this_bioid_list)
            if col == "child_1":
                child_1_idxs.append(cluster_idx)
            else:
                child_2_idxs.append(cluster_idx)
            n_bioids += bioid_counts[cluster_idx]

        cluster_child1_taccs, cluster_child2_taccs = cluster_tacc_lists_bothcols

        # hold all the taccs in this cluster
        this_cluster_taccs = cluster_child1_taccs + cluster_child2_taccs

        bioid_counts.append(n_bioids)
        distances.append(distance)
        clusters_at_idxs.append(this_cluster_taccs)

    # make the clusters dataframe with everything
    pdf_clusters_distances = pd.DataFrame(
        {
            "cluster_members": clusters_at_idxs,
            "n_taccs": [len(x) for x in clusters_at_idxs],
            "n_bioids": bioid_counts,
            "distance": distances,
            "tree_idx": range(len(clusters_at_idxs)),
            "child_1_idx": child_1_idxs,
            "child_2_idx": child_2_idxs,
        }
    )

    # throw out everything over the distance threshold
    if cluster_distance_threshold is not None:
        pdf_clusters_distances = pdf_clusters_distances[
            pdf_clusters_distances.distance < cluster_distance_threshold
        ]
        logger.debug(
            f"    distance threshold: {cluster_distance_threshold}, tree size: {len(pdf_clusters_distances)}"
        )

    # Calculate all distances as percentiles of the full distance distribution
    pdf_clusters_distances["distance_percentile"] = percentileofscore(
        pdf_tacc_node_connectivity_all.distance, pdf_clusters_distances.distance
    )

    return pdf_clusters_distances


class EcoclusterConstructor:
    def __init__(
        self,
        min_members: int = DEFAULT_MIN_MEMBERS,
        max_members: int = DEFAULT_MAX_MEMBERS,
        max_distance: ty.Optional[float] = None,
        max_mean_taccs_per_hla: float = DEFAULT_MAX_MEAN_TACCS_PER_HLA_FORCREATE,
    ) -> None:
        """
        Args:
            min_members (int, optional): Minimum TACCs to combine. Defaults to DEFAULT_MIN_MEMBERS.
            max_members (int, optional): Maximum TACCs to combine. Defaults to DEFAULT_MAX_MEMBERS.
            max_distance (float, optional): Maximum distance in the tree to combine. Defaults to None.
            max_mean_taccs_per_hla (float, optional): Maximum mean TACCs per HLA. Defaults to DEFAULT_MAX_MEAN_TACCS_PER_HLA.
        """
        self.min_members = min_members
        self.max_members = max_members
        self.max_distance = max_distance
        self.max_mean_taccs_per_hla = max_mean_taccs_per_hla

    def cluster_ecoclusters(
        self,
        pdf_node_connectivity: pd.DataFrame,
        tacc_names_in_order: ty.List[str],
        pdf_tacc_tcrs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construct ECOClusters, in a cross-HLA clustering-of-clusters setting. Steps:

        1. Walk up the tree accumulating clusters
        2. Walk down the tree in descending-sorted distance order. Return the first cluster
        encountered, for each TCR, that satisfies min_members (minimum number of TACCs
        subsumed), max_members (ditto but maximum) max_distance (maximum distance) and
        max_mean_taccs_per_hla (maximum mean # TACCs per HLA).
        3. Annotate each cluster with its set of HLAs

        This class is capable of splitting ECOclusters at the first level in the tree at which
        the mean number of HLAs per TACC exceeds a threshold, but by default this is set to an
        impossibly high number so that no splitting is done.

        Args:
            pdf_node_connectivity (PandasDataFrame): dataframe containing the AgglomerativeClustering result
            tacc_names_in_order: list of the TACC names in index order as referred to in above arg
            pdf_tacc_tcrs: TACC dataframe with one row per bioIdentity+TACC
        Raises:
            ValueError: raised if the clustering fails because the tree wasn't arranged properly.

        Returns:
            PandasDataFrame: The defined cross-HLA clustering. Similar to the within-HLA
                clustering, but no "hla" column
        """
        pdf_cluster_tree = report_clusters_entiretree(
            pdf_node_connectivity, tacc_names_in_order, pdf_tacc_tcrs
        )

        all_members = set().union(*[set(x) for x in pdf_cluster_tree.cluster_members])
        pdf_cluster_tree["order"] = range(len(pdf_cluster_tree))
        pdf_cluster_tree_distance_desc = pdf_cluster_tree.sort_values("order", ascending=False)

        if self.max_distance is not None:
            pdf_cluster_tree_distance_desc = pdf_cluster_tree_distance_desc[
                pdf_cluster_tree_distance_desc.distance <= self.max_distance
            ]

        taccs_in_clusters: ty.Set[str] = set()

        # make for easy lookup of TACC HLAs
        pdf_tacc_hlas = pdf_tacc_tcrs.drop_duplicates("cluster")
        tacc_hla_map = dict(zip(*[pdf_tacc_hlas.cluster, pdf_tacc_hlas.hla]))

        def lookup_tacclist_hlas(tacc_list: ty.List[str]) -> ty.Set[str]:
            return set([tacc_hla_map[tacc] for tacc in tacc_list])

        pdf_cluster_tree_distance_desc["n_taccs"] = (
            pdf_cluster_tree_distance_desc.cluster_members.apply(len)
        )

        # Calculate mean HLAs per TACC at each node
        pdf_cluster_tree_distance_desc["hlas"] = (
            pdf_cluster_tree_distance_desc.cluster_members.apply(lookup_tacclist_hlas)
        )
        pdf_cluster_tree_distance_desc["n_hlas"] = pdf_cluster_tree_distance_desc.hlas.apply(len)
        pdf_cluster_tree_distance_desc["mean_hlas_per_tacc"] = (
            pdf_cluster_tree_distance_desc.n_taccs / pdf_cluster_tree_distance_desc.n_hlas
        )

        # filter the tree on distance and mean HLAs per TACC
        pdf_cluster_tree_distance_desc = pdf_cluster_tree_distance_desc[
            (pdf_cluster_tree_distance_desc.mean_hlas_per_tacc <= self.max_mean_taccs_per_hla)
            & (pdf_cluster_tree_distance_desc.n_taccs <= self.max_members)
            & (pdf_cluster_tree_distance_desc.n_taccs >= self.min_members)
        ]

        cluster_tacc_members = []
        cluster_distances = []
        for _, row in pdf_cluster_tree_distance_desc.iterrows():
            cluster_taccs_set = set(row["cluster_members"])
            clustertaccs_inters_already = cluster_taccs_set.intersection(taccs_in_clusters)
            if len(clustertaccs_inters_already) == 0:
                # no overlap with existing clusters
                # new cluster, add it
                taccs_in_clusters.update(cluster_taccs_set)
                cluster_tacc_members.append(row["cluster_members"])
                cluster_distances.append(row["distance"])
            else:
                # overlap with existing clusters. It had better be the whole thing.
                if not len(clustertaccs_inters_already) == len(cluster_taccs_set):
                    # we should never see a cluster with some members already accounted for and others not
                    raise ValueError(
                        f"ERROR! cluster taccs already clustered: {clustertaccs_inters_already}, "
                        + "total in set: {cluster_taccs_set}"
                    )
        logger.debug(
            f"    Clusters: {len(cluster_tacc_members)}. Retained {len(taccs_in_clusters)} "
            + f"of {len(all_members)} TACCs"
        )

        # I originally added an array of all component-cluster HLAs as an "hla" column here.
        # But that drove spark insane, made the dataset impossible to work with, I don't
        # know why.

        pdf_ecoclustering = pd.DataFrame(
            {
                "cluster": [f"cluster_{i}" for i in range(len(cluster_tacc_members))],
                "members": cluster_tacc_members,
                "ASR_in_cluster": [0] * len(cluster_tacc_members),  # dummy value
                "cluster_distance": cluster_distances,
            }
        )
        pdf_ecoclustering["n_members"] = [len(x) for x in pdf_ecoclustering.members]
        pdf_ecoclustering["member_frequencies"] = [
            [0] * n for n in pdf_ecoclustering.n_members
        ]  # dummy values

        # reorder cluster tight to loose, just for fun
        if len(pdf_ecoclustering) > 0:
            pdf_ecoclustering = pdf_ecoclustering.sort_values("cluster_distance", ascending=False)
        pdf_ecocluster_tcrs = self.reformat_ecoclusters_one_row_per_tcr(
            pdf_ecoclustering, pdf_tacc_tcrs
        )
        return pdf_ecocluster_tcrs

    def reformat_ecoclusters_one_row_per_tcr(
        self, pdf_ecoclustering: pd.DataFrame, pdf_tacc_tcrs: pd.DataFrame
    ) -> pd.DataFrame:
        pdf_tacc_ecocluster_map = pdf_ecoclustering.explode("members")[  # type: ignore
            ["cluster", "members"]
        ].rename(columns={"cluster": "ecocluster", "members": "cluster"})
        pdf_ecocluster_tcrs = pdf_tacc_tcrs.merge(pdf_tacc_ecocluster_map, on="cluster")
        # reorder columns, putting ecocluster first
        pdf_ecocluster_tcrs = pdf_ecocluster_tcrs[
            ["ecocluster"] + [c for c in pdf_ecocluster_tcrs.columns if c != "ecocluster"]
        ]
        return pdf_ecocluster_tcrs