# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport INFINITY, isnan
from libcpp cimport bool as bool_t
np.import_array()

ctypedef np.npy_int64 ITYPE_t

cdef class LinkageUnionFind:
    """Structure for fast cluster labeling in unsorted dendrogram."""
    cdef int[:] parent
    cdef int[:] size
    cdef int next_label

    def __init__(self, int n):
        self.parent = np.arange(2 * n - 1, dtype=np.intc)
        self.next_label = n
        self.size = np.ones(2 * n - 1, dtype=np.intc)

    cdef int merge(self, int x, int y) noexcept:
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        cdef int size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    cdef find(self, int x):
        cdef int p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x

cdef label(double[:, :] Z, int n):
    """Correctly label clusters in unsorted dendrogram."""
    cdef LinkageUnionFind uf = LinkageUnionFind(n)
    cdef int i, x, y, x_root, y_root

    for i in range(n - 1):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = uf.find(x), uf.find(y)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        Z[i, 3] = uf.merge(x_root, y_root)

cdef inline np.npy_int64 condensed_index(np.npy_int64 n, np.npy_int64 i,
                                         np.npy_int64 j) noexcept:
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return n * i - (i * (i + 1) / 2) + (j - i - 1)
    elif i > j:
        return n * j - (j * (j + 1) / 2) + (i - j - 1)

cdef double cross_dist_c_(const double[:, :] corr,
                           const ITYPE_t[:] idx1,
                           const ITYPE_t[:] idx2,
                           int size1, int size2) noexcept:
    """
    Pure C-level implementation of the custom distance.

    corr: full correlation matrix (n x n)
    idx: index list of cluster members
    size: number of members in idx
    """
    cdef:
        int i, j
        double s = 0.0
        double val
        long long count = 0

    for i in range(size1):
        for j in range(size2):
            val = corr[idx1[i], idx2[j]]
            if not isnan(val):
                s += val
                count += 1

    if count == 0:
        # all NaNs → nan_to_num(nanmean) → 0 → distance 1
        return 1.0

    return 1.0 - (s / count)

def nn_chain(const double[:] dists, int n, const double[:, :] corr):
    """Perform hierarchy clustering using nearest-neighbor chain algorithm.

    Parameters
    ----------
    dists : ndarray
        A condensed matrix stores the pairwise distances of the observations.
    n : int
        The number of observations.
    method : int
        The linkage method. 0: single 1: complete 2: average 3: centroid
        4: median 5: ward 6: weighted

    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    """
    Z_arr = np.empty((n - 1, 4))
    cdef double[:, :] Z = Z_arr

    cdef double[:] D = dists.copy()  # Distances between clusters.
    cdef int[:] size = np.ones(n, dtype=np.intc)  # Sizes of clusters.

    # Variables to store neighbors chain.
    cdef int[:] cluster_chain = np.ndarray(n, dtype=np.intc)
    cdef int chain_length = 0

    cdef int i, k, x, y = 0, nx, ny, ni
    cdef double dist, current_min

    cdef list members
    # each point starts as its own cluster
    members = [None] * n
    for i in range(n):
        members[i] = np.array([i], dtype=np.int64)

    for k in range(n - 1):
        if chain_length == 0:
            chain_length = 1
            for i in range(n):
                if size[i] > 0:
                    cluster_chain[0] = i
                    break

        # Go through chain of neighbors until two mutual neighbors are found.
        while True:
            x = cluster_chain[chain_length - 1]

            # We want to prefer the previous element in the chain as the
            # minimum, to avoid potentially going in cycles.
            if chain_length > 1:
                y = cluster_chain[chain_length - 2]
                current_min = D[condensed_index(n, x, y)]
            else:
                current_min = INFINITY

            for i in range(n):
                if size[i] == 0 or x == i:
                    continue

                dist = D[condensed_index(n, x, i)]
                if dist < current_min:
                    current_min = dist
                    y = i

            if chain_length > 1 and y == cluster_chain[chain_length - 2]:
                break

            cluster_chain[chain_length] = y
            chain_length += 1

        # Merge clusters x and y and pop them from stack.
        chain_length -= 2

        # This is a convention used in fastcluster.
        if x > y:
            x, y = y, x

        # get the original numbers of points in clusters x and y
        nx = size[x]
        ny = size[y]

        # Record the new node.
        Z[k, 0] = x
        Z[k, 1] = y
        Z[k, 2] = current_min
        Z[k, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster

        members[y] = np.concatenate((members[y], members[x]))

        # Update the distance matrix.
        for i in range(n):
            ni = size[i]
            if ni == 0 or i == y:
                continue

            D[condensed_index(n, i, y)] = cross_dist_c_(corr,
               members[i], members[y],
               ni, size[y])


    # Sort Z by cluster distances.
    order = np.argsort(Z_arr[:, 2], kind='mergesort')
    Z_arr = Z_arr[order]

    # Find correct cluster labels inplace.
    label(Z_arr, n)

    return Z_arr
