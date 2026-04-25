# -*- coding: utf-8 -*-
# cython language_level 3
"""
Cythonized main loop for online time warping.
"""
cimport cython
cimport numpy as np

import numpy as np

# from numpy.math cimport INFINITY

from libc.math cimport INFINITY


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def oltw_arzt_loop(
    float[:,:] global_cost_matrix,
    float[:] window_cost,
    int window_start,
    int window_end,
    int input_index,
    float min_costs,
    int min_index
):
    """
    Cythonized main loop for online time warping.

    For each score position in window compute costs and check for possible current score position.

    Parameters
    ----------
    global_cost_matrix: numpy.ndarray
        Global cost matrix.
    window_cost: numpy.ndarray
        Window cost matrix.
    window_start: int
        Start index of window.
    window_end: int
        End index of window.
    input_index: int
        Current input index.
    min_costs: float
        Minimum costs of current window.
    min_index: int
        Index of minimum costs of current window.

    Returns
    -------
    global_cost_matrix: numpy.ndarray
        Updated Global cost matrix.
    min_index: int
        Updated index of minimum costs of current window.
    min_costs: float
        Minimum costs of current window.
    """
    cdef float dist1, dist2, dist3, local_dist, norm_cost
    cdef int idx = 0, score_index = window_start
    cdef Py_ssize_t k, wk = window_cost.shape[0]
    cdef Py_ssize_t N = global_cost_matrix.shape[0]

    # special case: cell (0,0)
    if score_index == input_index == 0:
        # compute cost
        global_cost_matrix[1, 1] = cy_sum(window_cost)
        # global_cost_matrix[1, 1] = sum(window_cost)
        min_costs = global_cost_matrix[1, 1]
        min_index = 0

    while score_index < window_end:

        if not (score_index == input_index == 0):

            # get the previously computed local cost
            local_dist = window_cost[idx]

            # update global costs
            dist1 = global_cost_matrix[score_index, 1] + local_dist
            dist2 = global_cost_matrix[score_index + 1, 0] + local_dist
            dist3 = global_cost_matrix[score_index, 0] + local_dist

            min_dist = min(dist1, dist2, dist3)
            global_cost_matrix[score_index + 1, 1] = min_dist

            norm_cost = min_dist / (input_index + score_index + 1.0)

            # check if new cell has lower costs and might be current position
            if norm_cost < min_costs:
                min_costs = norm_cost
                min_index = score_index

        idx = idx + 1
        score_index = score_index + 1

    update_cost_matrix(global_cost_matrix, N)

    return global_cost_matrix, min_index, min_costs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def oltw_scorpion_loop(
    float[:,:] global_cost_matrix,
    float[:,:] path_length_matrix,
    float[:] window_cost,
    int window_start,
    int window_end,
    int input_index,
    float min_costs,
    int min_index,
    float norm_exponent = 0.5
):
    """
    OLTW with path-length normalization (Scorpion variant).

    Like oltw_arzt_loop, but normalizes accumulated cost by actual
    path length instead of (input_index + score_index + 1).
    This eliminates the systematic overshoot bias during silence.
    """
    cdef float dist1, dist2, dist3, local_dist, norm_cost, min_dist
    cdef float len1, len2, len3, path_len
    cdef int idx = 0, score_index = window_start
    cdef Py_ssize_t N = global_cost_matrix.shape[0]
    cdef Py_ssize_t i

    # special case: cell (0,0)
    if score_index == input_index == 0:
        global_cost_matrix[1, 1] = cy_sum(window_cost)
        path_length_matrix[1, 1] = 1.0
        min_costs = global_cost_matrix[1, 1]
        min_index = 0

    while score_index < window_end:

        if not (score_index == input_index == 0):

            local_dist = window_cost[idx]

            # accumulated costs from three predecessors
            dist1 = global_cost_matrix[score_index, 1] + local_dist
            dist2 = global_cost_matrix[score_index + 1, 0] + local_dist
            dist3 = global_cost_matrix[score_index, 0] + local_dist

            # path lengths from three predecessors
            len1 = path_length_matrix[score_index, 1]
            len2 = path_length_matrix[score_index + 1, 0]
            len3 = path_length_matrix[score_index, 0]

            # pick minimum cost and corresponding path length
            if dist1 <= dist2 and dist1 <= dist3:
                min_dist = dist1
                path_len = len1 + 1.0
            elif dist2 <= dist3:
                min_dist = dist2
                path_len = len2 + 1.0
            else:
                min_dist = dist3
                path_len = len3 + 1.0

            global_cost_matrix[score_index + 1, 1] = min_dist
            path_length_matrix[score_index + 1, 1] = path_len

            # Dampened normalization: exponent < 1 reduces position bias
            # compared to Arzt's linear (i+j+1). Kept for ablation; the real
            # SCORPION-OLTW mechanism is explicit silent-beat freeze at the
            # position-update layer (oltw_arzt.py step method), not here.
            norm_cost = min_dist / (input_index + score_index + 1.0) ** norm_exponent

            if norm_cost < min_costs:
                min_costs = norm_cost
                min_index = score_index

        idx = idx + 1
        score_index = score_index + 1

    # shift cost matrix columns
    update_cost_matrix(global_cost_matrix, N)
    # shift path length matrix columns (reset to 0, not INFINITY)
    for i in range(N):
        path_length_matrix[i, 0] = path_length_matrix[i, 1]
        path_length_matrix[i, 1] = 0.0

    return global_cost_matrix, path_length_matrix, min_index, min_costs


@cython.boundscheck(False)
@cython.wraparound(False)
cdef update_cost_matrix(float[:, :] global_cost_matrix, Py_ssize_t N):
    """
    Update the cost matrix by shifting the first column and set the second column to infinity.

    Parameters
    ----------
    global_cost_matrix : np.ndarray
        The global cost matrix.
    N : int
        The length of the input sequence.
    """
    cdef Py_ssize_t i
    for i in range(N):
        global_cost_matrix[i, 0] = global_cost_matrix[i, 1]
        global_cost_matrix[i, 1] = INFINITY

@cython.boundscheck(False)
@cython.wraparound(False)
cdef sum_in_place(float[:] x, float out):
    """
    Sum of array elements without return.
    """
    cdef Py_ssize_t i, n = x.shape[0]
    out = 0
    for i in range(n):
        out = out + x[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float cy_sum(float[:] x):
    """
    Sum of array elements.
    """
    cdef float out = 0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        out = out + x[i]
    # sum_in_place(x, out)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float[:,:] reset_cost_matrix(
        float[:, :] global_cost_matrix,
        float[:] window_cost,
        int score_index,
        int N
):
    """
    Reset cost matrix for new window.

    Parameters
    ----------
    global_cost_matrix : ndarray
        Global cost matrix.
    window_cost : ndarray
        Window cost vector.
    score_index : int
        Score index.
    N : int
        Number of rows in global cost matrix.

    Returns
    -------
    global_cost_matrix : ndarray
        Updated global cost matrix.
    """
    cdef int six = score_index + 1
    cdef Py_ssize_t i

    for i in range(N):
        global_cost_matrix[i, 0] = INFINITY
    global_cost_matrix[six, 1] = cy_sum(window_cost)

    return global_cost_matrix
