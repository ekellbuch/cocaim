import sys
# sys.path.append('../')

import numpy as np
# import cocaim
from . import greedy, tool_grid, overlap_graph
import ipdb
import scipy as sp
import matplotlib.pyplot as plt

# add Sherman Morrison 

def discrete_diff_operator(L, order=2):
    """
    Returns discrete difference operator
    of order k+1
    where D(k+1) [n-k-1] x n
    Inputs:
    _______

    Outputs:
    ________
    """

    if order == 1:
        # assert L > 0
        D = (np.diag(np.ones(L-1)*1, 1) + np.diag(np.ones(L)*-1))[:(L-1), :]
    elif order == 2:
        # assert L > 1
        D = (np.diag(np.ones(L-1)*-2, 1) +
             np.diag(np.ones(L)*1) +
             np.diag(np.ones(L-2), 2))[:(L-2), :]
    return D


def extract_components(reject_off_all, reject_raw_count):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    number_tiles_off = len(reject_off_all)
    number_tiles_raw = reject_raw_count.shape[0]
    count_trim_raw_tiles = np.zeros((number_tiles_raw), )
    count_trim_off_tiles = np.zeros((number_tiles_off), )
    # Count trimmed off tiles

    for raw_tile in range(number_tiles_raw):
        count_trim_raw_tiles[raw_tile] = len(np.argwhere(
                                             reject_raw_count[raw_tile, :]))

        # Count trimmed on tiles
    for off_tile in range(number_tiles_off):
        if reject_off_all[off_tile] is None:
            continue
        count_trim_off_tiles[off_tile] = len(reject_off_all[off_tile])
    return count_trim_raw_tiles, count_trim_off_tiles


def flatten_spatial(spatial, nblocks, weighted=False):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    dim1, dim2 = nblocks
    spatials_out = []
    orders = ['F', 'F', 'F', 'F']
    for case in range(4):
        us = spatial[case]
        spatial_out = []
        for tile in range(len(us)):
            u1 = us[tile]
            d1, d2, d3 = u1.shape
            u1 = u1.reshape((d1*d2), d3, order=orders[case])
            if weighted:
                ak = tool_grid.pyramid_matrix((d1, d2))
                # corner cases only for outter most tile
                if False:  # case ==0:
                    if tile < dim2:
                        ak[0, :] = 1
                    if tile % dim2 == 0:
                        ak[:, 0] = 1
                    if tile >= (dim1-1)*dim2:
                        ak[-1, :] = 1
                    if (tile + 1) % dim2 == 0:
                        ak[:, -1] = 1
                u1 = u1*(ak.flatten(order=orders[case])[:, np.newaxis])
            spatial_out.append(u1)
        spatials_out.append(spatial_out)
    return spatials_out


def reorder_group(spatial_components,
                  temporal_components,
                  block_ranks,
                  block_indices,
                  nblocks,
                  weighted=False):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    keys_ = list(spatial_components.keys())
    off_ = [None, 'r', 'c', 'rc']
    spatial_all = []
    temporal_all = []
    ranks_all = []
    for case in range(4):
        U = spatial_components[keys_[case]]['full']
        V = temporal_components[keys_[case]]['full']
        K = block_ranks[keys_[case]]['full']
        indices = block_indices[keys_[case]]['full']

        # get tiles
        nblocks_case = nblock_offset(nblocks, off_[case])
        spatial_list, temporal_list = component_list(U, V, K, nblocks_case)

        # reorder offset
        offset = reorder_2Dcol(nblocks_case)

        # reorder components
        spatial, temporal = reorder_list(spatial_list,
                                         temporal_list,
                                         offset)
        # reorder ranks_
        ranks_ = K[offset]
        # append to results
        spatial_all.append(spatial)
        temporal_all.append(temporal)
        ranks_all.append(ranks_)

    spatials_out = flatten_spatial(spatial_all, nblocks, weighted=weighted)
    return spatials_out, temporal_all, ranks_all


def nblock_offset(nblocks, offset_case=None):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    d1, d2 = nblocks
    num_tiles_raw = np.prod(nblocks)
    if offset_case == 'r':
        num_tiles_offset = d1-1, d2  # np.prod((d1-1,d2))
    elif offset_case == 'c':
        num_tiles_offset = d1, d2-1  # np.prod((d1,d2-1))
    elif offset_case == 'rc':
        num_tiles_offset = d1-1, d2-1  # np.prod((d1-1,d2-1))
    else:
        num_tiles_offset = d1, d2
    return num_tiles_offset


def reorder_list(spatial_list, temporal_list, offset):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    spatial = []
    temporal = []
    for off in offset:
        spatial.append(spatial_list[off])
        temporal.append(temporal_list[off])
    return spatial, temporal


# reject components from raw offset
def reorder_2Dcol(nblocks):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    d1, d2 = nblocks

    i_row = 0
    i_col = 0
    offset = np.zeros((d1*d2), )
    for ii in range(d1*d2):
        if i_col == d2:
            i_row += 1
            i_col = 0
        offset[ii] = i_col*d1 + i_row
        i_col += 1
    return offset.astype('int')


def component_list(U, V, K, nblocks):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    d1, d2 = nblocks
    spatial_list = []  # [None]*n_tiles
    temporal_list = []  # [None]*n_tiles
    n_tiles = np.prod(nblocks)
    print(n_tiles)
    print(len(spatial_list))

    for bdx in range(n_tiles):
        spatial_list.append(U[bdx, :, :, :K[bdx]])
        temporal_list.append(V[bdx, :K[bdx], :])

    return spatial_list, temporal_list


def extract_rejected_components(spatial, temporal, rejection):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    num_tiles = len(spatial)
    spatial_outs = []
    temporal_outs = []
    n_components_extracted = []
    for tile in range(num_tiles):
        reject = rejection[tile]
        reject = np.setdiff1d(reject, [0, 1, 2, 3])
        spatial_tile = spatial[tile]
        temporal_tile = temporal[tile]

        if len(reject) == 0:
            spatial_outs.append(spatial_tile)
            temporal_outs.append(temporal_tile)
            n_components_extracted.append(0)
            continue

        n_components = len(temporal_tile)
        print(n_components)
        keep = np.setdiff1d(np.arange(n_components), reject)
        print(reject)
        print(keep)
        if n_components - len(reject) > 2:
            spatial_out = spatial_tile[:, keep]
            temporal_out = temporal_tile[keep, :]
        else:
            spatial_out = spatial_tile[:, 0][:, np.newaxis]
            temporal_out = temporal_tile[0, :][np.newaxis, :]
        n_components_extracted.append(len(temporal_tile) - len(temporal_out))
        spatial_outs.append(spatial_out)
        temporal_outs.append(temporal_out)

    return spatial_outs, temporal_outs, np.asarray(n_components_extracted)


def projection_matrix(X):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    # assume linearly independent rows
    return X.T.dot(np.linalg.inv(X.dot(X.T)))


def extract_neighbors(nblocks=[10, 10], offset_case='r'):
    """
    for a given offset
    output adjacency matrix which indicates
    which are the neighboring components in the original
    with offset_case is None matrix
    Adjacency matrix M1 (tile_idx_offset_movie x tile_idx_raw_movie)
    Inputs:
    _______

    Outputs:
    ________
    """
    d1, d2 = nblocks

    num_tiles_raw = np.prod(nblocks)
    if offset_case == 'r':
        num_tiles_offset = np.prod((d1-1, d2))
    elif offset_case == 'c':
        num_tiles_offset = np.prod((d1, d2-1))
    elif offset_case == 'rc':
        num_tiles_offset = np.prod((d1-1, d2-1))

    M1 = np.zeros(shape=(num_tiles_offset, num_tiles_raw))

    d1, d2 = nblocks
    if offset_case == 'r':
        for ii in range(num_tiles_offset):
            n1 = ii
            n2 = n1 + d2
            M1[ii, n1] = 1
            M1[ii, n2] = 1
    if offset_case == 'c':
        a = 0
        for ii in range(num_tiles_offset):
            if (ii + a) % d2 == 0:
                a += 1
            n1 = ii + a - 1
            n2 = n1 + 1
            M1[ii, n1] = 1
            M1[ii, n2] = 1

    if offset_case == 'rc':
        a = 0
        for ii in range(num_tiles_offset):
            if (ii + a) % d2 == 0:
                a += 1
            n1 = ii + a - 1
            n2 = n1 + 1
            n3 = d2 + ii + a - 1
            n4 = n3 + 1
            M1[ii, n1] = 1
            M1[ii, n2] = 1
            M1[ii, n3] = 1
            M1[ii, n4] = 1
    return M1


def neighbors_coldiag(nblocks):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    num_tiles_col = np.prod(nblock_offset(nblocks, offset_case='c'))
    num_tiles_diag = np.prod(nblock_offset(nblocks, offset_case='rc'))

    M1 = np.zeros(shape=(num_tiles_col, num_tiles_diag))
    d1, d2 = nblocks
    for ii in range(num_tiles_col):
        n1 = ii
        n2 = ii - d2 + 1
        if n2 < 0:
            n2 = ii
        # if in the last row skip
        if n1 >= num_tiles_diag:
            n1 = n2
        # print('%d, %d, %d'%(ii,n1,n2))
        M1[ii, [n1, n2]] = 1

    return M1


def combine_components_offsets(spatial_components,
                               temporal_components,
                               nblocks):
    """
    Given a list of temporal and spatial components
    # Given the known nblocks
    # we will have four lists of spatial
    and temporal components
    for each offset case
        for each tile in offset dmovie
            determine the neighboring tiles wrt to no offset denoised movie
            for each neighbor from original denoised movie
                determine if we can reject any of the components
    Inputs:
    _______

    Outputs:
    ________
    """
    num_tiles_raw = np.prod(nblocks)
    # There are three offset cases
    offset_cases = ['r', 'c', 'rc']

    # temporal_components_out = []
    # reject_components_raw = [None]*num_tiles_raw

    raw_out = []
    # Do we want main one to be the matrix wo offsets
    # For each offset case
    reject_offset_components = []
    reject_raw_components = []

    for step in range(offset_cases):
        # call function that inputs spatial_component 1
        # call function that inputs spatial_component 2
        # returns updated components for all of these

        # find neighbors wrt to raw
        raw_neighbors = extract_neighbors(nblocks, offset_cases[step])

        spatial_off, \
        temporal_off, \
        spatial_raw, \
        temporal_raw, \
        reject_off, \
        reject_raw = combine_components(spatial_components[0],
                                        temporal_components[0],
                                        spatial_components[step + 1],
                                        temporal_components[step + 1],
                                        raw_neighbors)

        # update raw?
        if False:
            spatial_components[0] = spatial_raw
            temporal_components[0] = temporal_raw

        # components at scale
        if True:
            spatial_components[step] = spatial_off
            temporal_components[step] = temporal_off

        reject_offset_components.append(reject_off)
        reject_raw_components.append(reject_raw)

    return spatial_components, temporal_components, \
           reject_offset_components, reject_raw_components


def combine_components(spatial_raw1,
                       temporal_raw1,
                       spatial_off,
                       temporal_off,
                       raw_neighbors,
                       nblocks,
                       block_height,
                       block_width):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    # Do we want main one to be the matrix wo offsets
    # Find neighbors wrt to raw

    number_tiles_off = len(spatial_off)
    number_tiles_raw = len(spatial_raw1)

    max_num_raw = max(list(map(len, temporal_raw1)))

    spatial_raw = list(spatial_raw1)
    temporal_raw = list(temporal_raw1)
    spatial_off_trun_all = list(spatial_off)
    temporal_off_trun_all = list(temporal_off)

    del temporal_off, spatial_off
    del temporal_raw1, spatial_raw1

    reject_raw_count = np.zeros((number_tiles_raw, max_num_raw))
    reject_raw_all = [None]*number_tiles_off
    reject_off_all = [None]*number_tiles_off

    for tile in range(number_tiles_off):
        # for tile in [0]:
        print('Running tile %d' % tile)

        temporal_tile = temporal_off_trun_all[tile]
        spatial_tile = spatial_off_trun_all[tile]

        if np.ndim(temporal_tile) == 1 or np.abs(temporal_tile).max() == 0:
            print('continued for tile %d' % tile)
            print(temporal_tile.shape[0])
            print(np.abs(temporal_tile).max())
            continue

        # Given neighbors wrt raw matrix from that tile
        neighbors_tile = np.nonzero(raw_neighbors[tile, :])[0]
        neighbors_comparison = []
        neighbors_rank = []

        # for each neighboring tile from original denoised movie
        Us = []
        Vs = []

        for neigh in neighbors_tile:
            # do not include rank 1 components
            if np.ndim(temporal_raw[neigh]) == 1:
                print('rank 1 component')
                print('continued for tile %d' % tile)
                continue
            else:
                # print('adding raw neigh %d'%neigh)
                # print(spatial_raw[neigh].shape)
                # print(temporal_raw[neigh].shape)
                Us.append(spatial_raw[neigh])
                Vs.append(temporal_raw[neigh])
                neighbors_comparison.append(neigh)
                neighbors_rank.append(temporal_raw[neigh].shape[0])

        if len(Us) == 0:
            print('continued for tile %d' % tile)
            print('No components?')
            continue

        Us = np.hstack(Us)
        Vs = np.vstack(Vs)

        # determine if we can reject any of the components
        # _, _ , \
        spatial_raw_trun, \
        temporal_raw_trun, \
        spatial_off_trun, \
        temporal_off_trun,\
        reject_raw, \
        reject_off = trim_neighboring_components(Us,
                                                 Vs,
                                                 spatial_tile,
                                                 temporal_tile,
                                                 neighbors_rank)
        # import pdb; pdb.set_trace()
        # need to split raw components

        reject_raw1 = np.asarray(reject_raw)
        # update raw components

        if len(reject_raw1) >= 1:
            if len(neighbors_rank) > 1:
                # import pdb; pdb.set_trace()
                spatial_rneigh = []
                temporal_rneigh = []
                offc = 0
                for rankidx in neighbors_rank:
                    spatial_rneigh.append(spatial_raw_trun[:, offc:offc +
                                                           rankidx])
                    temporal_rneigh.append(temporal_raw_trun[offc:offc +
                                                             rankidx, :])
                    offc += rankidx

            else:
                spatial_rneigh = [spatial_raw_trun]
                temporal_rneigh = [temporal_raw_trun]

            for ii, neigh in enumerate(neighbors_comparison):
                # print('Replacing raw %d'%neigh)
                # print(spatial_raw[neigh].shape)
                # print(spatial_rneigh[ii].shape)
                assert spatial_raw[neigh].shape == spatial_rneigh[ii].shape
                spatial_raw[neigh] = spatial_rneigh[ii]
                temporal_raw[neigh] = temporal_rneigh[ii]
                max_reject = neighbors_rank[ii]
                idx_ = reject_raw1 < max_reject
                reject_raw_count[neigh, reject_raw1[idx_]] = 1
                reject_raw1 = np.setdiff1d(reject_raw1,
                                           reject_raw1[idx_]) - max_reject
            reject_raw_all[tile] = reject_raw

        # update offset components
        if len(reject_off) >= 1:
            # print('Replacing off %d'%tile)
            # print(spatial_off_trun_all[tile].shape)
            # print(spatial_off_trun.shape)
            assert spatial_off_trun_all[tile].shape == spatial_off_trun.shape
            spatial_off_trun_all[tile] = spatial_off_trun
            temporal_off_trun_all[tile] = temporal_off_trun
            reject_off_all[tile] = reject_off

    # determine unique raw components to reject
    return spatial_raw, temporal_raw, spatial_off_trun_all, \
        temporal_off_trun_all, reject_raw_all, \
        reject_off_all, reject_raw_count


def component_test(ri, test='kurto', D=None):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    output = []
    sk_thres = 2
    if test == 'kurto':
        zs, ps = sp.stats.skewtest(ri)
        if np.abs(zs) <= sk_thres:
            output = [0]
    if test == 'l1tf':
        output = greedy.find_temporal_component(ri, D=D)
    return output


def trim_neighboring_components(U1, Vt1, U2, Vt2, neighbors_rank,
                                reject_raw=False,
                                verbose=False,
                                plot_en=False):
    """
    Given components from overlapping tiles
    # Eliminate redundant components
    # Step 1
    # Order the components in ascending order by the norm of U_i
    # (ie, smallest components first).
    # For each component, regress V_i onto the other components V_{\i}.
    #(Note that this should only involve the components j
    # such that U_j and U_i overlap.)
    # If the residual is temporally uncorrelated,
    # we conclude that V_i is redundant with V_{\i} and can be eliminated.
    # After eliminating V_i we need to distribute the signal
    # from this component (U_i * V_i regressed onto V_{\i})
    # to the remaining components.
    # To do this we take the regression weights a_k in the regression
    # V_i \approx \sum_k a_k V_k
    # and modify each U_k in the local sum above as
    # U_k = U_k + a_k U_i.
    Inputs:
    _______

    Outputs:
    ________
    """
    # Order the components in ascending order by the norm of U_i
    # (ie, smallest components first).
    rss_th = 0.05
    verbose = 0
    plot_en = 0
    if np.abs(Vt1).max() == 0 or np.abs(Vt2).max() == 0:
        # print('Dont')
        return U1, Vt1, U2, Vt2, [], []

    # Temporal components
    nraw, L = Vt1.shape
    noff = Vt2.shape[0]
    ntotal = nraw + noff

    # stack all components
    Vtb = np.vstack((Vt1, Vt2))

    # Calculate norms
    norms1 = np.sqrt(np.sum(Vtb**2, 1))
    S = np.diag(norms1)

    # Stack spatial components
    Ub = np.hstack((U1, U2))

    norms_ = np.sqrt(np.sum(Ub.dot(S)**2, 0))

    # replace by original components
    # order components by increasing norm
    sidx_ = np.argsort(norms_)

    # Determine all other components
    reject = []

    # reject all components with 0 norm
    reject_step = np.where(norms1 == 0)[0]
    if len(reject_step) > 0:
        for rej in reject_step:
            reject.append(rej)

    # discard 0 components
    sidx_ = diff_inplace(sidx_, reject)
    # print(norms_[sidx_])
    Vta = Vtb/norms1[:, np.newaxis]
    Ua = Ub.copy()

    D = greedy.difference_operator(L)

    reject_tgrid = 0
    reject_toff = 0

    # all components
    components = np.arange(ntotal)
    # form neighbors_rank
    ranks = np.cumsum(np.asarray(neighbors_rank))
    comp_segments = np.split(components, ranks)
    # comp_spatial = np.zeros(Ua[:,0].shape)

    len_rej0 = len(reject)
    upp_bound = max(1, min(min(neighbors_rank), noff) - 1)

    ignore = neighbors_rank
    ignore.append(0)
    ignore = np.cumsum(np.sort(ignore))
    if verbose:
        print(sidx_)
        print('reject0')
        print(reject)
        print('upper bound')
        print(upp_bound)
        print('nraw')
        print(nraw)
        print(neighbors_rank)
        print('noff')
        print(noff)
        print('ntotal')
        print(ntotal)
        print('ignore')
        print(ignore)
    RSS = np.zeros(ntotal)

    for row, idx in enumerate(sidx_):

        if idx in ignore:
            continue
        if len(reject) - len_rej0 >= upp_bound:
            print('reached max')
            break
        if idx in reject:
            continue
        # components to exclude
        idx_n = diff_inplace(components, [idx])
        # do not include rejected components
        idx_n = diff_inplace(idx_n, reject)

        rej = np.argwhere([idx in seg for seg in comp_segments])[0][0]

        idx_n = diff_inplace(idx_n, comp_segments[rej])

        vi = Vta[idx, :]
        vni = Vta[idx_n, :]
        Pvj = projection_matrix(vni)
        ai = vi.dot(Pvj)
        v_hat = ai.dot(vni)
        ri = vi - v_hat

        RSS[row] = np.sum(ri**2)
        if plot_en:

            plt.figure(figsize=(15, 10))
            plt.title('component %d' % idx)
            plt.plot(vi)
            plt.plot(v_hat)
            plt.show()

            plt.figure(figsize=(15, 10))
            plt.title('residual %d with RSS %.2f' % (idx, RSS[row]))
            plt.plot(ri)
            plt.show()

        # find components in residual
        residual_component = component_test(ri, D=D, test='l1tf')

        if len(residual_component) > 0 and RSS[row] >= rss_th:
            # print('Not rejected %d'%idx)
            continue
            # if reject_raw and reorder_[idx]<= nraw:
            # comes from original so do not include
            #    continue

        # Reject component if no signal in residual
        reject.append(idx)
        # redistribute the signal from component Ui*Vi
        # regressed onto V_\i to remaining components
        udist = Ua[:, idx]*ai[:, np.newaxis]
        S_comp = norms1[idx]
        S_other = norms1[idx_n]
        udist = udist*S_comp/S_other[:, np.newaxis]
        Ua[:, idx_n] = Ua[:, idx_n] + udist.T

        # empty out components
        Vta[idx, :] = 0
        Ua[:, idx] = 0

    if plot_en:
        plt.plot(np.arange(ntotal), RSS)
        plt.show()

    if len(reject) == 0:
        print('Did not throw away components')
        return U1, Vt1, U2, Vt2, [], []

    reject_all = np.asarray(reject)
    # reject off
    reject_toff = reject_all[reject_all >= nraw] - nraw
    # reject raw
    reject_tgrid = reject_all[reject_all < nraw]

    if verbose:
        print('reject')
        print(reject)
        print('reject_raw')
        print(reject_tgrid)
        print('reject off')
        print(reject_toff)

    if len(reject_step) > 0:
        for rej in reject_step:
            Vta[rej, :] = 0
            Ua[:, rej] = 0

    # renormalize
    Vta = Vta*norms1[:, np.newaxis]

    # split components according to original groups
    Vt1n, Vt2n = np.split(Vta, [nraw], axis=0)
    U1n, U2n = np.split(Ua, [nraw], axis=1)

    U1n = U1n.reshape(U1.shape)
    U2n = U2n.reshape(U2.shape)
    Vt1n = Vt1n.reshape(Vt1.shape)
    Vt2n = Vt2n.reshape(Vt2.shape)

    # for arr in [U1n,U2n,Vt1n,Vt2n]:
    #    print(arr.shape)

    return U1n, Vt1n, U2n, Vt2n, reject_tgrid, reject_toff


def diff_inplace(x, y):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    diff = set(x) - set(y)
    result = [o for o in x if o in diff]
    return result


def trim_component_group(Ustack, Vstack):
    """
    Inputs:
    _______

    Outputs:
    ________
    """
    num_components, L = Vstack.shape

    # Calculate norms
    norms1 = np.sqrt(np.sum(Vstack**2, 1))
    S = np.diag(norms1)

    # Normalize temporal component
    Vta = Vstack.copy()
    Vta[norms1 > 0] = Vstack[norms1 > 0] / norms1[norms1 > 0, np.newaxis]

    components = np.arange(num_components)

    # include norm and S to distribute
    norms_ = np.sqrt(np.sum(Ustack.dot(S)**2, 0))
    # norms_U = np.sqrt(np.sum(Ustack**2,0))

    # order components according to weighted norm
    sidx_ = np.argsort(norms_)

    RSS = np.zeros(num_components)
    D = greedy.difference_operator(L)

    plot_en = False
    # for corr
    rss_th = 0.05  # 0.1
    corr_th = 0.7  # 0.7
    # RSS = np.zeros(num_components)
    reject = []

    # exclude zeroed-out components
    exclude = np.where(norms1 == 0)[0]
    if np.any(exclude):
        sidx_ = np.setdiff1d(sidx_, exclude)
        components = np.setdiff1d(components, exclude)

    # set min num components
    if len(sidx_) <= 2:
        print('Less than 2 viable components')
        return Ustack, Vstack

    for row, idx in enumerate(sidx_):

        nidx = np.setdiff1d(components, idx)
        nidx = np.setdiff1d(nidx, reject)
        vi = Vta[idx, :]
        vni = Vta[nidx, :]
        Pvj = projection_matrix(vni)
        ai = vi.dot(Pvj)
        v_hat = ai.dot(vni)
        ri = vi - v_hat

        rss = np.sum(ri**2)
        # RSS[row] = rss

        # corr_metric = np.correlate(vi,v_hat)[0]
        # if corr_metric <= corr_th and rss >= rss_th:
        #    continue

        residual_component = component_test(ri, D=D, test='l1tf')
        if len(residual_component) > 0 and rss >= rss_th:
            continue

        # -- Reject component
        # Distribute spatial energy
        norm_idx = norms1[idx]
        # norm_other = norms1[nidx]
        # udist = (ai*norm_idx) # /norm_other)
        # Ustack[:,nidx] += (Ustack[:,idx][:,np.newaxis]*udist)
        for idx_nidx, cidx  in enumerate(nidx):
            # Ustack[:,idx_nidx] += Ustack[:,idx]*udist[idx_nidx]
            tmp = norm_idx*ai[idx_nidx]/norms1[cidx]
            Ustack[:, cidx] += Ustack[:, idx]*tmp

        # Delete from memory
        Ustack[:, idx] = 0
        Vstack[idx, :] = 0

        reject.append(idx)
    return Ustack, Vstack


def rx_graph_neighgroup(nblocks):
    """
    Inputs:
    _______

    Outputs:
    ________

    """
    nng2 = overlap_graph.rx_graph(nblocks)
    num_neigh_tiles = nng2.sum(0)
    friendly_tiles = np.argsort(num_neigh_tiles)[::-1]

    Cliques = []
    for ii, tile in enumerate(friendly_tiles):
        # print('For tile %d'%tile)
        neigh_tile_idx = np.nonzero(nng2[tile, :])[0]
        # neigh_run_idx[ii,tile]=1
        if len(neigh_tile_idx) == 0:
            continue
        Clique = np.sort(np.insert(neigh_tile_idx, 0, tile))
        Cliques.append(Clique)

    return Cliques
