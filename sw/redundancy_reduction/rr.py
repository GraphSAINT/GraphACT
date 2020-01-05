import scipy.sparse as sp
import scipy
import numpy as np
import argparse

from operator import itemgetter
import time


def parse_args():
    parser = argparse.ArgumentParser(description='arguments for redundancy reduction')
    parser.add_argument('--adj', type=str, required=True, help='the path to the adj file (scipy.csr_matrix stored as npz)')
    parser.add_argument('--round', type=int, required=True, help='total number of rounds to perform reduction')
    args = parser.parse_args()
    return args



def construct_ga(adj_gs):
    t1 = time.time()
    assert adj_gs.shape[0] == adj_gs.shape[1], "diagonal of subgraph should be 0"
    num_v = adj_gs.shape[0]
    weight_edges = dict()
    for v in range(num_v):
        neigh = np.sort(adj_gs.indices[adj_gs.indptr[v]:adj_gs.indptr[v+1]])
        #assert v not in neigh
        for iu,u in enumerate(neigh):
            for w in neigh[iu+1:]:
                if (u,w) not in weight_edges:
                    weight_edges[(u,w)] = 1
                else:
                    weight_edges[(u,w)] += 1
    return weight_edges, num_v


def obtain_precompute_edges(weight_edges,num_v):
    """
    operate on ga
    """
    M = []
    H = {k:v for k,v in weight_edges.items() if v>2}
    H_sorted = sorted(H.items(),key=itemgetter(1,0),reverse=True)
    S = np.ones(num_v)
    _W = 0
    for (u,v),weight in H_sorted:
        if not (S[u] and S[v]):
            continue
        _W += weight-1
        S[u] = 0; S[v] = 0
        M.append((u,v))
        if len(M) == int(num_v/2):
            break
    return M,_W


def obtain_compact_mat(adj_gs,M,feat):
    """
    obtain updated gs from M
    """
    ret_feat = np.zeros(feat.size+len(M))
    ret_feat[:feat.size] = feat
    idx = 0
    deg = np.ediff1d(adj_gs.indptr)
    num_v = deg.size
    # transpose adj first
    t1 = time.time()
    e_list = [[] for v in range(adj_gs.shape[0])]
    for v in range(adj_gs.shape[0]):
        n_list = adj_gs.indices[adj_gs.indptr[v]:adj_gs.indptr[v+1]]
        for n in n_list:
            e_list[n].append(v)
    e_list_full = []
    gs_t_indptr = np.zeros(adj_gs.shape[0]+1).astype(np.int32)       # indptr for adj_gs.T
    for i,el in enumerate(e_list):
        e_list_full.extend(sorted(el))
        gs_t_indptr[i+1] = gs_t_indptr[i] + len(el)
    gs_t_indices = np.array(e_list_full).astype(np.int32)            # indices for adj_gs.T
    # prepare I_edges here, after identifying the large-weight edges
    I_edges = dict()
    for (aggr1,aggr2) in M:
        # intersection of aggr1's neighbor and aggr2's neighbor
        _neigh1 = gs_t_indices[gs_t_indptr[aggr1]:gs_t_indptr[aggr1+1]]
        _neigh2 = gs_t_indices[gs_t_indptr[aggr2]:gs_t_indptr[aggr2+1]]
        I_edges[(aggr1,aggr2)] = np.intersect1d(_neigh1,_neigh2,assume_unique=True)
    for (aggr1,aggr2) in M:
        v_root = I_edges[(aggr1,aggr2)]
        ret_feat[num_v+idx] = ret_feat[aggr1]+ret_feat[aggr2]
        for v in v_root:
            neigh = adj_gs.indices[adj_gs.indptr[v]:adj_gs.indptr[v+1]]
            i1 = np.where(neigh==aggr1)[0][0]
            i2 = np.where(neigh==aggr2)[0][0]       # searchsorted not applicable here since we insert -1
            adj_gs.indices[adj_gs.indptr[v]+i1] = num_v+idx
            adj_gs.indices[adj_gs.indptr[v]+i2] = -1
            deg[v] -= 1
        idx += 1
    _indptr_new = np.cumsum(deg)
    indptr_new = np.zeros(num_v+idx+1)
    indptr_new[1:num_v+1] = _indptr_new
    indptr_new[num_v+1:] = _indptr_new[-1]
    indices_new = adj_gs.indices[np.where(adj_gs.indices>-1)]
    assert indices_new.size == indptr_new[-1]
    data_new = np.ones(indices_new.size)
    ret_adj = sp.csr_matrix((data_new,indices_new,indptr_new),shape=(num_v+len(M),num_v+len(M)))
    return ret_adj, ret_feat


f_tot_ops = lambda adj: adj.size-np.where(np.ediff1d(adj.indptr)>0)[0].size
f_tot_read = lambda adj: adj.size#-np.where(np.ediff1d(adj.indptr)==1)[0].size
max_deg = lambda adj: np.ediff1d(adj.indptr).max()
mean_deg = lambda adj: np.ediff1d(adj.indptr).mean()
sigma_deg2 = lambda adj: (np.ediff1d(adj.indptr)**2).sum()/adj.shape[0]


def main(adj, num_round):
    adj_gs = sp.load_npz(adj)
    num_v_orig = adj_gs.shape[0]
    tot_ops_orig = f_tot_ops(adj_gs)
    tot_read_orig = f_tot_read(adj_gs)
    feat = np.random.rand(adj_gs.shape[0])
    ground_truth = adj_gs@feat.reshape(-1,1)
    cnt_precompute = 0
    cnt_preread = 0
    for r in range(num_round):
        print("max deg: {}, avg deg: {:.2f}, (\Sigma deg^2)/|V|: {}".format(max_deg(adj_gs),mean_deg(adj_gs),sigma_deg2(adj_gs)))
        ops_prev = f_tot_ops(adj_gs)
        weight_edges,num_v = construct_ga(adj_gs)
        M,_W = obtain_precompute_edges(weight_edges,num_v)
        cnt_precompute += len(M)
        cnt_preread += 2*len(M)
        adj_gs,feat = obtain_compact_mat(adj_gs,M,feat)
        ops_new = f_tot_ops(adj_gs) + cnt_precompute
        read_new = f_tot_read(adj_gs) + cnt_preread
        print("previous ops: ", ops_prev)
        print("new ops: ", ops_new)
        print("match size: ",len(M))
        print("reduction comp compared to original: {:.2f} (precompute {:.3f} of original total ops, temp buffer {:.3f}% of |V|)"\
            .format(tot_ops_orig/ops_new,cnt_precompute/tot_ops_orig,cnt_precompute/num_v_orig*100))
        print("reduction comm compared to original: {:.2f}".format(tot_read_orig/read_new))
    optimized_result = adj_gs@feat.reshape(-1,1)
    np.testing.assert_allclose(ground_truth, optimized_result[:ground_truth.size], rtol=1e-8, atol=0)
    print("RESULT CORRECT!")


if __name__ == '__main__':
    args = parse_args()
    main(args.adj, args.round)
