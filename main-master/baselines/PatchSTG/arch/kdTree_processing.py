import numpy as np
import pickle
import pandas as pd

def read_meta(path):
    meta = pd.read_csv(path)
    lat = meta['Lat'].values
    lng = meta['Lng'].values
    locations = np.stack([lat,lng], 0)
    return locations

def augmentAlign(dist_matrix, auglen):
    # find the most similar points in other leaf nodes
    sorted_idx = np.argsort(dist_matrix.reshape(-1)*-1)
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(idx)
        if len(augidx) == auglen:
            break
    return np.array(augidx, dtype=int)

def kdTree(locations, times, axis):
    # locations: [2,N] contains lng and lat
    # times: depth of kdtree
    # axis: select lng or lat as hyperplane to split points
    sorted_idx = np.argsort(locations[axis])
    part1, part2 = np.sort(sorted_idx[:locations.shape[1]//2]), np.sort(sorted_idx[locations.shape[1]//2:])
    parts = []
    if times == 1:
        return [part1, part2], max(part1.shape[0], part2.shape[0])
    else:
        left_parts, lmxlen = kdTree(locations[:,part1], times-1, axis^1)
        right_parts, rmxlen = kdTree(locations[:,part2], times-1, axis^1)
        for part in left_parts:
            parts.append(part1[part])
        for part in right_parts:
            parts.append(part2[part])
    return parts, max(lmxlen, rmxlen)

# def reorderData(metapath, adjpath, recurtimes, sps):

#     locations = read_meta(metapath)
#     adj = np.load(adjpath)
#     # partition and pad data with new indices
#     parts_idx, _ = kdTree(locations, recurtimes, 0)
    
#     # parts_idx: segmented indices by kdtree
#     # adj: pad similar points through the cos_sim adj
#     # sps: spatial patch (small leaf nodes) size for padding
#     ori_parts_idx = np.array([], dtype=int)
#     reo_parts_idx = np.array([], dtype=int)
#     reo_all_idx = np.array([], dtype=int)
#     for i, part_idx in enumerate(parts_idx):
#         part_dist = adj[part_idx, :].copy()
#         part_dist[:, part_idx] = 0
#         if sps-part_idx.shape[0] > 0:
#             local_part_idx = augmentAlign(part_dist, sps-part_idx.shape[0])
#             auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
#         else:
#             auged_part_idx = part_idx

#         reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0])+sps*i])
#         ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
#         reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])

#     return ori_parts_idx, reo_parts_idx, reo_all_idx


def reorderData(metapath, adjpath, recurtimes, sps):
    locations = read_meta(metapath)

    # 修改为读取 pkl 文件
    with open(adjpath, 'rb') as f:
        adj = pickle.load(f)  # 使用 pickle 读取 .pkl 文件

    # partition and pad data with new indices
    parts_idx, _ = kdTree(locations, recurtimes, 0)
    
    # parts_idx: segmented indices by kdtree
    # adj: pad similar points through the cos_sim adj
    # sps: spatial patch (small leaf nodes) size for padding
    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        if sps - part_idx.shape[0] > 0:
            local_part_idx = augmentAlign(part_dist, sps - part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx

        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0]) + sps * i])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])

    return ori_parts_idx, reo_parts_idx, reo_all_idx

# metapath= 'data/SD/meta.csv'
# adjpath = 'data/SD/adj.npy'
# recurtimes = 5
# sps = 2

# ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(metapath, adjpath, sps)