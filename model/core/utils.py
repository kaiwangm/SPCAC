import torch
import MinkowskiEngine as ME


def make_sparse_tensor(features, tensor_stride, dimension, device, coordinates=None, coordinate_manager=None, coordinate_map_key=None, clone=True):
    if coordinate_manager is None and coordinate_map_key is None:
        sparse_tensor = ME.SparseTensor(
            features=features.clone() if clone else features,
            coordinates=coordinates.clone() if clone and coordinates is not None else coordinates,
            tensor_stride=tensor_stride,
            device=device
        )
    else:
        sparse_tensor = ME.SparseTensor(
            features=features.clone() if clone else features,
            tensor_stride=tensor_stride,
            coordinate_manager=coordinate_manager,
            coordinate_map_key=coordinate_map_key,
            device=device
        )

    return sparse_tensor


def _rebuild_sparse_tensor(src, features):
    """Rebuild a sparse tensor with new features, sharing geometry with src."""
    return make_sparse_tensor(
        features=features,
        coordinate_manager=src.coordinate_manager,
        coordinate_map_key=src.coordinate_map_key,
        tensor_stride=src.tensor_stride,
        dimension=3,
        device=src.device,
    )


def apply_ent(entropy_bottleneck, y, *args, quantize=False):
    """Apply entropy bottleneck to sparse tensor y.

    When quantize=False (default): forward pass, returns (y_hat, likelihoods).
    When quantize=True: explicit quantization, returns y_hat only.
    """
    y_feats = y.F.unsqueeze(0).permute(0, 2, 1)
    if quantize:
        y_nos = entropy_bottleneck.quantize(y_feats, *args)
        y_nos = y_nos.squeeze(0).permute(1, 0)
        if args and args[0] == "symbols":
            y_nos = y_nos.float()
        return _rebuild_sparse_tensor(y, y_nos)
    else:
        y_nos, y_likelihoods = entropy_bottleneck(y_feats, *args)
        y_nos = y_nos.squeeze(0).permute(1, 0)
        y_likelihoods = y_likelihoods.squeeze(0).permute(1, 0)
        return _rebuild_sparse_tensor(y, y_nos), y_likelihoods


def apply_nos(entropy_bottleneck, y, *args):
    """Pre-quantize without computing likelihoods (backward-compat wrapper)."""
    return apply_ent(entropy_bottleneck, y, *args, quantize=True)


def apply_cmp(entropy_bottleneck, y, *args, **kwargs):
    y_feats = y.F.unsqueeze(0).permute(0, 2, 1)
    y_strings = entropy_bottleneck.compress(y_feats, *args, **kwargs)
    return [y_strings], y_feats.size()[-1:]


def apply_dcmp(entropy_bottleneck, strings, y, *args, **kwargs):
    y_nos = entropy_bottleneck.decompress(strings[0], *args, **kwargs)
    y_nos = y_nos.squeeze(0).permute(1, 0)
    return _rebuild_sparse_tensor(y, y_nos)


def make_spzeros_channel(x, channel, value=0.0):
    """Create a sparse tensor with constant feature values, sharing coordinates with x."""
    return make_sparse_tensor(
        coordinates=x.C,
        features=torch.full((x.F.shape[0], channel), value, device=x.device),
        coordinate_manager=x.coordinate_manager,
        coordinate_map_key=x.coordinate_map_key,
        tensor_stride=x.tensor_stride,
        dimension=3,
        device=x.device
    )


def make_spones_channel(x, channel):
    """Create a sparse tensor with all-ones features (backward-compat wrapper)."""
    return make_spzeros_channel(x, channel, value=1.0)


def mask_spzeros_numpt(x, num_pt, mode='Pass1'):
    """Mask zeros for 2-pass model (kept for backward compatibility).
    For new code, prefer mask_spzeros_numpt_n."""
    num_gps = [num_pt, x.F.shape[0] - num_pt]
    pass_idx = int(mode.replace('Pass', '')) - 1
    return mask_spzeros_numpt_impl(x, num_gps, pass_idx)


def mask_spzeros_numpt_3(x, num_pt_1, num_pt_2, mode='Pass1'):
    """Mask zeros for 3-pass model (kept for backward compatibility).
    For new code, prefer mask_spzeros_numpt_n."""
    num_pt_3 = x.F.shape[0] - num_pt_1 - num_pt_2
    num_gps = [num_pt_1, num_pt_2, num_pt_3]
    pass_idx = int(mode.replace('Pass', '')) - 1
    return mask_spzeros_numpt_impl(x, num_gps, pass_idx)


def mask_spzeros_numpt_8(x, num_pt_1, num_pt_2, num_pt_3, num_pt_4, num_pt_5, num_pt_6, num_pt_7, mode='Pass1'):
    """Mask zeros for 8-pass model (kept for backward compatibility).
    For new code, prefer mask_spzeros_numpt_n."""
    num_pt_8 = x.F.shape[0] - num_pt_1 - num_pt_2 - num_pt_3 - num_pt_4 - num_pt_5 - num_pt_6 - num_pt_7
    num_gps = [num_pt_1, num_pt_2, num_pt_3, num_pt_4, num_pt_5, num_pt_6, num_pt_7, num_pt_8]
    pass_idx = int(mode.replace('Pass', '')) - 1
    return mask_spzeros_numpt_impl(x, num_gps, pass_idx)


def mask_spzeros_numpt_impl(x, num_gps, pass_idx):
    """Generic implementation: zero out features outside group `pass_idx`."""
    feats = x.F.clone()
    cumsum = 0
    for i, n in enumerate(num_gps):
        if i < pass_idx:
            feats[cumsum:cumsum + n, :] = 0.0
        elif i > pass_idx:
            feats[cumsum:cumsum + n, :] = 0.0
        cumsum += n
    return make_sparse_tensor(
        features=feats,
        coordinate_manager=x.coordinate_manager,
        coordinate_map_key=x.coordinate_map_key,
        tensor_stride=x.tensor_stride,
        dimension=3,
        device=x.device,
        clone=False,
    )


def mask_spzeros_numpt_n(x, num_gps, pass_idx):
    """Generic N-pass mask: keep features of group pass_idx, zero out the rest.

    Args:
        x: SparseTensor.
        num_gps: list of group sizes [num_gp_1, num_gp_2, ..., num_gp_N].
        pass_idx: 0-based index of the group to keep.
    """
    return mask_spzeros_numpt_impl(x, num_gps, pass_idx)


def group_sp_inv(x, idx_pass1, idx_pass2):
    points = x.C.clone()
    feats = x.F.clone()

    num_gp_1 = idx_pass1.shape[0]
    num_gp_2 = idx_pass2.shape[0]

    points_ans = torch.zeros_like(points).to(points.device)
    feats_ans = torch.zeros_like(feats).to(feats.device)

    points_ans[idx_pass1, :] = points[:num_gp_1, :]
    points_ans[idx_pass2, :] = points[num_gp_1:, :]

    feats_ans[idx_pass1, :] = feats[:num_gp_1, :]
    feats_ans[idx_pass2, :] = feats[num_gp_1:, :]

    ans = make_sparse_tensor(
        coordinates=points_ans,
        features=feats_ans,
        tensor_stride=1,
        dimension=3,
        device=x.device
    )

    return ans


def _group_sp_impl(x, idx_list):
    """Generic N-group sparse point grouping.

    Args:
        x: SparseTensor.
        idx_list: list of index tensors, one per group, partitioning all points.

    Returns:
        (grouped_sparse_tensor, num_gp_1, num_gp_2, ...)
    """
    points = x.C[:, 1:].clone()
    feats = x.F.clone()

    # Concatenate in group order
    points_grouped = torch.cat([points[idx] for idx in idx_list], dim=0)
    feats_grouped = torch.cat([feats[idx] for idx in idx_list], dim=0)

    # Compute group sizes via unique div8 points
    num_pt = len(torch.unique(points_grouped // 8, dim=0))
    num_gps = [len(torch.unique(points[idx] // 8, dim=0)) for idx in idx_list[:-1]]
    num_gps.append(num_pt - sum(num_gps))

    # Add batch index column
    points_grouped = torch.cat([
        torch.zeros(points_grouped.shape[0], 1).int().to(x.device),
        points_grouped
    ], dim=1)

    x_grouped = create_new_sparse_tensor(
        coordinates=points_grouped, features=feats_grouped,
        tensor_stride=x.tensor_stride, dimension=3, device=x.device,
    )
    return tuple([x_grouped] + num_gps)


def group_sp(x, return_idx=False):
    """2-pass grouping: parity-based (pos_emb % 2)."""
    points = x.C[:, 1:].clone()
    pos_emb = torch.sum(points // 8, dim=1)

    idx_pass1 = torch.where(pos_emb % 2 == 0)[0]
    idx_pass2 = torch.where(pos_emb % 2 == 1)[0]

    result = _group_sp_impl(x, [idx_pass1, idx_pass2])
    if not return_idx:
        return result
    return result + (idx_pass1, idx_pass2)


def group_sp_3(x):
    """3-pass grouping: (all-even, even-sum-not-all-even, odd-sum)."""
    points = x.C[:, 1:].clone()
    points_div8 = points // 8
    pos_emb = torch.sum(points_div8, dim=1)

    p1_bools = torch.logical_and(torch.logical_and(
        points_div8[:, 0] % 2 == 0, points_div8[:, 1] % 2 == 0), points_div8[:, 2] % 2 == 0)
    idx_pass1 = torch.where(p1_bools)[0]
    idx_pass2 = torch.where(torch.logical_and(
        pos_emb % 2 == 0, torch.logical_not(p1_bools)))[0]
    idx_pass3 = torch.where(pos_emb % 2 == 1)[0]

    return _group_sp_impl(x, [idx_pass1, idx_pass2, idx_pass3])


def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long(), step.long()
    vector = sum([array[:, i]*(step**i) for i in range(array.shape[-1])])

    return vector


def create_new_sparse_tensor(coordinates, features, tensor_stride, dimension, device, coordinate_manager=None):
    return make_sparse_tensor(
        features=features, coordinates=coordinates,
        tensor_stride=tensor_stride, dimension=dimension, device=device,
        coordinate_manager=coordinate_manager, clone=False,
    )


def sort_sparse_tensor(sparse_tensor, get_indices=False, get_inverse=False, indices=None):
    """ Sort points in sparse tensor according to their coordinates.
    """
    if indices == None:
        indices = torch.argsort(
            array2vector(
                sparse_tensor.C,
                sparse_tensor.C.max()+1
            )
        )

    sparse_tensor = create_new_sparse_tensor(
        coordinates=sparse_tensor.C[indices],
        features=sparse_tensor.F[indices],
        tensor_stride=sparse_tensor.tensor_stride,
        dimension=sparse_tensor.D,
        device=sparse_tensor.device
    )
    if get_inverse == False:
        if get_indices == False:
            return sparse_tensor
        else:
            return sparse_tensor, indices
    else:
        ordered = torch.Tensor(range(sparse_tensor.C.shape[0])).to(
            sparse_tensor.device).int()
        ordered = ordered[indices]
        indices_inv = torch.argsort(ordered)

        if get_indices == False:
            return sparse_tensor, indices_inv
        else:
            return sparse_tensor, indices, indices_inv


def sort_coords_tensor(coords):
    """ Sort points in sparse tensor according to their coordinates.
    """
    coords_tmp = torch.cat(
        [
            torch.zeros([coords.shape[0], 1]).to(coords.device).int(),
            coords,
        ],
        dim=1,
    )
    indices = torch.argsort(
        array2vector(
            coords_tmp,
            coords_tmp.max()+1
        )
    )

    coords_sorted = coords[indices]

    return coords_sorted


def isin(data, ground_truth):
    """ Input data and ground_truth are torch tensor of shape [N, D].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is in `ground_truth` and False otherwise.
    """
    device = data.device
    if len(ground_truth) == 0:
        return torch.zeros([len(data)]).bool().to(device)
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = torch.isin(data.to(device), ground_truth.to(device))

    return mask
