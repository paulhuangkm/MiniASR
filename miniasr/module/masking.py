'''
    File      [ masking.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Masking-related. ]
'''

import torch


def len_to_mask(lengths, max_length=-1, dtype=None):
    '''
        Converts lengths to a binary mask.
            lengths [long tensor]
        E.g.
            lengths = [5, 3, 1]
            mask = [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0]
            ]
    '''
    max_length = max_length if max_length > 0 else lengths.max().cpu().item()
    mask = (torch.arange(max_length, device=lengths.device, dtype=lengths.dtype)
            .expand(lengths.shape[0], max_length) < lengths.unsqueeze(1))
    if dtype is not None:
        mask = mask.type(dtype)
    return mask

def truncate_mask(dim, max_length=-1, window_size=100, dtype=None):
    max_length = max_length
    mask = torch.zeros(dim, max_length).to('cuda:0')
    # print(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(window_size + 1):
            l, r = max(i - j, 0), min(i + j, max_length - 1)
            mask[i][l] = mask[i][r] = 1
    if dtype is not None:
        mask = mask.type(dtype)
    return mask