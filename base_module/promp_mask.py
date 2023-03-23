import torch
import random
import matplotlib.pyplot as plt
import numpy as np

def create_future_mask(o, r=.15, sync=False):
    if r <= 0: 
      return torch.zeros_like(o).bool()
    if o.ndim == 2: 
      o = o[None]
    n_masks, mask_dims, mask_len = o.shape
    if sync == 'random': 
      sync = random.random() > .5
    dims = 1 if sync else mask_dims
    probs = torch.tensor(r, device=o.device)
    mask = torch.binomial(torch.tensor(1), probs).sample((n_masks, dims, mask_len))
    if sync: 
      mask = mask.repeat(1, mask_dims, 1)
    mask = torch.sort(mask,dim=-1, descending=False)[0].bool()
    
    return mask

def future_mask_prompt(len, lm, masking_ratio, i_len, f_len, prompt_len, mode="Prompt"):

    total_mask = np.ones(len, dtype=bool)
    total_mask[i_len : f_len] = np.zeros(f_len - i_len, dtype=bool)

    # Prompt_mask
    p_m = 1 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]

    # start
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(prompt_len):
        total_mask[f_len + i] = state
        if np.random.rand() < p[state]:
            state = 1 - state

    if mode == "Prompt":
        prompt_mask = np.ones(len, dtype=bool)
        prompt_mask[i_len : f_len + prompt_len] = np.zeros(f_len - i_len + prompt_len, dtype=bool)
    
    return total_mask, prompt_mask

def prompt_learning_mask(X, masking_ratio, lm, i_len, f_len, prompt_len, mode="Prompt"):
    X = X.squeeze()
    mask = np.ones(X.shape, dtype=bool)

    for m in range(X.shape[1]):
        # feature dimensional
        if m < X.shape[1] - 1:
            mask[: , m] = future_mask_prompt(X.shape[0], lm, masking_ratio, i_len, f_len, prompt_len, mode="Prompt")[0]
        else:
            mask[: , m] = future_mask_prompt(X.shape[0], lm, masking_ratio, i_len, f_len, prompt_len, mode="Prompt")[1]

    return mask

from torch.utils.data import Dataset

class Imputation_Prompting_Dataset(Dataset):
    def __init__(self, data, indices, masking_ratio, mean_mask_length, input_len, forcasting_len, prompt_len, mode="Prompt"):
        super(Imputation_Prompting_Dataset, self).__init__()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.feature_df = self.data[self.IDs]

        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.mode = mode
        self.input_len = input_len
        self.forcasting_len = forcasting_len
        self.prompt_len = prompt_len

    def __getitem__(self, ind):

        X = self.feature_df[self.IDs[ind]]  # (seq_length, feat_dim) array
        X = np.squeeze(X, axis=0)
        mask = prompt_learning_mask(X, self.masking_ratio, self.mean_mask_length, 
                                    self.input_len, self.forcasting_len, self.prompt_len, self.mode)  
                                    # (seq_length, feat_dim) boolean array
        
        return torch.from_numpy(X), torch.from_numpy(mask), self.IDs[ind]

    def update(self):
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        return len(self.IDs)

"""
Interaction Prompting Learning Dataset Preprocessing
"""


def interaction_prompt(len, lm, masking_ratio, i_len, f_len, prompt_len, mode=None):
    total_mask = np.ones(len, dtype=bool)
    inter_prompt = np.ones(len, dtype=bool)
    if mode == "Data_Preparing":
        total_mask[i_len : f_len + prompt_len] = np.zeros(f_len + prompt_len - i_len, dtype=bool)
    elif mode == "Data_Interaction":
        total_mask[:len] = np.zeros(len, dtype=bool)
    elif mode == "Interaction":
        p_m = 1 / lm
        p_u = p_m * masking_ratio / (1 - masking_ratio)
        p = [p_m, p_u]

        state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
        for i in range(len):
            inter_prompt[i] = state
            if np.random.rand() < p[state]:
                state = 1 - state
    
    return total_mask, inter_prompt
        

def inter_prompt_learning_mask(X, masking_ratio, lm, i_len, f_len, prompt_len, mode=None):
    X = X.squeeze()
    mask = np.ones(X.shape, dtype=bool)

    if mode == "Data_Preparing":
        for m in range(X.shape[1]):
            mask[:, m] = interaction_prompt(X.shape[0], lm, masking_ratio, i_len, f_len, prompt_len, mode="Data_Preparing")[0]

    elif mode == "Interaction" or mode == "Data_Interaction":
        for m in range(X.shape[1]):
            if m < X.shape[1] - 1:
                mask[: , m] = interaction_prompt(X.shape[0], lm, masking_ratio, i_len, f_len, prompt_len, mode="Interaction")[1]
            elif m == X.shape[1] - 1:
                mask[: , m] = interaction_prompt(X.shape[0], lm, masking_ratio, i_len, f_len, prompt_len, mode="Data_Interaction")[0]
    
    elif mode == 'sigl_prompt':
         mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
        #  mask[:, -1] = np.zeros(X.shape[0], dtype=bool)

    elif mode == "bernousep_prompt":
        mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        mask[:, -1] = np.zeros(X.shape[0], dtype=bool)

    elif mode == "bernou_prompt":
        mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])
        mask[:, -1] = np.zeros(X.shape[0], dtype=bool)

    elif mode == "horzion_prompt":
        whole = X.shape[1]
        masked_num = int(whole * (1 - masking_ratio))
        masked_selected = np.random.choice(whole, masked_num, replace=False)
        # print(masked_selected)
        for k in masked_selected:
            mask[:, k] = np.zeros(X.shape[0], dtype=bool)
            
    return mask

def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state
            
    return keep_mask

def iterable_prompt_mask(X, masking_ratio, lm, feature_idx, mode=None):
    X = X.squeeze()
    mask = np.ones(X.shape, dtype=bool)
    
    mask[:, feature_idx] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)
    for i in range(feature_idx + 1, X.shape[1]):
        mask[:, i] = np.zeros(X.shape[0], dtype=bool)

    return mask

class Imputation_Inter_Prompting_Dataset(Dataset):
    def __init__(self, data, indices, masking_ratio, mean_mask_length, input_len, forcasting_len, prompt_len, mode=None):
        super(Imputation_Inter_Prompting_Dataset, self).__init__()

        self.data = data
        self.IDs = indices
        self.feature_df = self.data[self.IDs]
        self.mode = mode
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.input_len = input_len
        self.forcasting_len = forcasting_len
        self.prompt_len = prompt_len

    def __getitem__(self, ind):

        X = self.feature_df[self.IDs[ind]]  # (seq_length, feat_dim) array
        X = np.squeeze(X, axis=0)
        mask = inter_prompt_learning_mask(X, self.masking_ratio, self.mean_mask_length, 
                                    self.input_len, self.forcasting_len, self.prompt_len, self.mode)  
                                    # (seq_length, feat_dim) boolean array
        
        return torch.from_numpy(X), torch.from_numpy(mask), self.IDs[ind]

    def update(self):
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        return len(self.IDs)


if __name__ == "__main__":
    X = torch.randn(12, 12)
    a = inter_prompt_learning_mask(X, masking_ratio=0.35, lm=4, i_len=12, f_len=12, prompt_len=4, mode="bernou_prompt")

    mask = iterable_prompt_mask(X, masking_ratio=0.35, lm=3, feature_idx=5)

    plt.figure(figsize=(10, 3))
    plt.pcolormesh(mask.transpose(1, 0))
    plt.show()

    # ipt = torch.randn(128, 12, 24)
    # conv = torch.nn.Conv1d(12, 128, 3)
    # opt = conv(ipt)
    # print('fuck')
    # print(opt.shape)