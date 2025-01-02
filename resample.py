import numpy as np

def to_np_float(x):
    return x.to_numpy().astype(float)

# Resample sequences to match the largest bin size
def distribution_matching(sequence1, sequence2, hist1, hist2, bin_edges):
    resampled_sequence1 = []
    resampled_indices1 = []
    resampled_sequence2 = []
    resampled_indices2 = []
    for count1, count2, (start, end) in zip(hist1, hist2, zip(bin_edges[:-1], bin_edges[1:])):
        if count1 > 0 and count2 > 0:
            bin_indices1 = np.where((sequence1 >= start) & (sequence1 < end))[0]
            bin_indices2 = np.where((sequence2 >= start) & (sequence2 < end))[0]
            
            if count1 > count2:
                indices = np.random.choice(bin_indices2, count1-count2, replace=True)
                resampled_sequence2.extend(sequence2[indices])
                resampled_indices2.extend(indices)
            elif count2 > count1:
                indices = np.random.choice(bin_indices1, count2-count1, replace=True)
                resampled_sequence1.extend(sequence1[indices])
                resampled_indices1.extend(indices)
    return np.array(resampled_indices1), np.array(resampled_indices2)

# resample 1000 for each bin
def resample_to_n(sequence1, sequence2, hist1, hist2, bin_edges, n=400):
    resampled_sequence1 = []
    resampled_indices1 = []
    resampled_sequence2 = []
    resampled_indices2 = []
    for count1, count2, (start, end) in zip(hist1, hist2, zip(bin_edges[:-1], bin_edges[1:])):
        if count1 > 0 and count2 > 0:
            bin_indices1 = np.where((sequence1 >= start) & (sequence1 < end))[0]
            bin_indices2 = np.where((sequence2 >= start) & (sequence2 < end))[0]
            
            indices = np.random.choice(bin_indices2, n, replace=True)
            resampled_sequence2.extend(sequence2[indices])
            resampled_indices2.extend(indices)
        
            indices = np.random.choice(bin_indices1, n, replace=True)
            resampled_sequence1.extend(sequence1[indices])
            resampled_indices1.extend(indices)
    return np.array(resampled_indices1), np.array(resampled_indices2)

# resample CL > 25 for fine-tuning
def resample_tail(sequence1, sequence2, hist1, hist2, bin_edges, n1=1000, n2=200):
    resampled_sequence1 = []
    resampled_indices1 = []
    resampled_sequence2 = []
    resampled_indices2 = []
    for count1, count2, (start, end) in zip(hist1, hist2, zip(bin_edges[:-1], bin_edges[1:])):
        if count1 > 0 and count2 > 0:
            bin_indices1 = np.where((sequence1 >= start) & (sequence1 < end))[0]
            bin_indices2 = np.where((sequence2 >= start) & (sequence2 < end))[0]
            
            if start > 50:
                indices1 = np.random.choice(bin_indices1, n1, replace=True)
                indices2 = np.random.choice(bin_indices2, n1, replace=True)
            else:
                indices1 = np.random.choice(bin_indices1, n2, replace=True)
                indices2 = np.random.choice(bin_indices2, n2, replace=True)
            
            resampled_sequence1.extend(sequence1[indices1])
            resampled_indices1.extend(indices1)
            resampled_sequence2.extend(sequence2[indices2])
            resampled_indices2.extend(indices2)
    return np.array(resampled_indices1), np.array(resampled_indices2)

def resample_CL_threshold(sequence1, sequence2, CL_threshold, greater=True):
    if greater:
        mask1 = sequence1 > CL_threshold
        mask2 = sequence2 > CL_threshold
    else:
        mask1 = sequence1 <= CL_threshold
        mask2 = sequence2 <= CL_threshold
    return np.where(mask1)[0], np.where(mask2)[0]