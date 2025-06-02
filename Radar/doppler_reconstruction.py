import numpy as np

N_DOPPLER_BINS = 128
N_RANGE_GATES = 200

def rd_bins_set(can_range, can_doppler, properties):
    """
    Calculate the range and doppler bins for the given range and doppler values.
    
    Args:
        range (float): The range value.
        doppler (float): The doppler value.
        properties (tuple): A tuple containing DOPPLER_RESOLUTION, RANGE_RESOLUTION, BIN_PER_SPEED.
    
    Returns:
        tuple: A tuple containing the list of bins and the band index.
    """
    RANGE_RESOLUTION = properties[1]
    DOPPLER_RESOLUTION = properties[0]
    r_bin = int(can_range // RANGE_RESOLUTION)
    d_bin = int(can_doppler // DOPPLER_RESOLUTION + N_DOPPLER_BINS // 2)
    band = int(d_bin // N_DOPPLER_BINS)
    d_bin = d_bin % N_DOPPLER_BINS
    bins_set  = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            r_bin_i = (r_bin + i) % N_RANGE_GATES
            d_bin_j = (d_bin + j) % N_DOPPLER_BINS
            bins_set.append((r_bin_i, d_bin_j))
    return bins_set, band

def doppler_reconstruct(range_doppler_matrix, targets_data, nb_bands, properties):
    """
    Reconstruct the range-doppler map based on resolved targets.
    
    Args:
        range_doppler_matrix (np.ndarray): The original range-doppler matrix.
        targets_data (list): List of target data with doppler bands.
        nb_bands (int): Number of doppler bands to consider.
        properties (tuple): A tuple containing DOPPLER_RESOLUTION, RANGE_RESOLUTION, BIN_PER_SPEED.
    
    Returns:
        np.ndarray: The reconstructed range-doppler map.
    """
    reconstructed_rdm = np.zeros((N_RANGE_GATES, (1 + 2*nb_bands) * N_DOPPLER_BINS ), dtype=np.float32)
    bands_sets = [[] for _ in range(2*nb_bands + 1)]
    for target in targets_data:
        bins_set, band = rd_bins_set(target.t_range, target.speed_radial, properties)
        bands_sets[band + nb_bands].extend(bins_set)
    
    bands_sets = [np.unique(bins_set, axis=0) for bins_set in bands_sets]
    for band in range(2 * nb_bands + 1):
        r_doppler = band * N_DOPPLER_BINS
        for r_bin, d_bin in bands_sets[band]:
            reconstructed_rdm[r_bin, r_doppler + d_bin] = range_doppler_matrix[r_bin, d_bin]
        
    return reconstructed_rdm