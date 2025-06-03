import numpy as np

N_DOPPLER_BINS = 128
N_RANGE_GATES = 200

def doppler_resolution(cfar_targets, targets_data, nb_bands=3):
    resolved_points = []
    if len(cfar_targets) == 0:
        return resolved_points
    
    x = []
    y = []
    for target in targets_data:
        x.append(target.t_range)
        y.append(target.speed_radial)
    x = np.array(x)
    y = np.array(y)
    
    for cfar_target in cfar_targets:
        RANGE_RESOLUTION = cfar_target['properties'][1]
        DOPPLER_RESOLUTION = cfar_target['properties'][0]
        cfar_range, cfar_doppler = cfar_target['centroid']
        cfar_range = cfar_range * RANGE_RESOLUTION
        cfar_doppler = cfar_doppler * DOPPLER_RESOLUTION -N_DOPPLER_BINS//2*DOPPLER_RESOLUTION
        candidate_dist = []
        candidate_bands = []
        for i in range(-nb_bands, nb_bands + 1):
            distances = np.sqrt((np.array(x) - cfar_range)**2 + (np.array(y) - (cfar_doppler+i*N_DOPPLER_BINS*DOPPLER_RESOLUTION))**2)
            if len(distances) == 0:
                continue
            closest_idx = np.argmin(distances)
            candidate_dist.append(distances[closest_idx])
            candidate_bands.append(i)
        if len(candidate_dist) == 0:
            continue
        candidate_idx = np.argmin(candidate_dist)
        doppler_band = candidate_bands[candidate_idx]
        if candidate_dist[candidate_idx] < 2:
            cfar_target['doppler_band'] = doppler_band
            idxs = np.array(cfar_target['idxs'])
            cfar_target['coord'] = np.zeros_like(idxs, dtype=np.float32)
            cfar_target['coord'][:, 0] = idxs[:, 0] * RANGE_RESOLUTION
            cfar_target['coord'][:, 1] = (idxs[:, 1] - N_DOPPLER_BINS//2 + doppler_band*N_DOPPLER_BINS) * DOPPLER_RESOLUTION
            resolved_points.append(cfar_target)
        elif abs(cfar_doppler) < 1:
            cfar_target['doppler_band'] = 0
            idxs = np.array(cfar_target['idxs'])
            cfar_target['coord'] = np.zeros_like(idxs, dtype=np.float32)
            cfar_target['coord'][:, 0] = idxs[:, 0] * RANGE_RESOLUTION
            cfar_target['coord'][:, 1] = (idxs[:, 1] - N_DOPPLER_BINS//2) * DOPPLER_RESOLUTION
            resolved_points.append(cfar_target)
    
    return resolved_points

def range_doppler_resolved(resolved_targets, nb_bands):
    """ Create a new extended range doppler map based on resolved targets.
    This function takes resolved targets and the number of (new positive/negative) bands, and returns a new range-doppler map

    Args:
        resolved_targets (list): List of resolved targets with doppler bands.
        nb_bands (int): Number of doppler bands to consider.
    """
    
    rd_map = np.zeros((N_RANGE_GATES, (1 + 2*nb_bands) * N_DOPPLER_BINS ), dtype=np.int8)
    for i, target in enumerate(resolved_targets):
        idxs = np.array(target['idxs'])
        doppler_band = target.get('doppler_band', 0)
        for idx in idxs:
            rd_map[idx[0], idx[1] + (doppler_band+nb_bands) * N_DOPPLER_BINS] = i+1
    return rd_map