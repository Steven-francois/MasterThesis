import numpy as np

N_DOPPLER_BINS = 128
N_RANGE_GATES = 200

def doppler_resolution(cfar_targets, targets_data):
    resolved_points = []
    if len(cfar_targets) == 0:
        return resolved_points
    
    RANGE_RESOLUTION = cfar_targets[0]['properties'][1]
    DOPPLER_RESOLUTION = cfar_targets[0]['properties'][0]
    
    x = []
    y = []
    for target in targets_data:
        x.append(target.t_range)
        y.append(target.speed_radial)
    x = np.array(x)
    y = np.array(y)
    
    for cfar_target in cfar_targets:
        cfar_range, cfar_doppler = cfar_target['centroid']
        cfar_range = cfar_range * RANGE_RESOLUTION
        cfar_doppler = cfar_doppler * DOPPLER_RESOLUTION -N_DOPPLER_BINS//2*DOPPLER_RESOLUTION
        candidate_dist = []
        candidate_bands = []
        for i in range(-1, 2):
            distances = np.sqrt((np.array(x) - cfar_range)**2 + (np.array(y) - (cfar_doppler+i*N_DOPPLER_BINS*DOPPLER_RESOLUTION))**2)
            closest_idx = np.argmin(distances)
            candidate_dist.append(distances[closest_idx])
            candidate_bands.append(i)
        candidate_idx = np.argmin(candidate_dist)
        doppler_band = candidate_bands[candidate_idx]
        if candidate_dist[candidate_idx] < 2:
            cfar_target['doppler_band'] = doppler_band
            resolved_points.append(cfar_target)
        elif abs(cfar_doppler) < 1.5:
            cfar_target['doppler_band'] = 0
            resolved_points.append(cfar_target)
    
    return resolved_points