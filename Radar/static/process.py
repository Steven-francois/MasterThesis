import json
from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader
from Radar.RadarCanReader import RadarCanReader
from Radar.static.background import background
import os
import numpy as np
from Radar.cfar import cfar, extract_targets
from scipy.ndimage import gaussian_filter
from Radar.doppler_resolution import doppler_resolution, range_doppler_resolved
from tqdm import trange

nb = "1_1"
data_folder = f"Data/{nb}/"
rdc_reader = RadarPacketPcapngReader()
rdc_reader.load(f"{data_folder}radar_cube_data")
can_reader = RadarCanReader()
can_reader.load_npy(f"{data_folder}radar_can_data.npy")
can_reader.filter_targets_speed(1, 200)
bg_idx_start, bg_rdc, mean_rdc = background(rdc_reader, "2025-05-14_14-08-15-875622", "2025-05-14_14-08-57-842836", mean=True, stop_time="2025-05-14_14-09-11-042610")
radar_folder = os.path.join(data_folder, "radar")
resolved_rdm_folder = os.path.join(data_folder, "radar", "rdm")
targets_folder = os.path.join(data_folder, "radar", "targets")
os.makedirs(resolved_rdm_folder, exist_ok=True)
os.makedirs(targets_folder, exist_ok=True)


nb_bands = 3  # Number of doppler bands to consider

with open(os.path.join(radar_folder, f"rdm.npy"), 'wb') as f_rdm:
    for i in trange(len(can_reader.can_targets)):
        range_doppler_matrix = rdc_reader.radar_cube_datas[i]
        range_doppler_matrix = np.max(range_doppler_matrix[:,:,:,0], axis=2)
        range_doppler_matrix = np.abs(range_doppler_matrix)
        bg_rdmatrix = range_doppler_matrix - bg_rdc[(i-bg_idx_start)%3] - 2*mean_rdc
        bg_rdmatrix[bg_rdmatrix < 0] = 0
        bg_rdmatrix = gaussian_filter(bg_rdmatrix, sigma=(2,1))  # Apply Gaussian filter for smoothing
        mask, _, _ = cfar(bg_rdmatrix, n_guard=(1,1), n_ref=(2,3), bias=1, method='CA', min_treshold=5)
        
        can_target = can_reader.can_targets[i]
        targets_data = can_target.targets_data
        
        properties = rdc_reader.all_properties[i]
        DOPPLER_RESOLUTION  = properties[0]
        RANGE_RESOLUTION    = properties[1]
        BIN_PER_SPEED       = properties[2]
        
        cfar_targets = extract_targets(range_doppler_matrix, mask, properties)
        r_targets = doppler_resolution(cfar_targets, targets_data, nb_bands)
        r_rdm = range_doppler_resolved(r_targets, nb_bands)
        
        # with open(os.path.join(resolved_rdm_folder, f"rdm_{i}.npy"), 'wb') as f:
        np.save(f_rdm, r_rdm)
        with open(os.path.join(targets_folder, f"targets_{i}.json"), 'w') as f:
            for target in r_targets:
                target['peak'] = [int(target['peak'][0]), int(target['peak'][1])]
                target['value'] = int(target['value'])
                target['idxs'] = [[int(idx), int(jdx)] for idx, jdx in target['idxs']]
                target['properties'] = [float(prop) for prop in target['properties']]
                target['coord'] = [[float(coord[0]), float(coord[1])] for coord in target['coord']]
            json.dump(r_targets, f)
        
        