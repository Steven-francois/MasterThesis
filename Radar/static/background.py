from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader
from Radar.RadarCanReader import RadarCanReader
from Radar.cfar import cfar, extract_targets
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from Radar.doppler_resolution import doppler_resolution


def background(rdc_reader, start_time, end_time, mean=False, stop_time=None):
    # Background timestamps
    background_start = datetime.strptime(start_time, "%Y-%m-%d_%H-%M-%S-%f").timestamp()
    background_end = datetime.strptime(end_time, "%Y-%m-%d_%H-%M-%S-%f").timestamp()
    if mean:
        capture_stop = datetime.strptime(stop_time, "%Y-%m-%d_%H-%M-%S-%f").timestamp()
    print(f"Background start: {background_start}, Background end: {background_end}, Capture stop: {capture_stop}")

    # Background data
    bg_idx_start = np.where(rdc_reader.timestamps > background_start)[0][0]
    bg_idx_end = np.where(rdc_reader.timestamps > background_end)[0][0]
    bg_idx_end = min(bg_idx_end, len(rdc_reader.timestamps) - 1)
    if mean:
        stop_idx = np.where(rdc_reader.timestamps > capture_stop)[0][0]

    bg_rdc = np.zeros((3, rdc_reader.radar_cube_datas[0].shape[0], rdc_reader.radar_cube_datas[0].shape[1]))
    bg_rdc[0] = np.mean([np.max(range_doppler_matrix[:, :, :, 0], axis=2) for range_doppler_matrix in rdc_reader.radar_cube_datas[bg_idx_start:bg_idx_end:3]], axis=0)
    bg_rdc[1] = np.mean([np.max(range_doppler_matrix[:, :, :, 0], axis=2) for range_doppler_matrix in rdc_reader.radar_cube_datas[bg_idx_start+1:bg_idx_end:3]], axis=0)
    bg_rdc[2] = np.mean([np.max(range_doppler_matrix[:, :, :, 0], axis=2) for range_doppler_matrix in rdc_reader.radar_cube_datas[bg_idx_start+2:bg_idx_end:3]], axis=0)
    if mean:
        mean_rdc = np.mean([np.max(range_doppler_matrix[:, :, :, 0], axis=2) for range_doppler_matrix in rdc_reader.radar_cube_datas[bg_idx_start:stop_idx]], axis=0)

    bg_rdc[0] = np.abs(bg_rdc[0])  
    bg_rdc[1] = np.abs(bg_rdc[1])
    bg_rdc[2] = np.abs(bg_rdc[2])
    mean_rdc = np.abs(mean_rdc)

    if mean:
        return bg_idx_start, bg_rdc, mean_rdc
    return bg_idx_start, bg_rdc


# Background from 2025-05-14_14-03-25-894467 to 2025-05-14_14-03-27-894547
# Stop at 2025-05-14_14-03-42-162317

rdc_reader = RadarPacketPcapngReader()
can_reader = RadarCanReader()
bg_idx_start, bg_rdc, mean_rdc = None, None, None



# Define radar cube dimensions
N_RANGE_GATES = 200
N_DOPPLER_BINS = 128
image_size = (N_RANGE_GATES, N_DOPPLER_BINS)
# Define plot limits
xmin = -80/3.6
xmax = 80/3.6
# xmin = -200/3.6
# xmax = 200/3.6
ymin = 0
ymax = 96


fig, (ax1, ax2, ax3)= plt.subplots(1, 3)
# fig, (ax1, ax2)= plt.subplots(1, 2)
img = ax1.imshow(np.zeros(image_size), vmin=0, vmax=100, aspect='auto', cmap='jet', origin='lower')
ax1.set_title("Range-Doppler Map w/ bg")
ax1.set_xlabel("Doppler (m/s)")
ax1.set_ylabel("Range (m)")
ax1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
img2 = ax2.imshow(np.zeros(image_size), vmin=0, vmax=100, aspect='auto', cmap='jet', origin='lower')
scat2 = ax2.scatter([], [], marker='o', color='r')
ax2.set_title("Range-Doppler Map w/o bg")
ax2.set_xlabel("Doppler (m/s)")
ax2.set_ylabel("Range (m)")
ax2.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
img3 = ax3.imshow(np.zeros(image_size), vmin=0, vmax=1, aspect='auto', cmap='gray', origin='lower')
scat = ax3.scatter([], [], marker='o', color='b')
ax3.set_title("CFAR Map w/o bg")
ax3.set_xlabel("Doppler (m/s)")
ax3.set_ylabel("Range (m)")
ax3.set(xlim=(xmin, xmax), ylim=(ymin, ymax))



def update(frame):
    range_doppler_matrix = rdc_reader.radar_cube_datas[frame]
    range_doppler_matrix = np.max(range_doppler_matrix[:,:,:,0], axis=2)
    range_doppler_matrix = np.abs(range_doppler_matrix)  # Convert to dB scale
    img.set_data(range_doppler_matrix)
    bg_rdmatrix = range_doppler_matrix - bg_rdc[(frame-bg_idx_start)%3] - 2*mean_rdc
    bg_rdmatrix[bg_rdmatrix < 0] = 0
    img2.set_data(bg_rdmatrix)
    # img2.set_data(bg_rdc[0])
    mask, _, _ = cfar(bg_rdmatrix, n_guard=(1,1), n_ref=(2,3), bias=3, method='CA', min_treshold=30)
    # mask, _, _ = cfar(range_doppler_matrix, n_guard=(1,1), n_ref=(2,3), bias=3, method='CA')
    img3.set_data(mask)
    
    can_target = can_reader.can_targets[frame]
    targets_data = can_target.targets_data
    x = []
    y = []
    values = []
    for target in targets_data:
        x.append(target.t_range)
        y.append(target.speed_radial)
        values.append(target.rcs)
    data = np.stack([y,x]).T
    scat.set_offsets(data)
    scat.set_array(np.array(values))
    
    
    properties = rdc_reader.all_properties[frame]
    DOPPLER_RESOLUTION  = properties[0]
    RANGE_RESOLUTION    = properties[1]
    BIN_PER_SPEED       = properties[2]
    xt_left = -N_DOPPLER_BINS//2*DOPPLER_RESOLUTION
    xt_right = (N_DOPPLER_BINS//2 - 1)*DOPPLER_RESOLUTION
    yt_bottom = 0
    yt_top = N_RANGE_GATES*RANGE_RESOLUTION
    img.set_extent([xt_left, xt_right, yt_bottom, yt_top])
    img2.set_extent([xt_left, xt_right, yt_bottom, yt_top])
    img3.set_extent([xt_left, xt_right, yt_bottom, yt_top])
    ax1.set_title(f"Image {frame}")
    
    cfar_targets = extract_targets(range_doppler_matrix, mask, properties)
    r_targets = doppler_resolution(cfar_targets, targets_data)
    r = []
    s = []
    for point in r_targets:
        r_range, r_doppler = point['centroid']
        r_range = r_range * RANGE_RESOLUTION + yt_bottom
        r_doppler = r_doppler * DOPPLER_RESOLUTION + xt_left + N_DOPPLER_BINS*DOPPLER_RESOLUTION*point['doppler_band']
        r.append(r_range)
        s.append(r_doppler)
    data = np.stack([s,r]).T
    scat2.set_offsets(data)
    return img, img2

if __name__ == "__main__":
    rdc_reader.load("Fusion/data/test/radar_cube_data")
    print(rdc_reader.timestamps[0], rdc_reader.timestamps[-1])
    can_reader.load_npy("Fusion/data/test/radar_can_data.npy")
    print(can_reader.can_targets[0].targets_header.real_time, can_reader.can_targets[-1].targets_header.real_time)
    can_reader.filter_targets_speed(1, 200)
    can_reader.cluster_with_dbscan(2,2)

    bg_idx_start, bg_rdc, mean_rdc = background(rdc_reader, "2025-05-14_14-03-25-894467", "2025-05-14_14-03-27-894547", mean=True, stop_time="2025-05-14_14-03-42-162317")

    ani = animation.FuncAnimation(fig, update, frames=range(1100, 1300), interval=100)
    plt.show()
    # ani.save("Fusion/data/2/radar_animation.gif", writer='pillow', fps=30)
    exit()
    frame = 1160
    
    range_doppler_matrix = rdc_reader.radar_cube_datas[frame]
    range_doppler_matrix = np.mean(range_doppler_matrix[:,:,:,0], axis=2)
    range_doppler_matrix = np.abs(range_doppler_matrix)  # Convert to dB scale
    img.set_data(range_doppler_matrix)
    bg_rdmatrix = range_doppler_matrix - bg_rdc[(frame-bg_idx_start)%3] #- mean_rdc
    bg_rdmatrix[bg_rdmatrix < 0] = 0
    img2.set_data(bg_rdmatrix)
    # img2.set_data(bg_rdc[0])
    mask, _, _ = cfar(bg_rdmatrix, n_guard=(1,1), n_ref=(2,3), bias=3, method='CA', min_treshold=10)
    img3.set_data(mask)
    
    
    
    can_target = can_reader.can_targets[frame]
    targets_data = can_target.targets_data
    x = []
    y = []
    values = []
    for target in targets_data:
        x.append(target.t_range)
        y.append(target.speed_radial)
        values.append(target.rcs)
    data = np.stack([y,x]).T
    scat.set_offsets(data)
    scat.set_array(np.array(values))
    
    
    properties = rdc_reader.all_properties[frame]
    DOPPLER_RESOLUTION  = properties[0]
    RANGE_RESOLUTION    = properties[1]
    BIN_PER_SPEED       = properties[2]
    xt_left = -N_DOPPLER_BINS//2*DOPPLER_RESOLUTION
    xt_right = (N_DOPPLER_BINS//2 - 1)*DOPPLER_RESOLUTION
    yt_bottom = 0
    yt_top = N_RANGE_GATES*RANGE_RESOLUTION
    img.set_extent([xt_left, xt_right, yt_bottom, yt_top])
    img2.set_extent([xt_left, xt_right, yt_bottom, yt_top])
    img3.set_extent([xt_left, xt_right, yt_bottom, yt_top])
    ax1.set_title(f"Image {frame}")
    
    cfar_targets = extract_targets(range_doppler_matrix, mask, properties)
    print(cfar_targets)
    closest_points = []
    for cfar_target in cfar_targets:
        cfar_range, cfar_doppler = cfar_target['centroid']
        cfar_range = cfar_range * RANGE_RESOLUTION + yt_bottom
        cfar_doppler = cfar_doppler * DOPPLER_RESOLUTION + xt_left
        candidate_dist = []
        candidate_idxs = []
        for i in range(-1, 2):
            distances = np.sqrt((np.array(x) - cfar_range)**2 + (np.array(y) - (cfar_doppler+i*N_DOPPLER_BINS*DOPPLER_RESOLUTION))**2)
            closest_idx = np.argmin(distances)
            candidate_dist.append(distances[closest_idx])
            candidate_idxs.append(closest_idx)
        candidate_idx = np.argmin(candidate_dist)
        closest_idx = candidate_idxs[candidate_idx]
        if candidate_dist[candidate_idx] < 10:
            closest_points.append({
                'cfar_target': cfar_target,
                'closest_point': {
                    'range': x[closest_idx],
                    'doppler': y[closest_idx],
                    'rcs': values[closest_idx]
                },
                'distance': distances[closest_idx]
            })
    r = []
    s = []
    v = []
    for point in closest_points:
        print(f"CFAR Target: {point['cfar_target']}, Closest Point: {point['closest_point']}, Distance: {point['distance']}")
        r.append(point['closest_point']['range'])
        s.append(point['closest_point']['doppler'])
        v.append(point['closest_point']['rcs'])
    data = np.stack([s,r]).T
    scat2.set_offsets(data)
    scat2.set_array(np.array(v))
    
    plt.show()