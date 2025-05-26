import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import label

# Based on the CFAR algorithm described in  Marshall Bruner's video on CFAR
def cfar(range_doppler, n_guard, n_ref, bias, method, min_treshold=30):
    """
    Constant False Alarm Rate (CFAR) algorithm for 2D radar data.

    Parameters
    ----------
    range_doppler : numpy.ndarray
        2D radar data matrix (range x doppler).
    n_guard : int
        Number of guard cells.
    n_ref : int
        Number of reference cells.
    bias : float
        Bias factor.
    method : str
        CFAR method to use (e.g. 'CA', 'GO', 'SO').
    min_treshold : float, optional
        Minimum threshold value (default is 30).

    Returns
    -------
    numpy.ndarray
        Binary mask of detected targets.
    """
    mask = np.zeros_like(range_doppler, dtype=bool)
    output = np.zeros_like(range_doppler)
    # for i in range(n_guard[0] + n_ref[0], range_doppler.shape[0] - (n_guard[0] + n_ref[0])):
    #     for j in range(n_guard[1] + n_ref[1], range_doppler.shape[1] - (n_guard[1] + n_ref[1])):
    for i in range(range_doppler.shape[0]):
        for j in range(range_doppler.shape[1]):
            left_ref_cells = range_doppler[i - (n_guard[0]+n_ref[0]):i - n_guard[0], j]
            right_ref_cells = range_doppler[i + n_guard[0] + 1:i + (n_guard[0]+n_ref[0]) + 1, j]
            top_ref_cells = range_doppler[i, j - (n_guard[1]+n_ref[1]):j - n_guard[1]]
            bottom_ref_cells = range_doppler[i, j + n_guard[1] + 1:j + (n_guard[1]+n_ref[1]) + 1]
            
            left_mean = np.mean(left_ref_cells)
            right_mean = np.mean(right_ref_cells)
            top_mean = np.mean(top_ref_cells)
            bottom_mean = np.mean(bottom_ref_cells)
            
            if method == 'CA':
                mean = np.mean(np.concatenate((left_ref_cells, right_ref_cells, top_ref_cells, bottom_ref_cells)))
            elif method == 'GO':
                mean = max(left_mean, right_mean, top_mean, bottom_mean)
            elif method == 'SO':
                mean = min(left_mean, right_mean, top_mean, bottom_mean)
            else:
                raise ValueError("Invalid CFAR method")
            
            threshold = max(mean * bias, min_treshold)
            output[i, j] = threshold
            if range_doppler[i, j] > threshold:
                mask[i, j] = 1
    targets_only = mask * range_doppler
    return mask, output, targets_only

def cfar_fast(range_doppler, n_guard, n_ref, bias, method, min_treshold=30):
    """
    Fast CFAR algorithm for 2D radar data using sliding window.

    Parameters
    ----------
    range_doppler : numpy.ndarray
        2D radar data matrix (range x doppler).
    n_guard : int
        Number of guard cells.
    n_ref : int
        Number of reference cells.
    bias : float
        Bias factor.
    method : str
        CFAR method to use (e.g. 'CA', 'GO', 'SO').
    min_treshold : float, optional
        Minimum threshold value (default is 30).

    Returns
    -------
    numpy.ndarray
        Binary mask of detected targets.
    """
    # Create a sliding window view of the range_doppler data
    window_shape = (n_guard[0] + n_ref[0] * 2 + 1, n_guard[1] + n_ref[1] * 2 + 1)
    pad = ((n_guard[0] + n_ref[0] -1, n_guard[0] + n_ref[0]), (n_guard[1] + n_ref[1]-1, n_guard[1] + n_ref[1]))
    windows = sliding_window_view(range_doppler, window_shape)
    
    # Calculate the mean for each window based on the selected CFAR method
    if method == 'CA':
        mean = np.mean(windows, axis=(2, 3))
    elif method == 'GO':
        mean = np.max(windows, axis=(2, 3))
    elif method == 'SO':
        mean = np.min(windows, axis=(2, 3))
    
    # Pad the mean to match the original range_doppler shape
    mean = np.pad(mean, pad_width=pad, mode='edge')
    # mean = mean[n_guard[0]:-n_guard[0], n_guard[1]:-n_guard[1]]
    
    # Apply bias and minimum threshold
    threshold = np.maximum(mean * bias, min_treshold)
    
    # Create a mask based on the threshold
    mask = range_doppler > threshold
    
    return mask, threshold, mask * range_doppler

def extract_targets(range_doppler, mask, properties):
    """
    Extract targets from the range_doppler data using the CFAR mask.

    Parameters
    ----------
    range_doppler : numpy.ndarray
        2D radar data matrix (range x doppler).
    mask : numpy.ndarray
        Binary mask of detected targets.

    Returns
    -------
    numpy.ndarray
        Extracted targets.
    """
    labeled_mask, n_targets = label(mask)
    targets = []
    # Separate targets from the range_doppler data
    for t in range(1, n_targets + 1):
        idxs = np.argwhere(labeled_mask == t)
        if len(idxs) == 0:
            continue
        
        range_idxs, doppler_idxs = idxs[:, 0], idxs[:, 1]
        target = range_doppler[range_idxs, doppler_idxs]
        
        # centroid
        range_centroid = np.mean(range_idxs)
        doppler_centroid = np.mean(doppler_idxs)
        # print(f"Target {t}: Centroid at ({range_centroid}, {doppler_centroid})")
        peak_idx = np.argmax(target)
        peak_range = range_idxs[peak_idx]
        peak_doppler = doppler_idxs[peak_idx]
        # print(f"Target {t}: Peak at ({peak_range}, {peak_doppler})")
        # print(f"Target {t}: Peak value {target[peak_idx]}")
        
        target_info = {
            'centroid': (range_centroid, doppler_centroid),
            'peak': (peak_range, peak_doppler),
            'value': target[peak_idx],
            'nb_cells': len(idxs),
            'idxs': idxs,
            'properties': properties
        }
        targets.append(target_info)
    return targets

if __name__ == "__main__":
    from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as RadarPacketReader
    nb_file = "50"
    rdc_file = f"Radar/data/radar_cube_data_{nb_file}" # Replace with your output file path
    rdc_reader = RadarPacketReader("", rdc_file)
    
    rdc_reader.load()
    radar_cube_data = rdc_reader.radar_cube_datas
    properties = rdc_reader.all_properties
    
    
    
    
    range_doppler = np.abs(radar_cube_data[100][:,:,0,0])
    
    # Apply CFAR algorithm
    mask, threshold, targets_only = cfar(range_doppler, n_guard=(1,1), n_ref=(2,3), bias=6, method='CA')
    tx1 = mask
    tx2 = np.roll(mask, 22, 1)
    tx3 = np.roll(mask, 42, 1)
    tx4 = np.roll(mask, -22, 1)
    tx5 = np.roll(mask, -42, 1)
    tx6 = np.roll(mask, -64, 1)
    # binary and operation between all tx
    mask_ = sum([tx1, tx2, tx3, tx4, tx5, tx6])
    fig, (ax, ax1, ax2, ax3, ax4, ax5, ax6, ax_) = plt.subplots(1, 8)
    ax.imshow(range_doppler, cmap='jet', origin='lower')
    ax.set_title("Original Data")
    ax1.imshow(tx1, cmap='gray', origin='lower')
    ax1.set_title("TX1")
    ax2.imshow(tx2, cmap='gray', origin='lower')
    ax2.set_title("TX2")
    ax3.imshow(tx3, cmap='gray', origin='lower')
    ax3.set_title("TX3")
    ax4.imshow(tx4, cmap='gray', origin='lower')
    ax4.set_title("TX4")
    ax5.imshow(tx5, cmap='gray', origin='lower')
    ax5.set_title("TX5")
    ax6.imshow(tx6, cmap='gray', origin='lower')
    ax6.set_title("TX6")
    ax_.imshow(mask_, cmap='jet', origin='lower')
    ax_.set_title("Detected Targets")
    plt.show()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # Plot the results
    # ax1.imshow(range_doppler, cmap='jet', origin='lower')
    # ax1.set_title("Original Data")
    # ax2.imshow(mask, cmap='gray', origin='lower')
    # ax2.set_title("Detected Targets")
    # plt.show()
    
    
    
    # plt.plot(10 * np.log10(np.abs(range_doppler[6])), label="X[k]", c="b")
    # plt.plot(10 * np.log10(np.abs(threshold[6])), label="Threshold", c="y")
    # plt.plot(10 * np.log10(np.abs(targets_only[6])), label="Targets", c="r")
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.title("X[k] with CFAR Threshold and Classified Targets")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Magnitude (dB)")
    # plt.show()
    # exit()
    targets = []
    def frame(i):
        N_DOPPLER_BINS = 128
        N_RANGE_GATES = 200
        RANGE_RESOLUTION = properties[i][1]
        DOPPLER_RESOLUTION = properties[i][0]
        xt_left = -N_DOPPLER_BINS//2*DOPPLER_RESOLUTION*3.6
        xt_right = (N_DOPPLER_BINS//2 - 1)*DOPPLER_RESOLUTION*3.6
        yt_bottom = 0
        yt_top = N_RANGE_GATES*RANGE_RESOLUTION
        print(i)
        ax2.clear()
        range_doppler = np.abs(radar_cube_data[i][:,:,0,0])
        img1 = ax1.imshow(range_doppler, cmap='jet', origin='lower')
        img1.set_extent([xt_left, xt_right, yt_bottom, yt_top])
        mask, threshold, targets_only = cfar_fast(range_doppler, n_guard=(1,1), n_ref=(2,3), bias=1, method='GO')
        img2 = ax2.imshow(mask, cmap='gray', origin='lower')
        img2.set_extent([xt_left, xt_right, yt_bottom, yt_top])
        targets.append(extract_targets(range_doppler, mask, properties[i]))
        labeled_mask, n_targets = label(mask)
        print(f"Number of targets detected: {n_targets}")
        img3 = ax3.imshow(labeled_mask, cmap='jet', origin='lower')
        img3.set_extent([xt_left, xt_right, yt_bottom, yt_top])
        ax3.set_title("Labeled Targets")
        ax3.set_xlabel("Doppler Bins")
        ax3.set_ylabel("Range Gates")
        plt.title("Detected Targets")
        # plt.savefig(f"Fusion/data/cfar_{nb_file}_{i}.png")
        # r = 33
        # plt.plot(10 * np.log10(np.abs(range_doppler[r])), label="X[k]", c="b")
        # plt.plot(10 * np.log10(np.abs(threshold[r])), label="Threshold", c="y")
        # plt.plot(10 * np.log10(np.abs(targets_only[r])), label="Targets", c="r")
        
    ani = animation.FuncAnimation(fig, frame, frames=range(390, 500), interval=62, repeat=False)
    # ani = animation.FuncAnimation(fig, frame, frames=range(115,118), interval=1500, repeat=False)
    plt.show()
    
    for i, chirp_targets in enumerate(targets):
        print(f"Chirp nb {i}:")
        for t, target in enumerate(chirp_targets):
            if target['centroid'][0] >=91 and target['centroid'][0] <= 107:# and target['centroid'][1] >= 100 and target['centroid'][1] <= 130:
                print(f"Target {t}: Centroid at {target['centroid']}, Peak at {target['peak']}, Peak value {target['value']}, Number of cells {target['nb_cells']}, Properties {target['properties']}")
    