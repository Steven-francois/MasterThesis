import numpy as np
import matplotlib.pyplot as plt

# Based on the CFAR algorithm described in  Marshall Bruner's video on CFAR
def cfar(range_doppler, n_guard, n_ref, bias, method):
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

    Returns
    -------
    numpy.ndarray
        Binary mask of detected targets.
    """
    mask = np.zeros_like(range_doppler, dtype=bool)
    output = np.zeros_like(range_doppler)
    for i in range(n_guard + n_ref, range_doppler.shape[0] - (n_guard + n_ref)):
        for j in range(n_guard + n_ref, range_doppler.shape[1] - (n_guard + n_ref)):
            left_ref_cells = range_doppler[i - (n_guard+n_ref):i - n_guard, j]
            right_ref_cells = range_doppler[i + n_guard + 1:i + (n_guard+n_ref) + 1, j]
            top_ref_cells = range_doppler[i, j - (n_guard+n_ref):j - n_guard]
            bottom_ref_cells = range_doppler[i, j + n_guard + 1:j + (n_guard+n_ref) + 1]
            
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
            
            threshold = mean * bias
            output[i, j] = threshold
            if range_doppler[i, j] > threshold:
                mask[i, j] = 1
    targets_only = mask * range_doppler
    return mask, output, targets_only

if __name__ == "__main__":
    from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as RadarPacketReader
    nb_file = "30"
    rdc_file = f"Fusion/data/radar_cube_data_{nb_file}" # Replace with your output file path
    rdc_reader = RadarPacketReader("", rdc_file)
    
    rdc_reader.load()
    radar_cube_data = rdc_reader.radar_cube_datas
    
    range_doppler = np.abs(radar_cube_data[0][:,:,0,0])
    
    # Apply CFAR algorithm
    mask, threshold, targets_only = cfar(range_doppler, n_guard=2, n_ref=2, bias=3, method='GO')
    
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(range_doppler, cmap='jet', origin='lower')
    ax1.set_title("Original Data")
    ax2.imshow(mask, cmap='gray', origin='lower')
    ax2.set_title("Detected Targets")
    # plt.show()
    
    
    
    plt.plot(10 * np.log10(np.abs(range_doppler[7])), label="X[k]", c="b")
    plt.plot(10 * np.log10(np.abs(threshold[7])), label="Threshold", c="y")
    plt.plot(10 * np.log10(np.abs(targets_only[7])), label="Targets", c="r")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.title("X[k] with CFAR Threshold and Classified Targets")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.show()