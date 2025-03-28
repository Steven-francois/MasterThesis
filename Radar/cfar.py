import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
            
            threshold = mean * bias
            output[i, j] = threshold
            if range_doppler[i, j] > threshold:
                mask[i, j] = 1
    targets_only = mask * range_doppler
    return mask, output, targets_only

if __name__ == "__main__":
    from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as RadarPacketReader
    nb_file = "21"
    rdc_file = f"Fusion/data/radar_cube_data_{nb_file}" # Replace with your output file path
    rdc_reader = RadarPacketReader("", rdc_file)
    
    rdc_reader.load()
    radar_cube_data = rdc_reader.radar_cube_datas
    
    
    
    
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
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
    
    def frame(i):
        print(i)
        ax2.clear()
        range_doppler = np.abs(radar_cube_data[i][:,:,0,0])
        ax1.imshow(range_doppler, cmap='jet', origin='lower')
        mask, threshold, targets_only = cfar(range_doppler, n_guard=(1,1), n_ref=(2,3), bias=6, method='CA')
        ax2.imshow(mask, cmap='gray', origin='lower')
        r = 33
        plt.plot(10 * np.log10(np.abs(range_doppler[r])), label="X[k]", c="b")
        plt.plot(10 * np.log10(np.abs(threshold[r])), label="Threshold", c="y")
        plt.plot(10 * np.log10(np.abs(targets_only[r])), label="Targets", c="r")
        
    ani = animation.FuncAnimation(fig, frame, frames=range(100,150), interval=1000)
    plt.show()
    exit()