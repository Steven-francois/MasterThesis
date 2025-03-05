import numpy as np
from tqdm import tqdm

class RadarPacketReader:
    def __init__(self, filename, rdc_file="rdc_file.npy"):
        self.filename = filename
        self.rdc_file = rdc_file
        self.fields = None
        self.timestamps = None
        self.nb_frames = None
        self.all_properties = None
        self.radar_cube_datas = None
        
        # Define parameters
        self.IP_SOURCE = "192.168.11.11"
        self.IP_DEST = "192.168.11.17"
        self.RADAR_CUBE_UDP_PORT = 50005  # Radar Cube Data port
        self.BIN_PROPERTIES_UDP_PORT = 50063  # Bin Properties port
        self.header_size = 22 + 64  # Header size from document
        
        # Data fields greater than 1 Byte are transmitted in Big Endian.
        self.dt_uint64 = np.dtype(np.uint64).newbyteorder('>')
        self.dt_uint32 = np.dtype(np.uint32).newbyteorder('>')
        self.dt_uint16 = np.dtype(np.uint16).newbyteorder('>')
        self.dt_int16 = np.dtype(np.int16).newbyteorder('>')
        self.dt_float32 = np.dtype(np.float32).newbyteorder('>')
        
        # Radar cube dimensions
        self.N_RANGE_GATES       = None
        self.N_DOPPLER_BINS      = None
        self.N_RX_CHANNELS       = None
        self.N_CHIRP_TYPES       = None
        self.FIRST_RANGE_GATE    = None
        
    
    def extract_header(self, data):
        data_ = data[22:] # Skip the first 22 bytes
        header_size = 64
        header = data_[24:header_size] # Extract the header
        offsets = header[:24]
        imag_offset, real_offset, range_gate_offset, doppler_bin_offset, rx_channel_offset, chirp_type_offset = np.frombuffer(offsets, dtype=self.dt_uint32)
        indexes = header[24:30]
        indexes = np.frombuffer(indexes, dtype=self.dt_uint16)
        range_gates, first_range_gate, doppler_bins = indexes
        rx_channels, chirp_types, element_size, element_type = header[30:34]
        
        # Return np array to be saved
        return np.array([imag_offset, real_offset, range_gate_offset, doppler_bin_offset, rx_channel_offset, chirp_type_offset, range_gates, first_range_gate, doppler_bins, rx_channels, chirp_types, element_size, element_type])
    
    def process_radar_cube_data(self, radar_data):
        # Check if the data size is correct
        expected_size = self.N_RANGE_GATES * self.N_DOPPLER_BINS * self.N_RX_CHANNELS * self.N_CHIRP_TYPES * 2 * 2
        if len(radar_data) != expected_size: #Skip incomplete frames
            # print(f"Expected size: {expected_size}, Received size: {len(radar_data)}")
            print("!", end="")
            return None
        print(".", end="")
        # Convert to NumPy array and reshape
        radar_cube = np.frombuffer(radar_data, dtype=self.dt_int16)
        radar_cube = radar_cube.reshape((self.N_CHIRP_TYPES, self.N_RANGE_GATES, self.N_RX_CHANNELS, self.N_DOPPLER_BINS, 2))
        # Transpose to (range gate, doppler, rx, chirp)
        radar_cube = np.transpose(radar_cube, (1, 3, 2, 0, 4))

        # Extract real and imaginary parts
        real_part = radar_cube[..., 0]
        imag_part = radar_cube[..., 1]

        # Convert to complex values
        complex_data = real_part + 1j * imag_part

        range_doppler_matrix = np.fft.fftshift(complex_data, axes=1)

        # Return np array to be saved
        return range_doppler_matrix
    
    def extract_properties(self, data):
        frame_counter  = np.frombuffer(data[14:18], dtype=self.dt_uint32)[0]
        properties = data[22+24:]
        properties = np.frombuffer(properties, dtype=self.dt_float32)

        # Return np array to be saved
        return (frame_counter, properties)
    
    def save(self):        
        with open(self.rdc_file, "wb") as f:
            np.save(f, self.fields)
            np.save(f, self.timestamps)
            np.save(f, self.nb_frames)
            for i in tqdm(range(self.nb_frames), desc="Saving Properties"):
                np.save(f, self.all_properties[i])
            for i in tqdm(range(self.nb_frames), desc="Saving Radar Cube Data"):
                np.save(f, self.radar_cube_datas[i])
                
    def load(self):
        with open(self.rdc_file, 'rb') as f:
            self.fields = np.load(f, allow_pickle=True)
            self.timestamps = np.load(f, allow_pickle=True)
            self.nb_frames = np.load(f, allow_pickle=True)
            self.all_properties = np.array([np.load(f, allow_pickle=True)])
            for _ in range(self.nb_frames-1):
                self.all_properties = np.append(self.all_properties, [np.load(f, allow_pickle=True)], axis=0)
            self.radar_cube_datas = np.array([np.load(f, allow_pickle=True)])
            for _ in range(self.nb_frames-1):
                self.radar_cube_datas = np.append(self.radar_cube_datas, [np.load(f, allow_pickle=True)], axis=0)
                
    def progress_bar(self, iterable, func, description):
        for i in tqdm(iterable, desc=description):
            func(i)