import numpy as np

class RadarPacketReader:
    def __init__(self, filename):
        self.filename = filename
        self.fields = None
        self.timestamps = None
        self.nb_frames = None
        self.all_properties = None
        self.radar_cube_data = None
        
        # Define parameters
        self.IP_SOURCE = "192.168.11.11"
        self.IP_DEST = "192.168.11.17"
        self.RADAR_CUBE_UDP_PORT = 50005  # Radar Cube Data port
        self.BIN_PROPERTIES_UDP_PORT = 50063  # Bin Properties port
        self.header_size = 22 + 64  # Header size from document
        
        self.read_file()
       
    def read_file(self):
        pass
    
    def extract_header(data):
        data_ = data[22:] # Skip the first 22 bytes
        header_size = 64
        header = data_[24:header_size] # Extract the header
        offsets = header[:24]
        imag_offset, real_offset, range_gate_offset, doppler_bin_offset, rx_channel_offset, chirp_type_offset = np.frombuffer(offsets, dtype=dt_uint32)
        indexes = header[24:30]
        indexes = np.frombuffer(indexes, dtype=dt_uint16)
        range_gates, first_range_gate, doppler_bins = indexes
        rx_channels, chirp_types, element_size, element_type = header[30:34]
        
        # Return np array to be saved
        return np.array([imag_offset, real_offset, range_gate_offset, doppler_bin_offset, rx_channel_offset, chirp_type_offset, range_gates, first_range_gate, doppler_bins, rx_channels, chirp_types, element_size, element_type])
    
    def extract_radar_cube_data(data):
        pass
    
    def extract_properties(data):
        frame_counter  = np.frombuffer(data[14:18], dtype=dt_uint32)[0]
        properties = data[22+24:]
        properties = np.frombuffer(properties, dtype=dt_float32)

        # Return np array to be saved
        return (frame_counter, properties)