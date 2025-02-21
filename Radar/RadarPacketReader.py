class RadarPacketReader:
    def __init__(self, file_path):
        self.file_path = file_path
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
        pass
    
    def extract_radar_cube_data(data):
        pass
    
    def extract_properties(data):
        pass