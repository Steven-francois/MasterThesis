from Radar.RadarPacketReader import RadarPacketReader, tqdm
import numpy as np
from pyshark import FileCapture

class RadarPacketPcapReader(RadarPacketReader):
    def __init__(self, filename, rdc_file="rdc_file.npy"):
        super().__init__(filename, rdc_file)
        self.pcap = FileCapture(filename)
        self.rdc_packets = []
        self.properties_packets = []
        print(f"Reading {filename}")
        self._read_packets()
        self.extract_radar_cube_data()
        self.fill_properties()
        
    def _udp_payload(self, pkt):
        return bytearray.fromhex(pkt.data.data)
        

    def _read_packets(self):
        def filter_packets(pkt):
            print(pkt)
            exit()
            if "udp" in pkt and int(pkt["udp"].dstport) == self.BIN_PROPERTIES_UDP_PORT and pkt["ip"].src == self.IP_SOURCE and pkt["ip"].dst == self.IP_DEST:
                self.properties_packets.append(pkt)
            elif "udp" in pkt and int(pkt["udp"].dstport) == self.RADAR_CUBE_UDP_PORT and pkt["ip"].src == self.IP_SOURCE and pkt["ip"].dst == self.IP_DEST:
                self.rdc_packets.append(pkt)

        self.progress_bar(self.pcap, filter_packets, "Filtering packets")
    
        def get_sort_key(pkt):
            payload = self._udp_payload(pkt)
            return (
                int.from_bytes(payload[14:18], byteorder="big"),
                int.from_bytes(payload[10:12], byteorder="big"),
            )
        self.rdc_packets.sort(key=get_sort_key)
        self.properties_packets.sort(key=get_sort_key)
        first_valid_packet = 0
        while self._udp_payload(self.rdc_packets[0])[18] != 0x01:
            first_valid_packet += 1
        self.rdc_packets = self.rdc_packets[first_valid_packet:]
        first_payload = bytes(self._udp_payload(self.rdc_packets[0]))
        self.fields = self.extract_header(first_payload)
        
        # Define radar cube dimensions
        self.N_RANGE_GATES       = int(self.fields[6])    # 200
        self.N_DOPPLER_BINS      = int(self.fields[8])    # 128
        self.N_RX_CHANNELS       = int(self.fields[9])    # 8
        self.N_CHIRP_TYPES       = int(self.fields[10])   # 2
        self.FIRST_RANGE_GATE    = int(self.fields[7])    # 0
        print(f"Range Gates: {self.N_RANGE_GATES}, Doppler Bins: {self.N_DOPPLER_BINS}, RX Channels: {self.N_RX_CHANNELS}, Chirp Types: {self.N_CHIRP_TYPES}")

            
    def extract_radar_cube_data(self):
        self.radar_cube_datas = []
        self.timestamps = []
        
        i = 0
        nb_packets = len(self.rdc_packets)
        with tqdm(total=nb_packets, desc="Extracting RDC") as pbar:
            while i<nb_packets:
                radar_cube_data = bytearray(self._udp_payload(self.rdc_packets[i])[22+64:])
                current_frame = self._udp_payload(self.rdc_packets[i])[14:18]
                timestamp = np.frombuffer(self._udp_payload(self.rdc_packets[i])[30:38], dtype=self.dt_uint64)[0]
                i+=1; pbar.update(1)
                while i<nb_packets and self._udp_payload(self.rdc_packets[i])[18] != 0x01 and self._udp_payload(self.rdc_packets[i])[14:18] == current_frame:
                    radar_cube_data.extend(self._udp_payload(self.rdc_packets[i])[22:])
                    i+=1; pbar.update(1)
                radar_cube_data = self.process_radar_cube_data(radar_cube_data)
                if radar_cube_data is not None:
                    self.timestamps.append(timestamp)
                    self.radar_cube_datas.append(radar_cube_data)
                else:
                    self.timestamps.append(0)
                    self.radar_cube_datas.append(np.zeros((self.N_RANGE_GATES, self.N_DOPPLER_BINS, self.N_RX_CHANNELS, self.N_CHIRP_TYPES), dtype=np.complex64))
        self.nb_frames = len(self.radar_cube_datas)
        print(f"\nProcessed Radar Cube Data: {self.nb_frames} frames")
        
    def fill_properties(self):
        self.all_properties = []
        all_frames_counter = []
        for i in tqdm(range(len(self.properties_packets)), desc="Extracting Properties"):
            property_payload = bytes(self._udp_payload(self.properties_packets[i]))
            frame_counter, properties = self.extract_properties(property_payload)
            all_frames_counter.append(frame_counter)
            self.all_properties.append(properties)

        # Fill missing properties
        for i in tqdm(range(len(self.all_properties)-1), desc="Filling Properties"):
            current_frame, current_properties = all_frames_counter[i], self.all_properties[i]
            next_frame, next_properties = all_frames_counter[i+1], self.all_properties[i+1]
            # Check for missing frames
            if next_frame - current_frame > 1:
                nb_missing_frames = next_frame - current_frame - 1
                delta_properties = (next_properties - current_properties) / nb_missing_frames
                for j in range(current_frame+1, next_frame):
                    self.all_properties.insert(i+1, current_properties + delta_properties * (j - current_frame))
                    print(f"Missing frame: {j}")
        first_property_frame = all_frames_counter[0]
        first_properties = self.all_properties[0]
        print(f"First property frame :{first_property_frame}")
        
        first_radar_cube_frame = np.frombuffer(self._udp_payload(self.rdc_packets[0])[14:18], dtype=self.dt_uint32)[0]
        frame_diff = first_radar_cube_frame - first_property_frame
        print(f"First Radar Cube Frame: {first_radar_cube_frame}, First Property Frame: {first_property_frame}, Frame Diff: {frame_diff}")
        if first_property_frame < first_radar_cube_frame:
            self.all_properties = self.all_properties[frame_diff:]
        else:
            for i in range(frame_diff):
                self.all_properties.insert(0, first_properties)
                print(f"Missing property frame: {first_radar_cube_frame-i}")
        len_diff = len(self.radar_cube_datas) - len(self.all_properties)        
        print(f"Radar Cube Data: {len(self.radar_cube_datas)}, Properties: {len(self.all_properties)}")
        if (len_diff > 0):
            for i in range(len_diff):
                self.all_properties.insert(-1, self.all_properties[-1])
        else:
            self.all_properties = self.all_properties[:len(self.radar_cube_datas)]
        print(f"Filled Properties: {len(self.all_properties)}")
        

    def __iter__(self):
        return iter(self.radar_cube_datas)

    def __len__(self):
        return len(self.radar_cube_datas)

    def __getitem__(self, key):
        return self.radar_cube_datas[key]

    def __str__(self):
        return f"RadarPacketPcapngReader({self.filename})"

    def __repr__(self):
        return str(self)