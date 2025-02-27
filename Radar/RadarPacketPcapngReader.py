from Radar.RadarPacketReader import RadarPacketReader, tqdm
from scapy.all import rdpcap, UDP, IP
import numpy as np


class RadarPacketPcapngReader(RadarPacketReader):
    def __init__(self, filename, rdc_file="rdc_file.npy"):
        super().__init__(filename, rdc_file)
        self.pcap = rdpcap(filename)
        self.rdc_packets = []
        self.properties_packets = []
        print(f"Reading {filename}")
        self._read_packets()
        self.extract_radar_cube_data()
        self.fill_properties()
        

    def _read_packets(self):
        def filter_packets(pkt):
            # print(pkt)
            # print(pkt.summary())
            # print(pkt.haslayer("UDP"))
            # print(UDP in pkt)
            # # print(pkt[UDP])
            # # print(pkt[UDP].dport)
            # print(IP in pkt)
            # # print(pkt[IP])
            # # print(pkt[IP].src)
            # # print(pkt[IP].dst)
            # # exit()
            if UDP in pkt and pkt[UDP].dport == self.RADAR_CUBE_UDP_PORT and pkt[IP].src == self.IP_SOURCE and pkt[IP].dst == self.IP_DEST :
                self.rdc_packets.append(pkt)
            elif UDP in pkt and pkt[UDP].dport == self.BIN_PROPERTIES_UDP_PORT and pkt[IP].src == self.IP_SOURCE and pkt[IP].dst == self.IP_DEST:
                self.properties_packets.append(pkt)
        self.progress_bar(self.pcap, filter_packets, "Filtering packets")
        # for pkt in self.pcap:
        #     if UDP in pkt and pkt[UDP].dport == self.RADAR_CUBE_UDP_PORT and pkt[IP].src == self.IP_SOURCE and pkt[IP].dst == self.IP_DEST :
        #         self.rdc_packets.append(pkt)
        #     elif UDP in pkt and pkt[UDP].dport == self.BIN_PROPERTIES_UDP_PORT and pkt[IP].src == self.IP_SOURCE and pkt[IP].dst == self.IP_DEST:
        #         self.properties_packets.append(pkt)
        def get_sort_key(pkt):
            payload = pkt[UDP].payload.load
            return (
                int.from_bytes(payload[14:18], byteorder="big"),
                int.from_bytes(payload[10:12], byteorder="big"),
            )
        self.rdc_packets.sort(key=get_sort_key)
        self.properties_packets.sort(key=get_sort_key)
        first_valid_packet = 0
        while self.rdc_packets[0][UDP].payload.load[18] != 0x01:
            first_valid_packet += 1
        self.rdc_packets = self.rdc_packets[first_valid_packet:]
        first_payload = bytes(self.rdc_packets[0][UDP].payload)
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
                while i<nb_packets and self.rdc_packets[i][UDP].payload.load[18] != 0x01:    # Skip invalid 1st packet of frame
                    # print(str(self.rdc_packets[i][UDP].payload.load[18]), end="")
                    i+=1; pbar.update(1)
                
                print(len(self.rdc_packets[i][UDP].payload.load[22:]))
                radar_cube_data = bytearray(self.rdc_packets[i][UDP].payload.load[22+64:])
                current_frame = self.rdc_packets[i][UDP].payload.load[14:18]
                timestamp = np.frombuffer(self.rdc_packets[i][UDP].payload.load[30:38], dtype=self.dt_uint64)[0]
                i+=1; pbar.update(1)
                nb_packets_found = 1
                while i<nb_packets and self.rdc_packets[i][UDP].payload.load[18] != 0x01 and self.rdc_packets[i][UDP].payload.load[14:18] == current_frame:
                    print(len(self.rdc_packets[i][UDP].payload.load[22:]))
                    radar_cube_data.extend(self.rdc_packets[i][UDP].payload.load[22:])
                    # print(str(self.rdc_packets[i][UDP].payload.load[18]), end="")
                    i+=1; pbar.update(1); nb_packets_found+=1
                exit()
                print(f"Frame: {current_frame}, Packets: {nb_packets_found}")
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
        for i in range(len(self.properties_packets)):
            property_payload = bytes(self.properties_packets[i][UDP].payload)
            frame_counter, properties = self.extract_properties(property_payload)
            all_frames_counter.append(frame_counter)
            self.all_properties.append(properties)

        # Fill missing properties
        for i in range(len(self.all_properties)-1):
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
        
        first_radar_cube_frame = np.frombuffer(self.rdc_packets[0][UDP].payload.load[14:18], dtype=self.dt_uint32)[0]
        frame_diff = first_radar_cube_frame - first_property_frame
        print(f"First Radar Cube Frame: {first_radar_cube_frame}, First Property Frame: {first_property_frame}, Frame Diff: {frame_diff}")
        if first_property_frame < first_radar_cube_frame:
            self.all_properties = self.all_properties[frame_diff:]
        else:
            for i in range(frame_diff):
                self.all_properties.insert(0, first_properties)
                print(f"Missing property frame: {first_radar_cube_frame-i}")
                
        if (len(self.radar_cube_datas) != len(self.all_properties)):
            print(f"Radar Cube Data: {len(self.radar_cube_datas)}, Properties: {len(self.all_properties)}")
            for i in range(len(self.radar_cube_datas)-len(self.all_properties)):
                self.all_properties.insert(-1, self.all_properties[-1])
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