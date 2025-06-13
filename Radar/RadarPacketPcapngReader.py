from Radar.RadarPacketReader import RadarPacketReader, tqdm
from scapy.all import rdpcap, UDP, IP, PcapReader
import numpy as np


class RadarPacketPcapngReader(RadarPacketReader):
    def __init__(self, filename=None, output_name="rdc_file"):
        super().__init__(filename, output_name)
        self.rdc_packets = []
        self.properties_packets = []
        self.__indexes_rcd = []
        self.__indexes_prop = []
        self.__start_msg = {}
        self.__end_msg = {}
    
    def read(self, filename=None):
        if filename is None:
            filename = self.filename
        self.filename = filename
        print(f"Reading {self.filename}")
        # self.pcap = rdpcap(self.filename, count=100000)
        self.__read_packets()
        self.allocate_memory()
        self.create_files()
        self.__extract_rcd()
        self.fill_properties()
        self.save()

    def __read_packets(self):
        with PcapReader(self.filename) as self.pcap:
            def filter_packets(pkt):
                if UDP in pkt and pkt[UDP].dport == self.RADAR_CUBE_UDP_PORT and pkt[IP].src == self.IP_SOURCE and pkt[IP].dst == self.IP_DEST :
                    i=len(self.rdc_packets)
                    self.rdc_packets.append(pkt)
                    payload = pkt[UDP].payload.load
                    frame, msg_nb, code = (
                        int.from_bytes(payload[14:18], byteorder="big"),
                        int.from_bytes(payload[10:12], byteorder="big"),
                        int.from_bytes(payload[18:19], byteorder="big"),
                    )
                    if code == 0x01:  # First packet of frame
                        self.__start_msg[frame] = (msg_nb, i)
                    elif code == 0x02:  # Last packet of frame
                        self.__end_msg[frame] = (msg_nb, i)
                    
                    self.__indexes_rcd.append((frame, msg_nb, code, i))
                elif UDP in pkt and pkt[UDP].dport == self.BIN_PROPERTIES_UDP_PORT and pkt[IP].src == self.IP_SOURCE and pkt[IP].dst == self.IP_DEST:
                    self.properties_packets.append(pkt)
                    payload = pkt[UDP].payload.load
                    frame, msg_nb = (
                        int.from_bytes(payload[14:18], byteorder="big"),
                        int.from_bytes(payload[10:12], byteorder="big"),
                    )
                    self.__indexes_prop.append((frame, msg_nb, len(self.properties_packets)-1))
            self.progress_bar(self.pcap, filter_packets, "Filtering packets")
            for frame in self.__start_msg:      # Check for missing end messages
                if frame not in self.__end_msg and frame+1 in self.__start_msg:
                    end_msg_nb, end_idx = self.__start_msg[frame+1]
                    end_msg_nb = (end_msg_nb - 1) % 0x10000
                    self.__end_msg[frame] = end_msg_nb
                    self.rdc_packets.append(self.rdc_packets[end_idx].copy())
                    self.rdc_packets[-1][UDP].payload.load = bytes(b'\x00'*1446)
                    self.__indexes_rcd.append((frame, end_msg_nb, 0x02, len(self.rdc_packets)-1))
            for frame in self.__end_msg:        # Check for missing start messages
                if frame not in self.__start_msg and frame-1 in self.__end_msg:
                    start_msg_nb, start_idx = self.__end_msg[frame-1]
                    start_msg_nb = (start_msg_nb + 1) % 0x10000
                    self.__start_msg[frame] = start_msg_nb
                    self.rdc_packets.append(self.rdc_packets[start_idx].copy())
                    self.rdc_packets[-1][UDP].payload.load = bytes(b'\x00'*1458)
                    self.__indexes_rcd.append((frame, start_msg_nb, 0x01, len(self.rdc_packets)-1))
            def get_rcd_sort_key(index):
                frame, msg_nb, _, _ = index
                start = self.__start_msg.get(frame, [None, None])[0]
                if start is None:
                    start = self.__end_msg.get(frame-1, [None, None])[0]
                    if start is None:
                        start = 0
                    else:
                        start+=1
                corrected_msg_nb = (msg_nb - start) % 0x10000
                return (frame, corrected_msg_nb)
            def get_prop_sort_key(index):
                frame, msg_nb, _ = index
                return (frame, msg_nb)
            print("start", len(self.__start_msg), "end", len(self.__end_msg))
            self.__indexes_rcd.sort(key=get_rcd_sort_key)
            self.__indexes_prop.sort(key=get_prop_sort_key)
            first_valid_packet = 0
            while self.__indexes_rcd[first_valid_packet][2] != 0x01:
                first_valid_packet += 1
            self.__indexes_rcd = self.__indexes_rcd[first_valid_packet:]
                
                
            first_payload = bytes(self.rdc_packets[0][UDP].payload)
            self.fields = self.extract_header(first_payload)
            
            # Define radar cube dimensions
            self.N_RANGE_GATES       = int(self.fields[6])    # 200
            self.N_DOPPLER_BINS      = int(self.fields[8])    # 128
            self.N_RX_CHANNELS       = int(self.fields[9])    # 8
            self.N_CHIRP_TYPES       = int(self.fields[10])   # 2
            self.FIRST_RANGE_GATE    = int(self.fields[7])    # 0
            print(f"Range Gates: {self.N_RANGE_GATES}, Doppler Bins: {self.N_DOPPLER_BINS}, RX Channels: {self.N_RX_CHANNELS}, Chirp Types: {self.N_CHIRP_TYPES}")

    def __extract_rcd(self):
        i = 0
        self.nb_frames = 0
        nb_packets = len(self.rdc_packets)
        nb_missing_frames = 0
        with tqdm(total=nb_packets, desc="Extracting RDC") as pbar:
            while i<nb_packets:
                while i<nb_packets and self.__indexes_rcd[i][2]!= 0x01:    # Skip invalid 1st packet of frame
                    i+=1; pbar.update(1)
                radar_cube_data = bytearray(self.rdc_packets[self.__indexes_rcd[i][-1]][UDP].payload.load[22+64:])
                current_frame = self.__indexes_rcd[i][0]
                previous_message_id = self.__indexes_rcd[i][1]
                timestamp = np.frombuffer(self.rdc_packets[self.__indexes_rcd[i][-1]][UDP].payload.load[30:38], dtype=self.dt_uint64)[0]
                time = float(self.rdc_packets[self.__indexes_rcd[i][-1]].time) #received time
                i+=1; pbar.update(1)
                nb_packets_found = 1
                while i<nb_packets and self.__indexes_rcd[i][2] != 0x01 and self.__indexes_rcd[i][0] == current_frame:
                    current_message_id = self.__indexes_rcd[i][1]
                    if current_message_id > previous_message_id + 1:
                        for _ in range(previous_message_id+1, current_message_id):
                            radar_cube_data.extend(bytes(b'\x00'*1436))
                    previous_message_id = current_message_id
                    radar_cube_data.extend(self.rdc_packets[self.__indexes_rcd[i][-1]][UDP].payload.load[22:])
                    time = min(time, float(self.rdc_packets[self.__indexes_rcd[i][-1]].time))
                    i+=1; pbar.update(1); nb_packets_found+=1
                radar_cube_data = self.process_radar_cube_data(radar_cube_data)
                nb_frame  = self.nb_frames % self.max_nb_frames
                self.timestamps[nb_frame] = timestamp
                self.time[nb_frame] = time
                if radar_cube_data is not None:
                    self.radar_cube_datas[nb_frame] = radar_cube_data
                else:
                    self.radar_cube_datas[nb_frame] = np.zeros((self.N_RANGE_GATES, self.N_DOPPLER_BINS, self.N_RX_CHANNELS, self.N_CHIRP_TYPES), dtype=np.complex64)
                    nb_missing_frames += 1
                self.nb_frames += 1
                
                if self.nb_frames % self.max_nb_frames == 0:
                    self.save_radar_cube_data()
        if self.nb_frames % self.max_nb_frames != 0:
            self.save_radar_cube_data()
        print(f"\nProcessed Radar Cube Data: {self.nb_frames} frames with {nb_missing_frames} missing frames")
            
    
        
    def fill_properties(self):
        self.all_properties = []
        all_frames_counter = []
        for i in tqdm(range(len(self.properties_packets)), desc="Extracting Properties"):
            property_payload = bytes(self.properties_packets[i][UDP].payload)
            frame_counter, properties = self.extract_properties(property_payload)
            all_frames_counter.append(frame_counter)
            self.all_properties.append(properties)

        # Fill missing properties
        for i in tqdm(range(len(self.all_properties)-1), desc="Filling missing properties"):
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
                
        if (self.nb_frames != len(self.all_properties)):
            print(f"Radar Cube Data: {self.nb_frames}, Properties: {len(self.all_properties)}")
            for i in range(self.nb_frames-len(self.all_properties)):
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
    

if __name__ == "__main__":
    import os
    test_folder = "Data//radar_test"
    os.makedirs(test_folder, exist_ok=True)
    test_rdc_file = os.path.join(test_folder, "radar_cube_data")
    
    # Example usage
    # reader = RadarPacketPcapngReader("D://Muse//static//DATA_20250514_140309//20250514_140309//radar_eth_20250514_140309.pcapng")
    reader = RadarPacketPcapngReader("D://Steven//MasterThesis//Fusion//static//DATA_20250514_140309//20250514_140309//radar_eth_20250514_140309.pcapng", test_rdc_file)
    reader.read()