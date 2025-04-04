import numpy as np
from tqdm import tqdm
from scapy.all import rdpcap, PcapReader
from bitstruct import unpack


class RadarCanReader():
    class CanTargetsHeader:
        def __init__(self, frame1, frame2, frame3):
            _, self.center_freq_idx, self.sweep_idx, self.tx_antenna_idx, self.ack_set, self.nof_targets, self.cycle_counter, _, self.cycle_duration = frame1
            timestamp_seconds = frame2[-1]
            timestamp_fraction = frame3[-1]
            self.timestamp = timestamp_seconds + (timestamp_fraction / 2**32-1)
            # self.timestamp = int(self.timestamp * 1e9)  # Convert to nanoseconds
        def __str__(self):
            return f"Header: center_freq_idx={self.center_freq_idx}, sweep_idx={self.sweep_idx}, tx_antenna_idx={self.tx_antenna_idx}, ack_set={self.ack_set}, nof_targets={self.nof_targets}, cycle_counter={self.cycle_counter}, cycle_duration={self.cycle_duration}, timestamp={self.timestamp}"
    class CanTargetsData:
        def __init__(self, frame1, frame2):
            _, self.speed_radial, _, self.az_angle, _, self.t_range, _ = frame1
            _, self.el_angle, _, self.noise, self.power, self.rcs, _ = frame2
            self.t_range = self.t_range * 0.04
            self.az_angle = (self.az_angle-511) * 0.16
            self.speed_radial = (self.speed_radial-2992) *0.04
            self.rcs = (self.rcs-75) * 0.02
            self.noise = self.noise * 0.5
            self.el_angle = (self.el_angle-511) * 0.04
            # Convert to numpy arrays for easier manipulation
            self.speed_radial = np.array(self.speed_radial)
            self.az_angle = np.array(self.az_angle)
            self.t_range = np.array(self.t_range)
            self.el_angle = np.array(self.el_angle)
            self.noise = np.array(self.noise)
            self.power = np.array(self.power)
            self.rcs = np.array(self.rcs)
        def __str__(self):
            return f"Data: speed_radial={self.speed_radial}, az_angle={self.az_angle}, t_range={self.t_range}, el_angle={self.el_angle}, noise={self.noise}, power={self.power}, rcs={self.rcs}"
    class CanTargets:
        def __init__(self, targets_header, targets_data=[]):
            self.targets_header = targets_header
            self.targets_data = targets_data
        
        def add_target_data(self, target_data):
            self.targets_data.append(target_data)
            
        def __str__(self):
            return f"Targets Header: {self.targets_header}\nNb of Targets Data: {len(self.targets_data)}"
            
    def __init__(self):
        self.can_targets = []

    def __startingHeader(self, packet):
        mode_signal, _, can_dlc, can_id = unpack("u2u62u32u32", bytes(packet.payload)[::-1])
        return can_id == 0x0400 and can_dlc == 0x8 and mode_signal == 0x00
    def __startingData(self, packet):
        _, mode_signal, can_dlc, can_id = unpack("u63u1u32u32", bytes(packet.payload)[::-1])
        return can_id > 0x0400 and can_dlc == 0x8 and mode_signal == 0x0
        
        
    def read(self, filename):
        print(f"Reading {filename}")
        packets = rdpcap(filename)
        first_valid_packet = 0
        while self.__startingHeader(packets[first_valid_packet]) == False:
            first_valid_packet += 1

        packets = packets[first_valid_packet:]
        self.can_packets = packets
        i = 0
        while i + 2 < len(self.can_packets):
            # Parse Header
            frame1 = unpack("u2u2u2u2u1u8u32u3u12", bytes(self.can_packets[i])[8:][::-1])
            mode_signal, center_freq_idx, sweep_idx, tx_antenna_idx, ack_set, nof_targets, cycle_counter, _, cycle_duration = frame1
            if mode_signal != 0x00:
                raise Exception("Invalid mode signal: Fisrt frame Header")
            frame2 = unpack("u2u30u32", bytes(self.can_packets[i+1])[8:][::-1])
            mode_signal, _, timestamps_seconds = frame2
            if mode_signal != 0x01:
                raise Exception("Invalid mode signal: Second frame Header")
            frame3 = unpack("u2u30u32", bytes(self.can_packets[i+2])[8:][::-1])
            mode_signal, _, timestamps_fraction = frame3
            if mode_signal != 0x02:
                raise Exception("Invalid mode signal: Third frame Header")
            can_targets_header = self.CanTargetsHeader(frame1, frame2, frame3)
            can_target = self.CanTargets(can_targets_header, [])
            
            i+=3
            if i>=len(self.can_packets):
                break
            # Parse data
            while self.__startingData(self.can_packets[i]):
                frame1 = unpack("u13u12u7u10u8u13u1", bytes(self.can_packets[i])[8:][::-1])
                _, speed_radial, _, az_angle, _, t_range, mode_signal = frame1
                if mode_signal != 0x0:
                    raise Exception("Invalid mode signal: Fisrt frame Data")
                frame2 = unpack("u17u10u12u8u8u8u1", bytes(self.can_packets[i+1])[8:][::-1])
                _, el_angle, _, noise, power, rcs, mode_signal = frame2
                if mode_signal != 0x1:
                    raise Exception("Invalid mode signal: Second frame Data")
                can_targets_data = self.CanTargetsData(frame1, frame2)
                can_target.add_target_data(can_targets_data)
                i+=2
                if i>=len(self.can_packets):
                    break
            self.can_targets.append(can_target)

if __name__ == "__main__":
    reader = RadarCanReader()
    reader.read("Radar/can/can_test2.pcapng")
    print(f"Number of CAN targets: {len(reader.can_targets)}")
    
    for i, can_target in enumerate(reader.can_targets):