import numpy as np
from tqdm import tqdm
from scapy.all import rdpcap, PcapReader
from bitstruct import unpack


class RadarCanReader():
    class CanTargetsHeader:
        def __init__(self, header):
            self.header = header
            self.can_id = header.can_id


    class CanTargets:
        def __init__(self, targets_header, targets_data=[]):
            self.targets_header = targets_header
            self.targets_data = targets_data
        
        def add_target(self, target):
            self.targets_data.append(target)
    def __init__(self):
        self.can_packets = []

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
            decoded = unpack("u2u2u2u2u1u8u32u3u12", bytes(self.can_packets[i])[8:][::-1])
            mode_signal, center_freq_idx, sweep_idx, tx_antenna_idx, ack_set, nof_targets, cycle_counter, _, cycle_duration = decoded
            print(decoded)
            if mode_signal != 0x00:
                raise Exception("Invalid mode signal: Fisrt frame Header")
            decoded = unpack("u2u30u32", bytes(self.can_packets[i+1])[8:][::-1])
            mode_signal, _, timestamps_seconds = decoded
            print(decoded)
            if mode_signal != 0x01:
                raise Exception("Invalid mode signal: Second frame Header")
            decoded = unpack("u2u30u32", bytes(self.can_packets[i+2])[8:][::-1])
            mode_signal, _, timestamps_fraction = decoded
            print(decoded)
            if mode_signal != 0x02:
                raise Exception("Invalid mode signal: Third frame Header")
            i+=3
            if i>=len(self.can_packets):
                break
            # Parse data
            while self.__startingData(self.can_packets[i]):
                print("----Data----")
                decoded = unpack("u13u12u7u10u8u13u1", bytes(self.can_packets[i])[8:][::-1])
                _, speed_radial, _, az_angle, _, t_range, mode_signal = decoded
                print(decoded)
                if mode_signal != 0x0:
                    raise Exception("Invalid mode signal: Fisrt frame Data")
                decoded = unpack("u17u10u12u8u8u8u1", bytes(self.can_packets[i+1])[8:][::-1])
                _, el_angle, _, noise, power, rcs, mode_signal = decoded
                print(decoded)
                if mode_signal != 0x1:
                    raise Exception("Invalid mode signal: Second frame Data")
                i+=2
                if i>=len(self.can_packets):
                    break
            print("----end----")
        exit()
        for packet in packets:
            # Frame ID
            # ...
            b = bytes(packet.payload)
            b  = b[8:]
            # print("===data===", b.hex())
            mode = unpack("u2u2u2u2", b[-1:])[0]
            if mode == 0:
                print(unpack("u2u2u2u2u1u8u32u3u12", b[::-1]))
                # print("mode", mode, "0x00")
                duration = b[:2][::-1]
                duration = unpack("u4u12", duration)[1]
                print("cycle duration", duration, "?=", 0.054976/0.000064)
                counter = b[1:6][::-1]
                counter = unpack("u1u32", counter)
                print("cycle counter", counter)
                targets = b[5:7][::-1]
                targets = unpack("u1u8u7", targets)
                print("nof targets", targets)
                print(unpack("u2u2u2u2", b[7:]))
            
                
if __name__ == "__main__":
    reader = RadarCanReader()
    reader.read("Radar/can/can_test2.pcapng")
    print(f"Number of CAN packets: {len(reader.can_packets)}")
    
    # def extract_last_bits(hex_string):
    #     """ Extracts the last 2 bits of a CAN message. """
    #     raw_bytes = bytes.fromhex(hex_string)  # Convert hex to bytes
    #     last_byte = raw_bytes[-1]  # Get the last byte
    #     last_two_bits = (last_byte >> 6) & 0b11  # Extract last 2 bits
    #     return last_two_bits

    # # CAN messages
    # messages = [
    #     "00040000080000005b83866d00008010",  # Expected: 0
    #     "0004000008000000d50d000000000040",  # Expected: 1
    #     "000400000800000085e464aa00000080",  # Expected: 2
    # ]

    # # Process messages
    # for msg in messages:
    #     last_bits = extract_last_bits(msg)
    #     print(f"Message: {msg} → Last 2 Bits: {last_bits}")
