import numpy as np
from tqdm import tqdm
from scapy.all import rdpcap, PcapReader
from bitstruct import unpack
from bitstring import BitStream, BitArray


class RadarCanReader():
    def __init__(self):
        self.can_packets = []
        
    def read(self, filename):
        print(f"Reading {filename}")
        packets = rdpcap(filename)
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
    reader.read("Radar/can/can_test.pcapng")
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
