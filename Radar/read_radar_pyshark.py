import numpy as np
import matplotlib.pyplot as plt
import struct
import pyshark

def extract_udp_packets(pcap_file, ip, port):
    # Open the pcap file
    cap = pyshark.FileCapture(pcap_file, display_filter=f'ip.dst == {ip} && udp.port == {port} and udp', use_json=True, include_raw=True)
    return cap


def main():
    pcap_file = "radar_log.pcapng"
    ip = "192.168.11.17"
    port = 50005

    cap = extract_udp_packets(pcap_file, ip, port)
    i=0
    for pkt in cap:
        print(type(pkt.udp.payload))
        print(len(pkt.frame_raw.value))
        print(pkt.get_raw_packet()[0])
        # with open("packet.bin", "wb") as f:
        #     f.write(pkt.get_raw_packet())
        print(pkt.udp.payload)
        # print(pkt.udp)
        if i == 3:
            break
        i+=1

if __name__ == "__main__":
    main()