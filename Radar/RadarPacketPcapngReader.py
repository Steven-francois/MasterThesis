from Radar.RadarPacketReader import RadarPacketReader
from scapy.all import rdpcap, UDP, IP
import numpy as np

# Data fields greater than 1 Byte are transmitted in Big Endian.
dt_uint64 = np.dtype(np.uint64).newbyteorder('>')
dt_uint32 = np.dtype(np.uint32).newbyteorder('>')
dt_uint16 = np.dtype(np.uint16).newbyteorder('>')
dt_int16 = np.dtype(np.int16).newbyteorder('>')
dt_float32 = np.dtype(np.float32).newbyteorder('>')

class RadarPacketPcapngReader(RadarPacketReader):
    def __init__(self, filename):
        super().__init__(filename)
        self.pcap = rdpcap(filename)
        self.rdc_packets = []
        self.properties_packets = []
        self._read_packets()

    def _read_packets(self):
        for pkt in self.pcap:
            if UDP in pkt and pkt[UDP].dport == self.RADAR_CUBE_UDP_PORT and pkt[IP].src == self.IP_SOURCE and pkt[IP].dst == self.IP_DEST :
                self.rdc_packets.append(pkt)

        for pkt in self.pcap:
            if UDP in pkt and pkt[UDP].dport == self.BIN_PROPERTIES_UDP_PORT and pkt[IP].src == self.IP_SOURCE and pkt[IP].dst == self.IP_DEST:
                self.properties_packets.append(pkt)
        
        self.rdc_packets.sort(key=lambda x: (np.frombuffer(x[UDP].payload.load[14:18], dtype=dt_uint32), np.frombuffer(x[UDP].payload.load[10:12], dtype=dt_uint16)))
        self.properties_packets.sort(key=lambda x: (np.frombuffer(x[UDP].payload.load[14:18], dtype=dt_uint32), np.frombuffer(x[UDP].payload.load[10:12], dtype=dt_uint16)))
        while self.rdc_packets[0][UDP].payload.load[18] != 0x01:
            self.rdc_packets.pop(0)

    def __iter__(self):
        return iter(self.packets)

    def __len__(self):
        return len(self.packets)

    def __getitem__(self, key):
        return self.packets[key]

    def __str__(self):
        return f"RadarPacketPcapngReader({self.filename})"

    def __repr__(self):
        return str(self)