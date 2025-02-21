from scapy.all import rdpcap, UDP, IP

class RadarPacketPcapngReader(RadarPacketReader):
    def __init__(self, filename):
        self.filename = filename
        self.pcap = rdpcap(filename)
        self.rdc_packets = []
        self._read_packets()

    def _read_packets(self):
        for pkt in self.pcap:
            if UDP in pkt and pkt[UDP].dport == self.RADAR_CUBE_UDP_PORT and pkt[IP].src == self.IP_SOURCE and pkt[IP].dst == self.IP_DEST :
                self.rdc_packets.append(pkt)

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