# from Radar.RadarPacketPcapReader import RadarPacketPcapReader as RadarPacketReader
from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as RadarPacketReader

def test_RadarPacketReader():
    # Test the RadarPacketPcapngReader class
    rdc_file = "Fusion/data/radar_cube_data_05.npy"
    rdc_reader = RadarPacketReader("Fusion/captures/radar_20250227_171252.pcapng", rdc_file)
    # rdc_reader = RadarPacketReader("Radar/captures/radar_log_04.pcapng", rdc_file)
    # rdc_reader.save()
    
    
test_RadarPacketReader()