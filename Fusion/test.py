# from Radar.RadarPacketPcapReader import RadarPacketPcapReader as RadarPacketReader
from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as RadarPacketReader
import tracemalloc
import linecache
import os

def test_RadarPacketReader():
    # Test the RadarPacketPcapngReader class
    rdc_file = "Fusion/data/radar_cube_data_20"
    rdc_reader = RadarPacketReader("Fusion/captures/radar_20250305_160743.pcapng", rdc_file)
    # rdc_reader = RadarPacketReader("Radar/captures/radar_log_21.pcapng", rdc_file)
    rdc_reader.read()
    # rdc_reader.save()

#StackOverflow: https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python
def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


# tracemalloc.start()
test_RadarPacketReader()
# snapshot = tracemalloc.take_snapshot()
# display_top(snapshot)
