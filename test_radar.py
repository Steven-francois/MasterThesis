from scapy.all import *
import numpy as np

dt_uint64 = np.dtype(np.uint64).newbyteorder('>')
dt_uint32 = np.dtype(np.uint32).newbyteorder('>')
dt_uint16 = np.dtype(np.uint16).newbyteorder('>')
dt_int16 = np.dtype(np.int16).newbyteorder('>')
dt_float32 = np.dtype(np.float32).newbyteorder('>')

def extract_properties(data):
    frame_counter  = np.frombuffer(data[14:18], dtype=dt_uint32)[0]
    properties = data[22+24:]
    properties = np.frombuffer(properties, dtype=dt_float32)

    # Return np array to be saved
    return (frame_counter, properties)

capture = sniff(filter="udp and dst port 50063", count=10)
print(capture)

for packet in capture:
    frame, prop = extract_properties(packet[UDP].payload.load)
    print(frame, prop, 128/2*prop[0], 200*prop[1])
