import socket
import numpy as np
import matplotlib.pyplot as plt
import struct

# Binary data format:
# 1. Header (64 bytes)
# 2. Payload (variable size)

# Data fields greater than 1 Byte are transmitted in Big Endian.
dt_uint32 = np.dtype(np.uint32).newbyteorder('>')
dt_uint16 = np.dtype(np.uint16).newbyteorder('>')

UDP_PACKET_SIZE = 65535   # Max size of UDP packet (MTU)

def open_socket(ip, port):
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    print("Socket opened")
    return sock

def close_socket(sock):
    sock.close()
    print("Socket closed")

# Skip incomplete frames
def skip_incomplete_frame(sock):
    data, addr = sock.recvfrom(UDP_PACKET_SIZE)
    while data[18] != 0x01:
        print(".", end="")
        data,addr = sock.recvfrom(UDP_PACKET_SIZE)
    print()
    return data, addr

def extract_header(data):
    data_ = data[22:] # Skip the first 22 bytes
    header_size = 64
    header = data_[24:header_size] # Extract the header
    offsets = header[:24]
    imag_offset, real_offset, range_gate_offset, doppler_bin_offset, rx_channel_offset, chirp_type_offset = np.frombuffer(offsets, dtype=dt_uint32)
    print("Imaginary offset: ", imag_offset)
    print("Real offset: ", real_offset)
    print("Range gate offset: ", range_gate_offset)
    print("Doppler bin offset: ", doppler_bin_offset)
    print("Rx channel offset: ", rx_channel_offset)
    print("Chirp type offset: ", chirp_type_offset)
    indexes = header[24:30]
    indexes = np.frombuffer(indexes, dtype=dt_uint16)
    range_gates, first_range_gate, doppler_bins = indexes
    print("Range gates: ", range_gates)
    print("First range gate: ", first_range_gate)
    print("Doppler bins: ", doppler_bins)
    rx_channels, chirp_types, element_size, element_type = header[30:34]
    print("Rx channels: ", rx_channels)
    print("Chirp types: ", chirp_types)
    print("Element size: ", element_size)
    print("Element type: ", element_type)
    print("matrix size: ", range_gates, doppler_bins, rx_channels, chirp_types, "=", range_gates*doppler_bins*rx_channels*chirp_types)
    print("----------------")
    return {"imag_offset": imag_offset, "real_offset": real_offset, "range_gate_offset": range_gate_offset, "doppler_bin_offset": doppler_bin_offset, "rx_channel_offset": rx_channel_offset, "chirp_type_offset": chirp_type_offset, "range_gates": range_gates, "first_range_gate": first_range_gate, "doppler_bins": doppler_bins, "rx_channels": rx_channels, "chirp_types": chirp_types, "element_size": element_size, "element_type": element_type}

def read_element_by_offset(radar_cube_data, offset, real_offset, imag_offset):
    real = radar_cube_data[offset+real_offset:offset+real_offset+2]
    imag = radar_cube_data[offset+imag_offset:offset+imag_offset+2]
    real = np.frombuffer(real, dtype=dt_uint16)
    imag = np.frombuffer(imag, dtype=dt_uint16)
    return complex(real, imag)

def radar_cube(radar_cube_data, fields):
    radar_cube = np.zeros((fields["range_gates"], fields["doppler_bins"], fields["rx_channels"], fields["chirp_types"]), dtype=np.complex64)
    for rg in range(fields["range_gates"]):
        for db in range(fields["doppler_bins"]):
            for rx in range(fields["rx_channels"]):
                for seq in range(fields["chirp_types"]):
                    offset = rg * fields["range_gate_offset"] + db * fields["doppler_bin_offset"] + rx * fields["rx_channel_offset"] + seq * fields["chirp_type_offset"]
                    radar_cube[rg, db, rx, seq] = read_element_by_offset(radar_cube_data, offset, fields["real_offset"], fields["imag_offset"])

    return radar_cube



def read_radar_RDCS(sock):
    i = 0
    # Receive data
    while i<2:
        data, addr = skip_incomplete_frame(sock)
        fields = extract_header(data)
        radar_cube_data = bytearray(data[22+64:])
        frame_number = np.frombuffer(data[14:18], dtype=dt_uint32)
        print("=================Received data from: ", addr, "\tData size: ", len(data), "\tFrame number: ", frame_number,"=================")
        # Extract field values from the header

        j = 0
        # while data[18] != 0x02:
        while False:
        # for j in range(2):
            data, addr = sock.recvfrom(UDP_PACKET_SIZE)
            message_counter = np.frombuffer(data[10:12], dtype=dt_uint16)
            fn = np.frombuffer(data[14:18], dtype=dt_uint32)
            # print("Received data from: ", addr, "\tData size: ", len(data), "\tMessage counter: ", message_counter) 
            # print(f"{message_counter}/", end="")
            print(f"{data[18]}{frame_number==fn}/", end="")
            radar_cube_data.extend(data[22+64:])
            j+=1
        print()
        print(f"{j} messages received")
        print("Data size: ", len(radar_cube_data))
        i+=1
        # with open("data.bin", "wb") as f:
        #     f.write(data)
    


        # # Extract field values from the header
        # print(data)
        # data_ = data[22:] # Skip the first 22 bytes
        # header_size = 64
        # header = data_[24:header_size] # Extract the header
        # offsets = header[:24]
        # imag_offset, real_offset, range_gate_offset, doppler_bin_offset, rx_channel_offset, chirp_type_offset = np.frombuffer(offsets, dtype=dt_uint32)
        # print("Imaginary offset: ", imag_offset)
        # print("Real offset: ", real_offset)
        # print("Range gate offset: ", range_gate_offset)
        # print("Doppler bin offset: ", doppler_bin_offset)
        # print("Rx channel offset: ", rx_channel_offset)
        # print("Chirp type offset: ", chirp_type_offset)
        # indexes = header[24:30]
        # indexes = np.frombuffer(indexes, dtype=dt_uint16)
        # indexes = indexes.tolist()
        # # indexes = [int(index) for index in indexes]
        # range_gates, first_range_gate, doppler_bins = indexes
        # print("Range gates: ", range_gates)
        # print("First range gate: ", first_range_gate)
        # print("Doppler bins: ", doppler_bins)
        # rx_channels = header[30]
        # # rx_channels = np.frombuffer(rx_channels, dtype=np.int8)
        # print("Rx channels: ", rx_channels)




    # # Paramètres du Radar Cube (à adapter en fonction de votre capteur)
    # N_RANGE_GATES = 200    # Nombre de cellules de distance
    # N_DOPPLER_BINS = 128   # Nombre de cellules Doppler
    # ELEMENT_SIZE = 2       # Taille d'un élément (en bytes)
    # REAL_OFFSET = 2        # Décalage mémoire pour la partie réelle
    # IMAG_OFFSET = 0        # Décalage mémoire pour la partie imaginaire
    
    # # Taille attendue du payload
    # expected_payload_size = N_RANGE_GATES * N_DOPPLER_BINS * ELEMENT_SIZE * 2  # (complexe = réel + imaginaire)
    
    # # Vérification de la taille du payload
    # if len(payload) < expected_payload_size:
    #     print(f"⚠️ Taille du payload reçue ({len(payload)} bytes) inférieure à la taille attendue ({expected_payload_size} bytes).")
    #     print("🔄 Vérifiez la configuration du capteur et assurez-vous que les dimensions sont correctes.")
    #     exit(1)
    
    # # 🟢 Conversion des données brutes en matrice complexe
    # radar_cube = np.zeros((N_RANGE_GATES, N_DOPPLER_BINS), dtype=np.complex64)
    
    # for rg in range(N_RANGE_GATES):
    #     for db in range(N_DOPPLER_BINS):
    #         offset = (rg * N_DOPPLER_BINS + db) * ELEMENT_SIZE * 2  # Position dans le payload
    #         # Vérification que l'offset ne dépasse pas la taille du buffer
    #         if offset + REAL_OFFSET + 2 > len(payload) or offset + IMAG_OFFSET + 2 > len(payload):
    #             print(f"⚠️ Offset {offset} dépasse la taille du buffer ({len(payload)} bytes). Ignoré.")
    #             continue  # Ignore cette valeur et passe à la suivante
    #         # Extraction de la partie réelle et imaginaire
    #         real_part = struct.unpack_from(">h", payload, offset + REAL_OFFSET)[0]
    #         imag_part = struct.unpack_from(">h", payload, offset + IMAG_OFFSET)[0]
    #         # Stockage en tant que nombre complexe
    #         radar_cube[rg, db] = complex(real_part, imag_part)
    
    # # 🟢 Transformation FFT Doppler (axe 1)
    # range_doppler_map = np.fft.fftshift(np.fft.fft2(radar_cube), axes=1)
    # range_doppler_map = 20 * np.log10(np.abs(range_doppler_map) + 1e-10)  # Ajout d'un epsilon pour éviter log(0)
    
    # # 🟢 Affichage du graphique Range-Doppler
    # plt.figure(figsize=(10, 6))
    # plt.imshow(range_doppler_map, aspect='auto', cmap='jet', extent=[-N_DOPPLER_BINS//2, N_DOPPLER_BINS//2, 0, N_RANGE_GATES])
    # plt.colorbar(label='Amplitude (dB)')
    # plt.xlabel('Fréquence Doppler (Hz)')
    # plt.ylabel('Distance (m)')
    # plt.title('Graphique Range-Doppler')
    # plt.show()


if __name__ == "__main__":
    ip = "192.168.11.17"
    port = 50005
    sock = open_socket(ip, port)
    read_radar_RDCS(sock)
    close_socket(sock)
