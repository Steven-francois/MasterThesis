import numpy as np
from tqdm import tqdm
from scapy.all import rdpcap, PcapReader
from bitstruct import unpack
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.cluster import DBSCAN


class RadarCanReader():
    class CanTargetsHeader:
        def __init__(self, frame1=None, frame2=None, frame3=None, time=None):
            if frame1 is not None and frame2 is not None and frame3 is not None and time is not None:
                _, self.center_freq_idx, self.sweep_idx, self.tx_antenna_idx, self.ack_set, self.nof_targets, self.cycle_counter, _, self.cycle_duration = frame1
                timestamp_seconds = frame2[-1]
                timestamp_fraction = frame3[-1]
                self.timestamp = timestamp_seconds + (timestamp_fraction / (2**32-1))
                self.real_time = time 
                # self.timestamp = int(self.timestamp * 1e9)  # Convert to nanoseconds
        @classmethod
        def from_values(cls, center_freq_idx, sweep_idx, tx_antenna_idx, ack_set, nof_targets, cycle_counter, cycle_duration, timestamp, real_time):
            obj = cls()
            obj.center_freq_idx = int(center_freq_idx)
            obj.sweep_idx = int(sweep_idx)
            obj.tx_antenna_idx = int(tx_antenna_idx)
            obj.ack_set = int(ack_set)
            obj.nof_targets = int(nof_targets)
            obj.cycle_counter = int(cycle_counter)
            obj.cycle_duration = int(cycle_duration)
            obj.timestamp = timestamp
            obj.real_time = real_time
            return obj
        def __str__(self):
            return f"Header: center_freq_idx={self.center_freq_idx}, sweep_idx={self.sweep_idx}, tx_antenna_idx={self.tx_antenna_idx}, ack_set={self.ack_set}, nof_targets={self.nof_targets}, cycle_counter={self.cycle_counter}, cycle_duration={self.cycle_duration}, timestamp={self.timestamp}, real_time={self.real_time}"
        def npy(self):
            return np.array([self.center_freq_idx, self.sweep_idx, self.tx_antenna_idx, self.ack_set, self.nof_targets, self.cycle_counter, self.cycle_duration, self.timestamp, self.real_time])
    class CanTargetsData:
        def __init__(self, frame1=None, frame2=None):
            if frame1 is not None and frame2 is not None:
                # Unpack the data from the frames
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
        @classmethod
        def from_values(cls, speed_radial, az_angle, t_range, el_angle, noise, power, rcs):
            obj = cls()
            obj.speed_radial = speed_radial
            obj.az_angle = az_angle
            obj.t_range = t_range
            obj.el_angle = el_angle
            obj.noise = noise
            obj.power = power
            obj.rcs = rcs
            return obj
        def __str__(self):
            return f"Data: speed_radial={self.speed_radial}, az_angle={self.az_angle}, t_range={self.t_range}, el_angle={self.el_angle}, noise={self.noise}, power={self.power}, rcs={self.rcs}"
        def npy(self):
            return np.array([self.speed_radial, self.az_angle, self.t_range, self.el_angle, self.noise, self.power, self.rcs])
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
        can_id, can_dlc = unpack("u32u8", bytes(packet))
        mode_signal = unpack("u2", bytes(packet)[8:][::-1])[0]
        return can_id == 0x0400 and can_dlc == 0x8 and mode_signal == 0x00
    def __startingData(self, packet):
        can_id, can_dlc = unpack("u32u8", bytes(packet))
        mode_signal = unpack("u63u1", bytes(packet)[8:][::-1])[1]
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
        print(f"Number of packets: {len(self.can_packets)}")
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
            can_targets_header = self.CanTargetsHeader(frame1, frame2, frame3, self.can_packets[i].time)
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
                if i+2>=len(self.can_packets):
                    break
            self.can_targets.append(can_target)

    def save(self, filename):
        with open(filename, "wb") as f:
            for can_target in self.can_targets:
                f.write(str(can_target).encode())
                for target_data in can_target.targets_data:
                    f.write(str(target_data).encode())
        print(f"Saved {len(self.can_targets)} CAN targets to {filename}")
        print(f"Number of CAN targets: {len(self.can_targets)}")
    def save_npy(self, filename):
        with open(filename, "wb") as f:
            np.save(f, len(self.can_targets))
            for can_target in self.can_targets:
                can_target.targets_header.nof_targets = len(can_target.targets_data)
                targets_header = can_target.targets_header.npy()
                np.save(f, targets_header)
                for target_data in can_target.targets_data:
                    targets_data = target_data.npy()
                    np.save(f, targets_data)
        print(f"Saved {len(self.can_targets)} CAN targets to {filename}")
        print(f"Number of CAN targets: {len(self.can_targets)}")

    def load(self, filename):
        with open(filename, "rb") as f:
            data = f.read()
            can_targets = data.split(b"CAN_TARGETS")
            for can_target in can_targets:
                if len(can_target) > 0:
                    can_target = can_target.decode()
                    targets_header = can_target.split("Targets Header: ")[1].split("\n")[0]
                    targets_data = can_target.split("Targets Data: ")[1].split("\n")[0]
                    self.can_targets.append(self.CanTargets(targets_header, targets_data))
        print(f"Loaded {len(self.can_targets)} CAN targets from {filename}")
    def load_npy(self, filename):
        with open(filename, "rb") as f:
            nb_can_targets = np.load(f, allow_pickle=True)
            self.can_targets = []
            for _ in tqdm(range(nb_can_targets), desc="Loading CAN targets"):
                targets_header = np.load(f, allow_pickle=True)
                can_target = self.CanTargets(self.CanTargetsHeader.from_values(*targets_header), [])
                for _ in range(can_target.targets_header.nof_targets):
                    target_data = self.CanTargetsData.from_values(*np.load(f, allow_pickle=True))
                    can_target.add_target_data(target_data)
                self.can_targets.append(can_target)
        print(f"Loaded {len(self.can_targets)} CAN targets from {filename}")
        print(f"Number of CAN targets: {len(self.can_targets)}")

    def cluster_with_dbscan(self,
                            eps_space: float,
                            eps_speed: float,
                            min_samples: int = 2):
        """
        For each CAN frame, cluster targets by 3D-space (x,y,z) AND radial speed.
        - eps_space: max meters difference along each Cartesian axis
        - eps_speed: max m/s difference in radial speed
        - min_samples: DBSCAN core-point threshold
        """
        for frame in tqdm(self.can_targets, desc="Clustering CAN targets"):
            # Skip empty frames
            data = frame.targets_data
            if not data:
                continue

            # Convert spherical (range, az, el) to Cartesian (x,y,z)
            ranges = np.array([t.t_range       for t in data])
            azs    = np.deg2rad([t.az_angle    for t in data])
            els    = np.deg2rad([t.el_angle    for t in data])
            xs     = ranges * np.cos(els) * np.sin(azs)
            ys     = ranges * np.cos(els) * np.cos(azs)
            zs     = ranges * np.sin(els)
            speeds = np.array([t.speed_radial for t in data])

            # Build (N,4) array and normalize each column by its eps
            features = np.stack([xs, ys, zs, speeds], axis=1)
            scale    = np.array([eps_space, eps_space, eps_space, eps_speed])
            normed   = features / scale  # now each axis has “unit threshold” = 1.0

            # DBSCAN with Chebyshev so that max‐diff ≤1 in each dim ---
            db     = DBSCAN(eps=1.0,
                            min_samples=min_samples,
                            metric='euclidean',
                            n_jobs=-1)
            labels = db.fit_predict(normed)

            # Build averaged clusters
            new_targets = []
            for lbl in sorted(set(labels)):
                # get indices for this label (including -1 if you want to keep noise)
                if lbl == -1:
                    idxs = np.where(labels == -1)[0]
                    for i in idxs:
                        new_targets.append(data[i])
                    continue
                idxs = np.where(labels == lbl)[0]

                # compute mean of each field
                mean_vals = {
                    'speed_radial': np.mean(speeds[idxs]),
                    'az_angle'    : np.rad2deg(np.arctan2(
                                        np.mean(xs[idxs]),
                                        np.mean(ys[idxs])
                                    )),  # back‐compute avg azimuth
                    't_range'     : np.mean(ranges[idxs]),
                    'el_angle'    : np.rad2deg(np.arctan2(
                                        np.mean(zs[idxs]),
                                        np.linalg.norm([np.mean(xs[idxs]), np.mean(ys[idxs])])
                                    )),
                    'noise'       : np.mean([data[i].noise  for i in idxs]),
                    'power'       : np.mean([data[i].power  for i in idxs]),
                    'rcs'         : np.mean([data[i].rcs    for i in idxs]),
                }

                avg = RadarCanReader.CanTargetsData.from_values(**mean_vals)
                new_targets.append(avg)

            # Replace frame’s target list & update header
            frame.targets_data = new_targets
            frame.targets_header.nof_targets = len(new_targets)


if __name__ == "__main__":
    reader = RadarCanReader()
    # reader.read("Radar/can/can_test2.pcapng")
    reader.read("Fusion/DATA_20250417_153534/20250417_154040/radar_can_20250417_154040.pcapng")
    reader.save_npy("Fusion/data/can_data_41.npy")
    # reader.load("Radar/can/can_test2.bin")
    # reader.save_npy("Radar/data/can_data_50.npy")
    # reader.load_npy("Radar/data/can_data_50.npy")
    # reader.load_npy("Radar/can/can_test2.npy")
    print(f"Number of CAN targets: {len(reader.can_targets)}")
    target_idxs = []
    for i in range(len(reader.can_targets)):
        can_target = reader.can_targets[i]
        targets_data = can_target.targets_data
        if len(targets_data) > 0:
            target_idxs.append(i)
    print(f"Number of CAN targets with data: {len(target_idxs)}")
    fig, ax = plt.subplots()
    plt.xlim(-200,200)
    plt.ylim(0,96)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Range (m)")
    scat = ax.scatter([], [], marker='o', color='b')
    
    def animate(i):
        # plt.clf()
        can_target = reader.can_targets[i]
        targets_data = can_target.targets_data
        x = []
        y = []
        values = []
        for target in targets_data:
            x.append(target.t_range)
            y.append(target.speed_radial)
            values.append(target.rcs)
        data = np.stack([y,x]).T
        scat.set_offsets(data)
        scat.set_array(np.array(values))
        plt.title(f"Frame {i}")
        return fig, ax
    ani = FuncAnimation(fig, animate, frames=target_idxs, interval=62)
    plt.show()