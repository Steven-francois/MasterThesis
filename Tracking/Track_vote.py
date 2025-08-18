import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# --- Gating configuration (tunable) ---
# Base threshold and growth rate per second for time-since-update aware gating.
# Effective threshold = BASE * (1 + GROWTH_PER_SEC * track.time_since_update)
GATE_BASE_RADAR = 6.0
GATE_GROWTH_PER_SEC_RADAR = 50
GATE_BASE_LIDAR = 5.0
GATE_GROWTH_PER_SEC_LIDAR = 0.01

# --- Fusion configuration (tunable) ---
FUSE_WEIGHT_RADAR = 1.0
FUSE_WEIGHT_LIDAR = 1.0

class Track:
    count = 0

    def __init__(self, detection, ts, nb, cam_id=None):
        # Identity and bookkeeping
        self.id = Track.count
        Track.count += 1
        self.history = [detection]
        self.history_ts = [ts]
        self.nbs = [nb]
        self.time_since_update = 0.0
        self.last_update_time = ts
        self.history_cam_id = [cam_id]
        self.cam_id = cam_id

        # Store last known measurements
        self.radar_coord = detection[2] if len(detection) > 2 else None
        self.lidar_coord = detection[1] if len(detection) > 1 else None

        # Per-sensor prediction logs
        self.radar_predictions = []  # list of np.ndarray shape (2,)
        self.lidar_predictions = []  # list of np.ndarray shape (3,)

        # Initialize Radar Kalman filter (2D: position, velocity or similar)
        self.radar_kf = None
        if self.radar_coord is not None:
            kf_r = KalmanFilter(dim_x=2, dim_z=2)
            dt = 1.0
            kf_r.F = np.array([[1, dt],
                               [0, 1]])
            kf_r.H = np.array([[1, 0],
                               [0, 1]])
            kf_r.R *= 0.9
            kf_r.P *= 10.0
            kf_r.Q *= 0.1
            kf_r.x = np.array(self.radar_coord).reshape(2, 1)
            self.radar_kf = kf_r

        # Initialize LiDAR Kalman filter (6D state: x,vx,y,vy,z,vz with 3D measurements)
        self.lidar_kf = None
        if self.lidar_coord is not None:
            kf_l = KalmanFilter(dim_x=6, dim_z=3)
            dt = 1.0
            kf_l.F = np.array([[1, dt, 0,  0, 0,  0],
                               [0,  1, 0,  0, 0,  0],
                               [0,  0, 1, dt, 0,  0],
                               [0,  0, 0,  1, 0,  0],
                               [0,  0, 0,  0, 1, dt],
                               [0,  0, 0,  0, 0,  1]])
            kf_l.H = np.array([[1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0]])
            kf_l.R *= 0.9
            kf_l.P *= 10.0
            kf_l.Q *= 0.1
            # Initialize position components of state
            kf_l.x[::2] = np.array(self.lidar_coord).reshape(3, 1)
            self.lidar_kf = kf_l

    def _radar_transition_dt(self, dt: float):
        if self.radar_kf is not None:
            self.radar_kf.F = np.array([[1, dt],
                                        [0, 1]])

    def _lidar_transition_dt(self, dt: float):
        if self.lidar_kf is not None:
            self.lidar_kf.F = np.array([[1, dt, 0,  0, 0,  0],
                                        [0,  1, 0,  0, 0,  0],
                                        [0,  0, 1, dt, 0,  0],
                                        [0,  0, 0,  1, 0,  0],
                                        [0,  0, 0,  0, 1, dt],
                                        [0,  0, 0,  0, 0,  1]])

    def predict(self, new_timestamp, kalman=True):
        # Always use Kalman prediction per sensor; kalman flag kept for API compatibility
        dt = new_timestamp - self.last_update_time
        self.time_since_update = dt

        pred = {"timestamp": new_timestamp}

        if self.radar_kf is not None:
            self._radar_transition_dt(dt)
            self.radar_kf.predict()
            radar_pred = self.radar_kf.x.reshape(-1)
            pred["radar_coord"] = radar_pred
            self.radar_predictions.append(radar_pred)

        if self.lidar_kf is not None:
            self._lidar_transition_dt(dt)
            self.lidar_kf.predict()
            lidar_pred = self.lidar_kf.x[::2].reshape(-1)
            pred["lidar_coord"] = lidar_pred
            self.lidar_predictions.append(lidar_pred)

        return pred

    def update(self, detection, ts, nb, cam_id=None):
        # Bookkeeping
        self.history.append(detection)
        # kf_detection = [[], self.lidar_kf.x[::2].reshape(-1) if self.lidar_kf is not None else [], self.radar_kf.x.reshape(-1) if self.radar_kf is not None else []]
        # self.history.append(kf_detection)
        self.history_ts.append(ts)
        self.nbs.append(nb)
        self.time_since_update = 0.0
        self.last_update_time = ts
        if cam_id is not None:
            self.cam_id = cam_id
        self.history_cam_id.append(cam_id if cam_id is not None else -1)

        # Update radar KF if measurement available
        if len(detection) > 2 and detection[2] is not None and self.radar_kf is not None:
            self.radar_coord = detection[2]
            self.radar_kf.update(np.array(self.radar_coord).reshape(2, 1))

        # Update lidar KF if measurement available
        if len(detection) > 1 and detection[1] is not None and self.lidar_kf is not None:
            self.lidar_coord = detection[1]
            self.lidar_kf.update(np.array(self.lidar_coord).reshape(3, 1))

# --- Track management functions
def compute_cost(tracks, detections, lidar=False):
    cost = np.zeros((len(tracks), len(detections["targets_nb"])))
    for i, track in enumerate(tracks):
        pred = track.predict(detections["timestamp"], True)
        for j in range(len(detections["targets_nb"])):
            try:
                if lidar:
                    meas = detections["targets"][j][1]
                    pred_coord = pred.get("lidar_coord", None)
                else:
                    meas = detections["targets"][j][2]
                    pred_coord = pred.get("radar_coord", None)

                if pred_coord is None or meas is None:
                    cost[i, j] = np.inf
                else:
                    cost[i, j] = np.linalg.norm(np.array(pred_coord) - np.array(meas))
            except Exception:
                # Any shape/index issues -> disallow this pair
                cost[i, j] = np.inf
    return cost

def associate_tracks_and_detections(tracks, detections, max_age=5, lidar=False, fuse=False, cam_ids=None):
    """Associate tracks to detections using Hungarian algorithm.

    Args:
        tracks: list of Track* objects.
        detections: dict with keys 'timestamp', 'targets', 'targets_nb'.
        max_age: unused here; track aging is handled by caller.
        lidar: if True and fuse=False, use LiDAR cost; else use Radar cost.
        fuse: if True, fuse Radar and LiDAR costs into a single matrix.

    Returns:
        matches, unmatched_tracks, unmatched_detections
    """
    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections["targets_nb"])))

    C_r = compute_cost(tracks, detections, lidar=False)
    C_l = compute_cost(tracks, detections, lidar=True)
    if fuse:
        # C_f = C_r * FUSE_WEIGHT_RADAR + C_l * FUSE_WEIGHT_LIDAR
        C_f = np.zeros_like(C_r)
        for i in range(len(C_r)):
            for j in range(len(C_r[0])):
                C_f[i,j] = min(1.4*C_r[i,j], 0.6*C_l[i,j])
                # C_f[i,j] = min(C_r[i,j], C_l[i,j])
        row_ind, col_ind = linear_sum_assignment(C_f)
        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_detections = set(range(len(detections["targets_nb"])))

        if cam_ids is not None:
            for i, track in enumerate(tracks):
                for j in range(len(detections["targets_nb"])):
                    if len(cam_ids) > 0 and track.cam_id == cam_ids[j]:
                        # C_f[:, j] = 1000
                        # C_f[i, :] = 1000
                        C_f[i, j] = 0  # Apply additional weighting for same-camera matches
        for i, j in zip(row_ind, col_ind):
            gate = GATE_BASE_LIDAR * (1.0 + GATE_GROWTH_PER_SEC_LIDAR * tracks[i].time_since_update)
            if C_f[i, j] < gate:
                matches.append((i, j))
                unmatched_tracks.discard(i)
                unmatched_detections.discard(j)
        return matches, list(unmatched_tracks), list(unmatched_detections)

            

    else:
        if lidar:
            cost_matrix = C_l
        else:
            cost_matrix = C_r
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            if lidar:
                gate = GATE_BASE_LIDAR * (1.0 + GATE_GROWTH_PER_SEC_LIDAR * tracks[i].time_since_update)
            else:
                gate = GATE_BASE_RADAR * (1.0 + GATE_GROWTH_PER_SEC_RADAR * tracks[i].time_since_update)
            if cost_matrix[i, j] < gate:
                matches.append((i, j))
                unmatched_tracks.discard(i)
                unmatched_detections.discard(j)


    return matches, list(unmatched_tracks), list(unmatched_detections)

if __name__ == "__main__":
    import os
    import json
    from tqdm import tqdm
    
    nb = "x1_0"
    data_folder = f"Data/{nb}/"
    data_folder = f"D:/p_{nb}/"
    fusion_folder = os.path.join(data_folder, "fusion")
    tracks_folder = os.path.join(data_folder, "tracks")
    
    # tracking_interval = slice(300, 540)
    # tracking_interval = slice(1204, 1241)
    tracking_interval = slice(2500, 2600)
    # tracking_interval = slice(3228, 3300)
    # tracking_interval = slice(0, 450)
    # tracking_interval =  slice(3587, 3624)
    
    
    with open(os.path.join(fusion_folder, "targets.npy"), "rb") as f:
        num_frames = np.load(f, allow_pickle=True)
        fusion_frames = []
        CL_frames = []
        RL_frames = []
        frames = []
        tracks_id = []
        ts = []
        min_id = 1
        max_id = 0
        min_cam_id = 1000
        max_cam_id = 0
        cam_ids = []
        for i in range(num_frames):
            fusion_frames.append(np.load(f, allow_pickle=True))
            CL_frames.append(np.load(f, allow_pickle=True))
            RL_frames.append(np.load(f, allow_pickle=True))
            with open(os.path.join(fusion_folder, f"targets_{i}.json"), 'r') as json_file:
                frame_data = json.load(json_file)
                ts.append(frame_data["timestamp"])
                frames.append(frame_data)
                if i >= tracking_interval.start and i < tracking_interval.stop:
                    c_ids = np.array(frame_data["cam_info"]["ids"])
                    if len(fusion_frames[-1]) > 0 and len(c_ids) > 0:
                        c_ids = c_ids[fusion_frames[-1][:, 0]]
                    elif len(fusion_frames[-1]) == 0:
                        c_ids = c_ids[:0]
                    min_cam_id = int(min(min_cam_id, min(c_ids) if len(c_ids) > 0 else min_cam_id))
                    max_cam_id = int(max(max_cam_id, max(c_ids) if len(c_ids) > 0 else max_cam_id))
                    cam_ids.append(c_ids.tolist())
        # for i in range(tracking_interval.start, tracking_interval.stop):
        #     with open(os.path.join(tracks_folder, f"frame_{i:04d}.json"), 'r') as json_file:
        #         track_data = json.load(json_file)["tracks"]
        #         if len(track_data) > 0:
        #             min_id = min(min_id, min(track_data))
        #             max_id = max(max_id, max(track_data))
        #         tracks_id.append(track_data)

    
    # print(min_id, max_id)
    # gt_tracks = []
    # for i in range(min_id, max_id + 1):
    #     t = []
    #     for ids in tracks_id:
    #         if i in ids:
    #             t.append(ids.index(i))
    #         else:
    #             t.append(-1)
    #     print(f"Ground truth track {i}: {t}")
    #     gt_tracks.append(t)
    # print(f"Number of ground truth tracks: {len(gt_tracks)}")
    # print()

    
    print(min_cam_id, max_cam_id)
    cam_tracks = []
    cam_tracks_ts = []
    for i in range(min_cam_id, max_cam_id + 1):
        t = []
        c_ts = []
        for idx, ids in enumerate(cam_ids):
            if i in ids:
                det = ids.index(i)
                t.append(det)
                c_ts.append(frames[tracking_interval][idx]["timestamp"])
        print(f"Camera track {i}: {t}")
        cam_tracks.append(t)
        cam_tracks_ts.append(c_ts)

    tracks = []
    old_tracks = []
    max_age = 100

    for frame_idx, detections in tqdm(enumerate(frames[tracking_interval]), desc="Tracking Progress"):
        matches, unmatched_tracks, unmatched_dets = associate_tracks_and_detections(tracks, detections, max_age)
        
        # Update matched tracks
        for i, j in matches:
            tracks[i].update(detections["targets"][j],  detections["timestamp"], j)
            
        # Create new tracks for unmatched detections
        for j in unmatched_dets:
            new_track = Track(detections["targets"][j], detections["timestamp"], j)
            tracks.append(new_track)
            
        # Remove old tracks
        old_tracks.extend([track for track in tracks if track.time_since_update > max_age])
        tracks = [track for track in tracks if track.time_since_update <= max_age]
        
    # Plotting the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for track in tracks:
        coords = np.array([detection[2] for detection in track.history])
        preds = np.array(track.radar_predictions)
        # print(f"Track {track.id} history: {coords}")
        # print(f"Track {track.id} predictions: {preds}")
        plt.plot(coords[:, 0], coords[:, 1], marker='o', label=f'Track {track.id}')
        # plt.plot(preds[:, 0], preds[:, 1], linestyle='-', label=f'Predicted {track.id}') if len(preds) > 0 else None
    for track in old_tracks:
        coords = np.array([detection[2] for detection in track.history])
        preds = np.array(track.radar_predictions)
        plt.plot(coords[:, 0], coords[:, 1], marker='o', label=f'Track {track.id}')
        # plt.plot(preds[:, 0], preds[:, 1], linestyle='-', label=f'Predicted {track.id} (old)', alpha=0.5)
    plt.xlabel('range coordinate (m)')
    plt.ylabel('Doppler coordinate (m/s)')
    plt.title('Tracked Targets Over Time (Radar)')
    plt.legend()
    plt.grid()
    plt.show()

    tracks_lidar = []
    old_tracks_lidar = []
    max_age = 100
    Track.count = 0  # Reset track count for Lidar tracks
    with open(os.path.join(fusion_folder, "tracks_lidar.npy"), "wb") as f:
        ti = np.array(tracking_interval)
        np.save(f, ti)

        for frame_idx, detections in tqdm(enumerate(frames[tracking_interval]), desc="Tracking Progress"):
            matches, unmatched_tracks, unmatched_dets = associate_tracks_and_detections(tracks_lidar, detections, max_age, True)
            tracks_dets = np.zeros(Track.count, dtype=int)
            tracks_dets[:] = -1
            
            # Update matched tracks
            for i, j in matches:
                tracks_lidar[i].update(detections["targets"][j],  detections["timestamp"], j)
                tracks_dets[tracks_lidar[i].id] = j

            # Create new tracks for unmatched detections
            for j in unmatched_dets:
                new_track = Track(detections["targets"][j], detections["timestamp"], j)
                tracks_lidar.append(new_track)
                tracks_dets = np.append(tracks_dets, j)
                
            # Remove old tracks
            old_tracks_lidar.extend([track for track in tracks_lidar if track.time_since_update > max_age])
            tracks_lidar = [track for track in tracks_lidar if track.time_since_update <= max_age]
            # print(tracks_dets)
            np.save(f, tracks_dets)
        
    # Plotting the results
    # import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for track in tracks_lidar:
        coords = np.array([detection[1] for detection in track.history])
        preds = np.array(track.lidar_predictions)
        # print(f"Track {track.id} history: {coords} ts: {track.history_ts}")
        # print(f"Track {track.id} predictions: {preds}")
        plt.plot(coords[:, 0], coords[:, 2], marker='o', label=f'Track {track.id}')
        # plt.plot(preds[:, 0], preds[:, 2], linestyle='-', label=f'Predicted {track.id}')
    for track in old_tracks_lidar:
        coords = np.array([detection[1] for detection in track.history])
        preds = np.array(track.lidar_predictions)
        plt.plot(coords[:, 0], coords[:, 2], marker='o', label=f'Track {track.id}')
        # plt.plot(preds[:, 0], preds[:, 2], linestyle='-', label=f'Predicted {track.id} (old)', alpha=0.5)
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Tracked Targets Over Time (LiDAR)')
    plt.xlim(0, 130)
    plt.ylim(-1, 4)
    plt.legend()
    plt.grid()
    plt.savefig("tracked_targets_lidar.png", transparent=True)
    plt.show()


    # Fusion
    tracks_fusion = []
    old_tracks_fusion = []
    max_age = 100
    Track.count = 0  # Reset track count for Fusion tracks

    for frame_idx, detections in tqdm(enumerate(frames[tracking_interval]), desc="Tracking Progress"):
        matches, unmatched_tracks, unmatched_dets = associate_tracks_and_detections(tracks_fusion, detections, max_age, False, True, cam_ids[frame_idx] if len(cam_ids) > 0 else None)

        # Update matched tracks
        for i, j in matches:
            tracks_fusion[i].update(detections["targets"][j], detections["timestamp"], j, cam_ids[frame_idx][j] if len(cam_ids[frame_idx]) > 0 else None)

        # Create new tracks for unmatched detections
        for j in unmatched_dets:
            new_track = Track(detections["targets"][j], detections["timestamp"], j, cam_ids[frame_idx][j] if len(cam_ids[frame_idx]) > 0 else None)
            tracks_fusion.append(new_track)

        # Remove old tracks
        old_tracks_fusion.extend([track for track in tracks_fusion if track.time_since_update > max_age])
        tracks_fusion = [track for track in tracks_fusion if track.time_since_update <= max_age]
    plt.figure(figsize=(10, 6))
    for track in tracks_fusion:
        coords = np.array([detection[1] for detection in track.history])
        preds = np.array(track.lidar_predictions)
        # print(f"Track {track.id} history: {coords} ts: {track.history_ts}")
        # print(f"Track {track.id} predictions: {preds}")
        plt.plot(coords[:, 0], coords[:, 2], marker='o', label=f'Track {track.id}')
        # plt.plot(preds[:, 0], preds[:, 2], linestyle='-', label=f'Predicted {track.id}')
    for track in old_tracks_fusion:
        coords = np.array([detection[1] for detection in track.history])
        preds = np.array(track.lidar_predictions)
        plt.plot(coords[:, 0], coords[:, 2], marker='o', label=f'Track {track.id}')
        # plt.plot(preds[:, 0], preds[:, 2], linestyle='-', label=f'Predicted {track.id} (old)', alpha=0.5)
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.ylim(5, -2)
    plt.title('Tracked Targets Over Time (Fusion)')
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(figsize=(10, 6))
    for track in tracks_fusion:
        coords = np.array([detection[2] for detection in track.history])
        preds = np.array(track.radar_predictions)
        # print(f"Track {track.id} history: {coords}")
        # print(f"Track {track.id} predictions: {preds}")
        plt.plot(coords[:, 0], coords[:, 1], marker='o', label=f'Track {track.id}')
        # plt.plot(preds[:, 0], preds[:, 1], linestyle='-', label=f'Predicted {track.id}') if len(preds) > 0 else None
    for track in old_tracks_fusion:
        coords = np.array([detection[2] for detection in track.history])
        preds = np.array(track.radar_predictions)
        plt.plot(coords[:, 0], coords[:, 1], marker='o', label=f'Track {track.id}')
        # plt.plot(preds[:, 0], preds[:, 1], linestyle='-', label=f'Predicted {track.id} (old)', alpha=0.5)
    plt.xlabel('Range coordinate (m)')
    plt.ylabel('Doppler coordinate (m/s)')
    plt.title('Tracked Targets Over Time (Fusion)')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting selected targets for each track
    # for i, track in enumerate(gt_tracks):
    #     plt.plot(np.array(ts[tracking_interval]) + 0.03, track, marker='o', label=f'Track_gt {i} NBs')
    for track in tracks_fusion:
        plt.plot(np.array(track.history_ts) + 0.02, track.nbs, marker='o', label=f'Track_fusion {track.id} NBs')
    for track in old_tracks_fusion:
        plt.plot(np.array(track.history_ts) + 0.02, track.nbs, marker='o', label=f'Track_fusion {track.id} NBs (old)', alpha=0.5)
    for track in tracks_lidar:
        plt.plot(np.array(track.history_ts) + 0.01, track.nbs, marker='o', label=f'Track_lidar {track.id} NBs')
    for track in old_tracks_lidar:
        plt.plot(np.array(track.history_ts) + 0.01, track.nbs, marker='o', label=f'Track_lidar {track.id} NBs (old)', alpha=0.5)
    for track in tracks:
        plt.plot(track.history_ts, track.nbs, marker='o', label=f'Track {track.id} NBs')
    for track in old_tracks:
        plt.plot(track.history_ts, track.nbs, marker='o', label=f'Track {track.id} NBs (old)', alpha=0.5)
    frames_ts = [frame["timestamp"] for frame in frames[tracking_interval]]
    frames_nbs = [len(frame["targets_nb"]) for frame in frames[tracking_interval]]
    # plt.plot(frames_ts, frames_nbs, marker='x', label='Number of targets in frame')
    plt.xlabel('Timestamp')
    plt.ylabel('Target number')
    plt.title('Target Numbers Over Time')
    plt.legend()
    plt.grid()
    plt.show()