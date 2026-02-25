import cv2
import numpy as np

def compute_derotated_optical_flow(
    prev_frame,
    curr_frame,
    gyro_delta,
    K,
    prev_pts=None,
    min_features=150,
    min_distance=10,  # Built-in spatial distribution!
    flip_axes=False
):
    
    # --- 1. Seeding with Shi-Tomasi ---
    if prev_pts is None or len(prev_pts) == 0:
        prev_pts = cv2.goodFeaturesToTrack(
            prev_frame, 
            maxCorners=300, 
            qualityLevel=0.03, 
            minDistance=min_distance,
            blockSize=7
        )

    if prev_pts is None: 
        return np.empty((0, 2)), np.empty((0, 2)), None

    # --- 2. Track ---
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_frame, curr_frame, prev_pts, None,
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    if curr_pts is None: return np.empty((0, 2)), np.empty((0, 2)), None

    status = status.ravel() == 1
    good_prev = prev_pts.reshape(-1, 2)[status]
    good_curr = curr_pts.reshape(-1, 2)[status]

    # --- 3. Reseed if feature count low ---
    if len(good_prev) < min_features:
        # Create a blank mask the size of the image
        mask = 255 * np.ones_like(prev_frame)
        
        # Black out the areas around our EXISTING points so we don't detect new ones there
        for p in good_prev:
            cv2.circle(mask, (int(p[0]), int(p[1])), min_distance, 0, -1)
            
        # Detect new points only in the white areas of the mask
        new_pts = cv2.goodFeaturesToTrack(
            prev_frame, 
            maxCorners=300 - len(good_prev), 
            qualityLevel=0.03, 
            minDistance=min_distance,
            blockSize=7,
            mask=mask
        )

        if new_pts is not None:
            curr_new, st_new, _ = cv2.calcOpticalFlowPyrLK(
                prev_frame, curr_frame, new_pts, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            if curr_new is not None:
                st_new = st_new.ravel() == 1
                good_prev = np.vstack([good_prev, new_pts.reshape(-1, 2)[st_new]])
                good_curr = np.vstack([good_curr, curr_new.reshape(-1, 2)[st_new]])

    if len(good_prev) == 0: return np.empty((0, 2)), np.empty((0, 2)), None

    # --- 4. Derotation math ---
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    dp, dq, dr = gyro_delta 
    
    if flip_axes:
        wx, wy, wz = -dq, -dr, dp 
    else:
        wx, wy, wz = dq, dr, dp 

    u, v = good_prev[:, 0], good_prev[:, 1]
    u_new, v_new = good_curr[:, 0], good_curr[:, 1]

    du_meas, dv_meas = u_new - u, v_new - v
    x, y = (u - cx) / fx, (v - cy) / fy

    dx_rot = x * y * wx - (1 + x**2) * wy + y * wz
    dy_rot = (1 + y**2) * wx - x * y * wy - x * wz

    du_trans = du_meas - (dx_rot * fx)
    dv_trans = dv_meas - (dy_rot * fy)

    translational_flow = np.stack([du_trans, dv_trans], axis=1)
    next_prev_pts = good_curr.reshape(-1, 1, 2)

    return good_prev, translational_flow, next_prev_pts
