import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
MARKER_IDS = [0, 1, 2, 3, 4]  
MARKER_SIZE = 200  # pixels
CAPTURE_INTERVAL = 10  # #sec to take image
CALIBRATION_DIR = "calibration_images"
CHESSBOARD_SIZE = (11, 7)  # chessboard size
SQUARE_SIZE = 2.1  # cm
NUM_CALIBRATION_IMGS = 15

# "fisheye" distortion fix; save in npz file if cam position stays the same
CALIBRATION_FILE = "camera_calibration.npz"

# already generated; this code is for if they're lost/corrupted
def generate_aruco_markers():
    for marker_id in MARKER_IDS:
        marker = aruco.generateImageMarker(ARUCO_DICT, marker_id, MARKER_SIZE)
        cv2.imwrite(f"marker_{marker_id}.png", marker)
    print("Markers saved as PNG files.")

# if calibration file already exists, use it; otherwise use new images
def calibrate_camera():
    if os.path.exists(CALIBRATION_FILE):
        print("Loading existing camera calibration...")
        data = np.load(CALIBRATION_FILE)
        camera_matrix, dist_coeffs = data['camera_matrix'], data['dist_coeffs']
        return camera_matrix, dist_coeffs
    
    print("Camera calibration!")
    print(f"Please show a {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} chessboard pattern to the camera")
    print(f"Need to capture {NUM_CALIBRATION_IMGS} images")
    
    # create directory for calibration images if it doesn't exist
    if not os.path.exists(CALIBRATION_DIR):
        os.makedirs(CALIBRATION_DIR)
    
    # prep object points
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE
    objpoints = []  # 3D points
    imgpoints = []  # 2D points in image
    cap = cv2.VideoCapture(0)    
    img_count = 0
    while img_count < NUM_CALIBRATION_IMGS:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        display_frame = frame.copy()
        
        if ret:
            # refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # draw corners (debugging?)
            cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners2, ret)
            
            # instructions
            cv2.putText(display_frame, f"Image {img_count+1}/{NUM_CALIBRATION_IMGS}: Press SPACE to capture", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Chessboard not detected", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Camera Calibration", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # quit if 'q' pressed
        if key == ord('q'):
            break
        # when spacebar pressed, take pic and store points
        elif key == ord(' ') and ret: 
            img_name = os.path.join(CALIBRATION_DIR, f"calibration_{img_count}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Saved {img_name}")
            objpoints.append(objp)
            imgpoints.append(corners2)
            img_count += 1
            time.sleep(0.5)
    
    cap.release()
    cv2.destroyAllWindows()
    
    if img_count < NUM_CALIBRATION_IMGS:
        print(f"Warning: Only captured {img_count}/{NUM_CALIBRATION_IMGS} images.") 
    print("Calibrating camera, please wait...")
    
    # calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    if ret:
        print("Camera calibration successful!")
        print(f"Camera Matrix:\n{camera_matrix}")
        print(f"Distortion Coefficients:\n{dist_coeffs}")
        
        # save calibration results
        np.savez(CALIBRATION_FILE, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print(f"Calibration saved to {CALIBRATION_FILE}")
        
        return camera_matrix, dist_coeffs
    else:
        print("Camera calibration failed.")
        return None, None

def undistort_image(frame, camera_matrix, dist_coeffs):
    if camera_matrix is None or dist_coeffs is None:
        return frame
    
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    # crop image
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted

def detect_aruco_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT)
    marker_corners = {}
    marker_centers = {}
    robot_corners = None

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            corner_points = corners[i].reshape(4, 2)
            marker_corners[marker_id] = corner_points
            
            # center
            center_x = int(np.mean(corner_points[:, 0]))
            center_y = int(np.mean(corner_points[:, 1]))
            marker_centers[marker_id] = (center_x, center_y)

            # draw outline, ID
            cv2.polylines(frame, [corner_points.astype(int)], True, (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {marker_id}", (center_x + 10, center_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # if robot, store corners
            if marker_id == 0:
                robot_corners = [(int(x), int(y)) for x, y in corner_points]
                
                # label each corner 0-3
                for j, (x, y) in enumerate(robot_corners):
                    cv2.putText(frame, f"{j}", (x, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return frame, marker_corners, marker_centers, robot_corners

def get_crop_corners(marker_corners):
    if not all(marker_id in marker_corners for marker_id in [1, 2, 3, 4]):
        return None

    top_left = marker_corners[1][2]  # bottom right of marker 1
    top_right = marker_corners[2][3]  # bottom left of marker 2
    bottom_left = marker_corners[3][1]  # top right of marker 3
    bottom_right = marker_corners[4][0]  # top left of marker 4
    
    # convert to points
    x1, y1 = int(top_left[0]), int(top_left[1])
    x2, y2 = int(bottom_right[0]), int(bottom_right[1])
    # check
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    return (x1, y1, x2, y2)

def main():
    generate_aruco_markers()
    
    # calibrate
    camera_matrix, dist_coeffs = calibrate_camera()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    print("Press 'q' to quit. Press 'c' to recalibrate camera.")
    last_capture_time = time.time()
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # undistort
        if camera_matrix is not None and dist_coeffs is not None:
            frame = undistort_image(frame, camera_matrix, dist_coeffs)

        # detect markers
        frame_with_markers, marker_corners, marker_centers, robot_corners = detect_aruco_markers(frame)
        if robot_corners is not None:
            print(f"Robot marker (ID 0) corners: {robot_corners}")
        
        # crop frame
        crop_coords = None
        if len(marker_corners) >= 4:  # We need at least the 4 corner markers
            crop_coords = get_crop_corners(marker_corners)
        
        # display frame
        display_frame = frame_with_markers.copy()
        
        if crop_coords is not None:
            x1, y1, x2, y2 = crop_coords
            
            # crop frame
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cropped_frame = frame[y1:y2, x1:x2]
            
            # detect robot
            if cropped_frame.size > 0:
                cropped_with_robot, _, _, robot_corners_cropped = detect_aruco_markers(cropped_frame)

                # display corners
                if robot_corners_cropped is not None:
                    print(f"Robot marker (ID 0) corners in cropped image: {robot_corners_cropped}")
                current_time = time.time()
                if current_time - last_capture_time >= CAPTURE_INTERVAL:
                    filename = f"cropped_desk_{counter}.jpg"
                    cv2.imwrite(filename, cropped_with_robot)
                    print(f"Saved {filename}")
                    if 0 in marker_centers:
                        robot_x, robot_y = marker_centers[0]
                        print(f"Robot position in cropped image: ({robot_x}, {robot_y})")
                    
                    counter += 1
                    last_capture_time = current_time
                
                cv2.imshow("Cropped Desk", cropped_with_robot)
        
        # show frame with crop
        cv2.imshow("Aruco Marker Detection", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # press c to recalibrate if image is bad
        elif key == ord('c'):
            print("Recalibrating camera...")
            cv2.destroyAllWindows()
            if os.path.exists(CALIBRATION_FILE):
                os.remove(CALIBRATION_FILE)
            camera_matrix, dist_coeffs = calibrate_camera()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()