import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os
import math
import asyncio
from bleak import BleakClient, BleakScanner
import re
import ble_connect


ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
MARKER_IDS = [0, 1, 2, 3, 4]  
MARKER_SIZE = 200  # pixels
CAPTURE_INTERVAL = 10  # #sec to take image
CALIBRATION_DIR = "calibration_images"
CHESSBOARD_SIZE = (11, 7)  # chessboard size
SQUARE_SIZE = 2.1  # cm
NUM_CALIBRATION_IMGS = 15
MOVEMENT_UNIT = 1  # cm 
# Replace this with your ESP32's BLE address
ADDRESS = "CC:BA:97:01:7C:71"

# Replace with your ESP32's BLE characteristic UUID
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# Command Mappings
COMMANDS = {
    "w": "F",  # Forward
    "s": "B",  # Backward
    "a": "L",  # Left
    "d": "R",  # Right
    "q": "exit"  # Exit
}
# Valid command characters
VALID_COMMANDS = {"F", "B", "L", "R", "N", "M", "O", "P"}
# Custom messages
CUSTOM_MESSAGES = {"finished", "stuck"}

# Course correction parameters
MAX_CORRECTION_ATTEMPTS = 3
POSITION_TOLERANCE = 0.5  # cm - acceptable distance from target
WAIT_AFTER_COMMAND = 2  # seconds to wait after sending a command before checking position

# Pixel to cm conversion ratio (will be calculated during runtime)
PIXEL_TO_CM_RATIO = None

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
        camera_matrix, dist_coeffs, (w, h), 0, (w, h))
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
    robot_orientation = None

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
            
            # if robot, store corners and calculate orientation
            if marker_id == 0:
                robot_corners = [(int(x), int(y)) for x, y in corner_points]
                
                # Calculate robot orientation (front direction)
                # Assuming corner 0 and 1 form the front edge of the marker
                front_mid_x = (corner_points[0][0] + corner_points[1][0]) / 2
                front_mid_y = (corner_points[0][1] + corner_points[1][1]) / 2
                
                # Vector from center to front midpoint
                dir_x = front_mid_x - center_x
                dir_y = front_mid_y - center_y
                
                # Calculate angle in radians, then convert to degrees
                robot_orientation = math.atan2(dir_y, dir_x) * 180 / math.pi
                
                # Draw orientation line
                end_x = int(center_x + 30 * math.cos(robot_orientation * math.pi / 180))
                end_y = int(center_y + 30 * math.sin(robot_orientation * math.pi / 180))
                cv2.line(frame, (center_x, center_y), (end_x, end_y), (255, 0, 0), 2)
                
                # label each corner 0-3
                for j, (x, y) in enumerate(robot_corners):
                    cv2.putText(frame, f"{j}", (x, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return frame, marker_corners, marker_centers, robot_corners, robot_orientation

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

def calculate_pixel_to_cm_ratio(marker_corners):
    """Calculate the pixel to cm ratio based on the corner markers"""
    global PIXEL_TO_CM_RATIO
    
    if not all(marker_id in marker_corners for marker_id in [1, 2, 3, 4]):
        return None
    
    # Get the distance between markers in pixels
    top_left = marker_corners[1][2]  # bottom right of marker 1
    top_right = marker_corners[2][3]  # bottom left of marker 2
    
    # Calculate pixel distance
    pixel_distance = np.sqrt((top_right[0] - top_left[0])**2 + (top_right[1] - top_left[1])**2)
    
    # Known distance in cm (based on your setup)
    # Assuming markers are placed at known distances
    # For example, if markers are placed 50cm apart:
    known_distance_cm = 50.0  # adjust this value based on your actual setup
    
    # Calculate ratio
    PIXEL_TO_CM_RATIO = known_distance_cm / pixel_distance
    
    return PIXEL_TO_CM_RATIO

def pixels_to_cm(x_px, y_px, ratio):
    """Convert pixel coordinates to cm"""
    if ratio is None:
        return None, None
    
    x_cm = x_px * ratio
    y_cm = y_px * ratio
    
    return x_cm, y_cm

def generate_movement_commands(current_x_cm, current_y_cm, current_orientation, target_x_cm, target_y_cm):
    """Generate movement commands to reach the target position"""
    commands = []
    
    # Calculate distance to move in each direction
    dx = target_x_cm - current_x_cm
    dy = target_y_cm - current_y_cm
    
    # Define the four possible orientations (0 = east, 90 = south, 180 = west, 270 = north)
    # These are approximate directions in the image coordinate system
    EAST = 0
    SOUTH = 90
    WEST = 180
    NORTH = 270
    
    # Normalize the orientation to 0-360 range
    current_orientation = current_orientation % 360
    if current_orientation < 0:
        current_orientation += 360
    
    # Round to nearest cardinal direction
    closest_orientation = round(current_orientation / 90) * 90
    if closest_orientation == 360:
        closest_orientation = 0
    
    # Move in X direction first
    if abs(dx) > 0.1:  # Small threshold to account for measurement errors
        # Determine required orientation to move in x direction
        target_orientation = EAST if dx > 0 else WEST
        
        # Add rotation commands to face the right direction for x movement
        rotation_commands = get_rotation_commands(closest_orientation, target_orientation)
        commands.extend(rotation_commands)
        closest_orientation = target_orientation
        
        # Calculate number of movement units
        units = abs(dx) / MOVEMENT_UNIT
        num_units = round(units)
        
        # Add movement command
        direction = "F" if dx > 0 else "F"  # Both are forward because we rotated to face the right direction
        if num_units > 0:
            commands.append(f"{num_units}{direction}")
    
    # Move in Y direction next
    if abs(dy) > 0.1:  # Small threshold to account for measurement errors
        # Determine required orientation to move in y direction
        target_orientation = SOUTH if dy > 0 else NORTH
        
        # Add rotation commands to face the right direction for y movement
        rotation_commands = get_rotation_commands(closest_orientation, target_orientation)
        commands.extend(rotation_commands)
        
        # Calculate number of movement units
        units = abs(dy) / MOVEMENT_UNIT
        num_units = round(units)
        
        # Add movement command
        direction = "F" if dy > 0 else "F"  # Both are forward because we rotated to face the right direction
        if num_units > 0:
            commands.append(f"{num_units}{direction}")
    
    return commands

def get_rotation_commands(current_orientation, target_orientation):
    """Get commands to rotate from current orientation to target orientation"""
    commands = []
    
    # Calculate the smallest angle to rotate
    diff = (target_orientation - current_orientation) % 360
    if diff > 180:
        diff -= 360
    
    # Determine rotation direction
    if diff == 0:
        # No rotation needed
        return commands
    elif diff == 90 or diff == -270:
        # Turn right
        commands.append("R")
    elif diff == -90 or diff == 270:
        # Turn left
        commands.append("L")
    elif abs(diff) == 180:
        # Turn around (two right turns)
        commands.append("R")
        commands.append("R")
    
    return commands

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
async def find_device(name):
    devices = await BleakScanner.discover()
    for device in devices:
        if device.name and name in device.name:
            return device.address
    return None

async def send_command(cmd):
    ADDRESS = await find_device("ESP32_Robot")

    if cmd:
        async with BleakClient(ADDRESS) as client:
            if await client.is_connected():
                print(f"Connected to {ADDRESS}")

                if command in COMMANDS:
                    command = COMMANDS[command]
                    parsed_command = ble_connect.parse_command(command)
                    await client.write_gatt_char(CHARACTERISTIC_UUID, parsed_command.encode())
                    print(f"Sent command: {parsed_command}")
                    await asyncio.sleep(1)  # Delay between commands
                else:
                    print("Invalid command. Please try again.")
            print("Failed to connect!")
    return

def get_current_robot_position(cap, camera_matrix, dist_coeffs, pixel_to_cm_ratio):
    """Get the current position and orientation of the robot from camera"""
    # Grab several frames to ensure we have a good reading
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return None, None, None
        
        # Undistort frame
        if camera_matrix is not None and dist_coeffs is not None:
            frame = undistort_image(frame, camera_matrix, dist_coeffs)
        
        # Detect markers
        _, marker_corners, marker_centers, _, robot_orientation = detect_aruco_markers(frame)
        
        # Check if robot marker is detected
        if 0 in marker_centers and pixel_to_cm_ratio:
            robot_x_px, robot_y_px = marker_centers[0]
            robot_x_cm, robot_y_cm = pixels_to_cm(robot_x_px, robot_y_px, pixel_to_cm_ratio)
            
            return robot_x_cm, robot_y_cm, robot_orientation
    
    print("Warning: Could not detect robot position after multiple attempts")
    return None, None, None

def move_robot_with_correction(cap, camera_matrix, dist_coeffs, pixel_to_cm_ratio, target_x_cm, target_y_cm):
    """Move robot to target position with course correction"""
    attempts = 0
    
    while attempts < MAX_CORRECTION_ATTEMPTS:
        # Get current robot position
        robot_x_cm, robot_y_cm, robot_orientation = get_current_robot_position(
            cap, camera_matrix, dist_coeffs, pixel_to_cm_ratio)
        
        if robot_x_cm is None or robot_y_cm is None or robot_orientation is None:
            print("Error: Could not determine robot position")
            return False
            
        # Calculate distance to target
        distance = calculate_distance(robot_x_cm, robot_y_cm, target_x_cm, target_y_cm)
        
        print(f"Attempt {attempts+1}: Robot at ({robot_x_cm:.2f}, {robot_y_cm:.2f}), "
              f"target at ({target_x_cm:.2f}, {target_y_cm:.2f}), distance: {distance:.2f} cm")
        
        # Check if we're close enough to target
        if distance <= POSITION_TOLERANCE:
            print(f"Target reached within tolerance ({distance:.2f} cm <= {POSITION_TOLERANCE} cm)")
            return True
            
        # Generate and send movement commands
        commands = generate_movement_commands(
            robot_x_cm, robot_y_cm, robot_orientation, target_x_cm, target_y_cm)
            
        print(f"Course correction commands: {commands}")
        
        # Send each command
        for cmd in commands:
            success = send_command(cmd)
            if not success:
                print(f"Error sending command: {cmd}")
                return False
                
            # Wait a moment for the robot to execute the command
            time.sleep(WAIT_AFTER_COMMAND)
            
        attempts += 1
    
    print(f"Warning: Robot failed to reach target position after {MAX_CORRECTION_ATTEMPTS} attempts")
    return False

def navigate_to_position(cap, camera_matrix, dist_coeffs, pixel_to_cm_ratio, target_x_cm, target_y_cm):
    """Navigate robot to the specified position with course correction"""
    print(f"Navigating to position: ({target_x_cm:.2f}, {target_y_cm:.2f}) cm")
    
    # Initial movement to target
    robot_x_cm, robot_y_cm, robot_orientation = get_current_robot_position(
        cap, camera_matrix, dist_coeffs, pixel_to_cm_ratio)
    
    if robot_x_cm is None or robot_y_cm is None or robot_orientation is None:
        print("Error: Could not determine robot position")
        return False
    
    # Generate initial movement commands
    commands = generate_movement_commands(
        robot_x_cm, robot_y_cm, robot_orientation, target_x_cm, target_y_cm)
    
    print(f"Initial commands: {commands}")
    
    # Send each command
    for cmd in commands:
        success = send_command(cmd)
        if not success:
            print(f"Error sending command: {cmd}")
            return False
        
        # Wait a moment for the robot to execute the command
        time.sleep(WAIT_AFTER_COMMAND)
    
    # Perform course correction
    return move_robot_with_correction(cap, camera_matrix, dist_coeffs, pixel_to_cm_ratio, target_x_cm, target_y_cm)

def main():
    generate_aruco_markers()
    camera_matrix, dist_coeffs = calibrate_camera()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    print("Press 'q' to quit. Press 'c' to recalibrate camera. Press 'm' to enter movement mode.")
    last_capture_time = time.time()
    counter = 0
    
    # Initialize pixel to cm ratio
    pixel_to_cm_ratio = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # undistort
        if camera_matrix is not None and dist_coeffs is not None:
            frame = undistort_image(frame, camera_matrix, dist_coeffs)

        # detect markers
        frame_with_markers, marker_corners, marker_centers, robot_corners, robot_orientation = detect_aruco_markers(frame)
        
        # Calculate pixel to cm ratio if not already done
        if pixel_to_cm_ratio is None and len(marker_corners) >= 4:
            pixel_to_cm_ratio = calculate_pixel_to_cm_ratio(marker_corners)
            if pixel_to_cm_ratio:
                print(f"Calculated pixel to cm ratio: {pixel_to_cm_ratio}")
        
        # Display current robot position in cm if available
        if 0 in marker_centers and pixel_to_cm_ratio:
            robot_x_px, robot_y_px = marker_centers[0]
            robot_x_cm, robot_y_cm = pixels_to_cm(robot_x_px, robot_y_px, pixel_to_cm_ratio)
            orientation_text = f"Orientation: {robot_orientation:.1f} degrees" if robot_orientation is not None else "Orientation: unknown"
            cv2.putText(frame_with_markers, f"Robot: ({robot_x_cm:.2f} cm, {robot_y_cm:.2f} cm) {orientation_text}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
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
                cropped_with_robot, _, _, robot_corners_cropped, _ = detect_aruco_markers(cropped_frame)

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
                        
                        # Convert to cm if ratio is available
                        if pixel_to_cm_ratio:
                            robot_x_cm, robot_y_cm = pixels_to_cm(robot_x, robot_y, pixel_to_cm_ratio)
                            print(f"Robot position in cm: ({robot_x_cm:.2f}, {robot_y_cm:.2f})")
                    
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
            
        # press m to enter movement mode
        elif key == ord('m'):
            if 0 not in marker_centers:
                print("Cannot enter movement mode: Robot marker (ID 0) not detected.")
                continue
                
            if pixel_to_cm_ratio is None:
                print("Cannot enter movement mode: Pixel to cm ratio not calculated.")
                continue
            
            # Get current robot position
            robot_x_px, robot_y_px = marker_centers[0]
            robot_x_cm, robot_y_cm = pixels_to_cm(robot_x_px, robot_y_px, pixel_to_cm_ratio)
            
            # Get destination from user
            print(f"Current robot position: ({robot_x_cm:.2f} cm, {robot_y_cm:.2f} cm)")
            print("Enter destination coordinates in pixels (x,y):")
            try:
                input_str = input().strip()
                target_x_px, target_y_px = map(int, input_str.strip('()').split(','))
                
                # Convert target to cm
                target_x_cm, target_y_cm = pixels_to_cm(target_x_px, target_y_px, pixel_to_cm_ratio)
                
                print(f"Target position in cm: ({target_x_cm:.2f}, {target_y_cm:.2f})")
                
                # Navigate to position with course correction
                if robot_orientation is not None:
                    success = navigate_to_position(
                        cap, camera_matrix, dist_coeffs, pixel_to_cm_ratio, 
                        target_x_cm, target_y_cm)
                    
                    if success:
                        print("Navigation completed successfully")
                    else:
                        print("Navigation failed")
                else:
                    print("Could not determine robot orientation. Cannot generate movement commands.")
            except ValueError:
                print("Invalid input format. Use format 'x,y' or '(x,y)'.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()