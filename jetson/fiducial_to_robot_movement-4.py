import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os
import math
import asyncio
from bleak import BleakClient, BleakScanner
import re

# very important for NVIDIA
pipeline = "v4l2src device=/dev/video0 ! image/jpeg,framerate=30/1 ! jpegdec ! videoconvert ! appsink"

# ArUco marker settings
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
MARKER_IDS = [0, 1, 2, 3, 4]  
MARKER_SIZE = 200  # pixels
CAPTURE_INTERVAL = 10  # sec to take image
CALIBRATION_DIR = "calibration_images"
CHESSBOARD_SIZE = (11, 7)  # chessboard size
SQUARE_SIZE = 2.0  # cm
NUM_CALIBRATION_IMGS = 15
MOVEMENT_UNIT = 0.5  # cm 

# Pixel to cm conversion ratio (will be calculated during runtime)
PIXEL_TO_CM_RATIO = None

# Camera calibration file
CALIBRATION_FILE = "camera_calibration.npz"

# BLE settings
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
VALID_COMMANDS = {"F", "B", "L", "R", "N", "M", "O", "P"}
CUSTOM_MESSAGES = {"finished", "stuck"}
COMMAND_DELAY = 1.0  # seconds between sending commands
POSITION_CHECK_DELAY = 2.0  # seconds to wait for robot to stabilize after movement

# ArUco marker generation
def generate_aruco_markers():
    for marker_id in MARKER_IDS:
        marker = aruco.generateImageMarker(ARUCO_DICT, marker_id, MARKER_SIZE)
        cv2.imwrite(f"marker_{marker_id}.png", marker)
    print("Markers saved as PNG files.")

# Profiling function
def profile_time(start_time, task_name):
    elapsed = time.time() - start_time
    print(f"[{task_name}] Elapsed time: {elapsed:.4f} seconds")

# Camera calibration function
def calibrate_camera():
    start_total = time.time()
    if os.path.exists(CALIBRATION_FILE):
        print("Loading existing camera calibration...")
        data = np.load(CALIBRATION_FILE)
        camera_matrix, dist_coeffs = data['camera_matrix'], data['dist_coeffs']
        profile_time(start_total, "Load Calibration")
        return camera_matrix, dist_coeffs
    
    print("Camera calibration!")
    print(f"Please show a {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} chessboard pattern to the camera")
    print(f"Need to capture {NUM_CALIBRATION_IMGS} images")
    
    # Create directory for calibration images if it doesn't exist
    if not os.path.exists(CALIBRATION_DIR):
        os.makedirs(CALIBRATION_DIR)
    
    # Prep object points
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE
    objpoints = []  # 3D points
    imgpoints = []  # 2D points in image

    # Optimized GStreamer pipeline with reduced resolution
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None, None

    img_count = 0

    while img_count < NUM_CALIBRATION_IMGS:
        start_frame = time.time()
        ret, frame = cap.read()
        profile_time(start_frame, "Frame Capture")

        if not ret:
            print("Error: Could not read frame.")
            break

        # CPU-based grayscale conversion (faster for lower resolutions)
        start_gray = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        profile_time(start_gray, "CPU Grayscale Conversion")

        # Chessboard detection
        start_detect = time.time()
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        profile_time(start_detect, "Chessboard Detection")

        display_frame = frame.copy()

        if ret:
            # Refine corner positions
            start_refine = time.time()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            profile_time(start_refine, "Corner Refinement")

            # Draw corners and display the frame
            cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners2, ret)
            cv2.putText(display_frame, f"Image {img_count+1}/{NUM_CALIBRATION_IMGS}: Press SPACE to capture",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save calibration data if spacebar is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                img_name = os.path.join(CALIBRATION_DIR, f"calibration_{img_count}.jpg")
                cv2.imwrite(img_name, frame)
                print(f"Saved {img_name}")
                objpoints.append(objp)
                imgpoints.append(corners2)
                img_count += 1
                time.sleep(0.2)
        else:
            cv2.putText(display_frame, "Chessboard not detected",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Camera Calibration", display_frame)

        # Quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # Camera Calibration Process
    print("Calibrating camera, please wait...")
    start_calibrate = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    profile_time(start_calibrate, "Camera Calibration")

    if ret:
        print("Camera calibration successful!")
        print(f"Camera Matrix:\n{camera_matrix}")
        print(f"Distortion Coefficients:\n{dist_coeffs}")

        # Save calibration results
        np.savez(CALIBRATION_FILE, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        profile_time(start_total, "Total Calibration Process")
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

# ArUco marker detection
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
    known_distance_cm = 84.455 # adjust this value based on your actual setup
    
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

# Motion planning functions
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

# BLE communication functions
def parse_command(input_str):
    """Parse input like '10F', 'M', or custom messages like 'finished'."""
    input_str = input_str.strip().lower()
    if input_str in CUSTOM_MESSAGES:
        return input_str
    match = re.fullmatch(r'(\d*)([FBLRNMOP])', input_str.upper())
    if match:
        count = match.group(1)
        command = match.group(2)
        return (count if count else "1") + command
    return None

async def find_device(name):
    devices = await BleakScanner.discover()
    for device in devices:
        if device.name and name in device.name:
            return device.address
    return None

async def send_single_command_ble(cmd):
    """Send a single command over BLE"""
    ADDRESS = await find_device("Workspace_Wizard")  # Adjust to match your device's name
    if not ADDRESS:
        print("ESP32 device not found!")
        return False
    
    try:
        async with BleakClient(ADDRESS) as client:
            if await client.is_connected():
                print(f"Connected to {ADDRESS}")
                
                parsed_command = parse_command(cmd)
                if parsed_command:
                    print(f"Sending command: {parsed_command}")
                    await client.write_gatt_char(CHARACTERISTIC_UUID, parsed_command.encode())
                    print(f"Command sent: {parsed_command}")
                    return True
                else:
                    print(f"Invalid command format: {cmd}")
                    return False
            else:
                print("Failed to connect to ESP32!")
                return False
    except Exception as e:
        print(f"Error in BLE communication: {e}")
        return False

# Function to capture the current state of the robot
async def capture_robot_state(camera_matrix, dist_coeffs, pixel_to_cm_ratio):
    """Capture the current state (position and orientation) of the robot"""
    # Create a video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return None, None, None
    
    # Wait a bit for the camera to stabilize
    await asyncio.sleep(POSITION_CHECK_DELAY)
    
    # Take multiple readings to get more accurate position
    robot_x_cm_sum = 0
    robot_y_cm_sum = 0
    robot_orientation_sum = 0
    valid_readings = 0
    max_readings = 5
    
    for _ in range(max_readings):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            continue
        
        # Undistort
        if camera_matrix is not None and dist_coeffs is not None:
            frame = undistort_image(frame, camera_matrix, dist_coeffs)
        
        # Detect markers
        _, _, marker_centers, _, robot_orientation = detect_aruco_markers(frame)
        
        if 0 in marker_centers and robot_orientation is not None and pixel_to_cm_ratio is not None:
            robot_x_px, robot_y_px = marker_centers[0]
            robot_x_cm, robot_y_cm = pixels_to_cm(robot_x_px, robot_y_px, pixel_to_cm_ratio)
            
            robot_x_cm_sum += robot_x_cm
            robot_y_cm_sum += robot_y_cm
            robot_orientation_sum += robot_orientation
            valid_readings += 1
        
        # Small delay between readings
        await asyncio.sleep(0.1)
    
    cap.release()
    
    if valid_readings > 0:
        avg_x_cm = robot_x_cm_sum / valid_readings
        avg_y_cm = robot_y_cm_sum / valid_readings
        avg_orientation = robot_orientation_sum / valid_readings
        
        print(f"Captured robot state: ({avg_x_cm:.2f} cm, {avg_y_cm:.2f} cm), {avg_orientation:.2f} degrees")
        return avg_x_cm, avg_y_cm, avg_orientation
    else:
        print("Failed to capture robot state: No valid readings.")
        return None, None, None

# New function to execute commands with position correction
async def execute_commands_with_correction(commands, target_x_cm, target_y_cm, camera_matrix, dist_coeffs, pixel_to_cm_ratio):
    """Execute commands with position correction after each turn"""
    if not commands:
        print("No commands to execute.")
        return
    
    # Get initial robot state
    robot_x_cm, robot_y_cm, robot_orientation = await capture_robot_state(camera_matrix, dist_coeffs, pixel_to_cm_ratio)
    if robot_x_cm is None:
        print("Failed to get initial robot state. Aborting.")
        return
    
    # Process commands one by one
    i = 0
    while i < len(commands):
        current_cmd = commands[i]
        print(f"Executing command {i+1}/{len(commands)}: {current_cmd}")
        
        # Send the current command
        success = await send_single_command_ble(current_cmd)
        if not success:
            print(f"Failed to send command: {current_cmd}. Aborting.")
            return
        
        # Wait for the command to be executed
        await asyncio.sleep(COMMAND_DELAY)
        
        # Check if this was a turn command (R or L)
        is_turn = current_cmd in ["R", "L"]
        
        # After a turn, recalculate the path
        if is_turn or i == len(commands) - 1:  # Also recalculate after the last command
            # Get updated robot state
            robot_x_cm, robot_y_cm, robot_orientation = await capture_robot_state(camera_matrix, dist_coeffs, pixel_to_cm_ratio)
            if robot_x_cm is None:
                print("Failed to get updated robot state. Continuing with original plan.")
                i += 1
                continue
            
            # If this was the last command, check if we reached the target
            if i == len(commands) - 1:
                dx = target_x_cm - robot_x_cm
                dy = target_y_cm - robot_y_cm
                distance_to_target = math.sqrt(dx*dx + dy*dy)
                
                print(f"Distance to target: {distance_to_target:.2f} cm")
                if distance_to_target < 5.0:  # Within 5cm is considered success
                    print("Successfully reached the target!")
                    return
                else:
                    print("Target not reached accurately. Recalculating path...")
            
            # If there are more commands, recalculate the path from the current position
            if i < len(commands) - 1:
                new_commands = generate_movement_commands(
                    robot_x_cm, robot_y_cm, robot_orientation, target_x_cm, target_y_cm)
                
                if new_commands:
                    print(f"Recalculated path: {new_commands}")
                    # Replace the remaining commands with the new ones
                    commands = commands[:i+1] + new_commands
        
        i += 1
    
    print("All commands executed.")

# Modified function: now automatically navigates to marker 3 with position correction
async def navigate_to_marker(marker_centers, pixel_to_cm_ratio, robot_orientation, camera_matrix, dist_coeffs, target_marker_id=3):
    # Check if both robot (marker 0) and target marker are visible
    if 0 not in marker_centers:
        print(f"Cannot navigate: Robot marker (ID 0) not detected.")
        return
    
    if target_marker_id not in marker_centers:
        print(f"Cannot navigate: Target marker (ID {target_marker_id}) not detected.")
        return
            
    if pixel_to_cm_ratio is None:
        print("Cannot navigate: Pixel to cm ratio not calculated.")
        return
    
    # Get current robot position
    robot_x_px, robot_y_px = marker_centers[0]
    robot_x_cm, robot_y_cm = pixels_to_cm(robot_x_px, robot_y_px, pixel_to_cm_ratio)
    
    # Get target position
    target_x_px, target_y_px = marker_centers[target_marker_id]
    target_x_cm, target_y_cm = pixels_to_cm(target_x_px, target_y_px, pixel_to_cm_ratio)
    
    print(f"Current robot position: ({robot_x_cm:.2f} cm, {robot_y_cm:.2f} cm)")
    print(f"Target position (marker {target_marker_id}): ({target_x_cm:.2f} cm, {target_y_cm:.2f} cm)")
    
    # Generate initial movement commands
    if robot_orientation is not None:
        commands = generate_movement_commands(
            robot_x_cm, robot_y_cm, robot_orientation, target_x_cm, target_y_cm)
        
        print("Initial movement commands:")
        print(commands)
        
        # Execute commands with position correction
        print(f"Navigating to marker {target_marker_id} with position correction...")
        await execute_commands_with_correction(
            commands, target_x_cm, target_y_cm, camera_matrix, dist_coeffs, pixel_to_cm_ratio)
    else:
        print("Could not determine robot orientation. Cannot generate movement commands.")

# Main function with asyncio support
async def main():
    #generate_aruco_markers()
    
    # calibrate
    camera_matrix, dist_coeffs = calibrate_camera()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    print("Press 'q' to quit. Press 'c' to recalibrate camera. Press 'm' to navigate to marker 3.")
    last_capture_time = time.time()
    counter = 0
    
    # Initialize pixel to cm ratio
    pixel_to_cm_ratio = None
    latest_robot_orientation = None
    latest_marker_centers = {}

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
        
        # Store latest valid data
        if marker_centers:
            latest_marker_centers = marker_centers
        if robot_orientation is not None:
            latest_robot_orientation = robot_orientation
        
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
        
        # Display target marker 3 position if available
        if 3 in marker_centers and pixel_to_cm_ratio:
            target_x_px, target_y_px = marker_centers[3]
            target_x_cm, target_y_cm = pixels_to_cm(target_x_px, target_y_px, pixel_to_cm_ratio)
            cv2.putText(frame_with_markers, f"Target (Marker 3): ({target_x_cm:.2f} cm, {target_y_cm:.2f} cm)", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
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

                current_time = time.time()
                if current_time - last_capture_time >= CAPTURE_INTERVAL:
                    if 0 in marker_centers:
                        robot_x, robot_y = marker_centers[0]
                    
                    counter += 1
                    last_capture_time = current_time
                
                cv2.imshow("Cropped Desk", cropped_with_robot)
        
        # show frame with crop
        cv2.imshow("Aruco Marker Detection", display_frame)
        
        # Handle user input
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
        # press m to automatically navigate to marker 3 with position correction
        elif key == ord('m'):
            # Use latest available data
            await navigate_to_marker(
                latest_marker_centers, 
                pixel_to_cm_ratio, 
                latest_robot_orientation,
                camera_matrix,
                dist_coeffs,
                target_marker_id=3
            )

    cap.release()
    cv2.destroyAllWindows()

# Entry point with asyncio
if __name__ == "__main__":
    asyncio.run(main())