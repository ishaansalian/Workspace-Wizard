import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os
import math
import asyncio
from bleak import BleakClient, BleakScanner
import re

# ArUco marker settings
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
MARKER_IDS = [0, 1, 2, 3, 4]  
MARKER_SIZE = 200  # pixels
CAPTURE_INTERVAL = 10  # sec to take image
CALIBRATION_DIR = "calibration_images"
CHESSBOARD_SIZE = (11, 7)  # chessboard size
SQUARE_SIZE = 2.1  # cm
NUM_CALIBRATION_IMGS = 15
MOVEMENT_UNIT = 1.0  # cm 

# Pixel to cm conversion ratio (will be calculated during runtime)
PIXEL_TO_CM_RATIO = None

# Camera calibration file
CALIBRATION_FILE = "camera_calibration.npz"

# BLE settings
BLE_ADDRESS = "4D9AF5DE-80E1-6702-F956-D874824235C9"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
VALID_COMMANDS = {"F", "B", "L", "R", "N", "M", "O", "P"}
CUSTOM_MESSAGES = {"finished", "stuck"}
COMMAND_DELAY = 1.0  # seconds between sending commands

# ArUco marker generation
def generate_aruco_markers():
    for marker_id in MARKER_IDS:
        marker = aruco.generateImageMarker(ARUCO_DICT, marker_id, MARKER_SIZE)
        cv2.imwrite(f"marker_{marker_id}.png", marker)
    print("Markers saved as PNG files.")

# Camera calibration functions
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

async def send_commands_ble(commands):
    ADDRESS = await find_device("ESP32_Robot")  # Adjust to match your device's name
    if not ADDRESS:
        print("ESP32 device not found!")
        return
    """Send a list of commands over BLE with a delay between each"""
    try:
        async with BleakClient(BLE_ADDRESS) as client:
            if await client.is_connected():
                print(f"Connected to {BLE_ADDRESS}")
                
                for cmd in commands:
                    parsed_command = parse_command(cmd)
                    if parsed_command:
                        print(f"Sending command: {parsed_command}")
                        await client.write_gatt_char(CHARACTERISTIC_UUID, parsed_command.encode())
                        print(f"Command sent: {parsed_command}")
                        await asyncio.sleep(COMMAND_DELAY)  # Delay between commands
                    else:
                        print(f"Invalid command format: {cmd}")
                
                print("All commands sent successfully.")
                return True
            else:
                print("Failed to connect to ESP32!")
                return False
    except Exception as e:
        print(f"Error in BLE communication: {e}")
        return False

# Modified function: now automatically navigates to marker 3
async def enter_movement_mode(marker_centers, pixel_to_cm_ratio, robot_orientation):
    # Check if both robot (marker 0) and target (marker 3) are visible
    if 0 not in marker_centers:
        print("Cannot enter movement mode: Robot marker (ID 0) not detected.")
        return
    
    if 3 not in marker_centers:
        print("Cannot enter movement mode: Target marker (ID 3) not detected.")
        return
            
    if pixel_to_cm_ratio is None:
        print("Cannot enter movement mode: Pixel to cm ratio not calculated.")
        return
    
    # Get current robot position
    robot_x_px, robot_y_px = marker_centers[0]
    robot_x_cm, robot_y_cm = pixels_to_cm(robot_x_px, robot_y_px, pixel_to_cm_ratio)
    
    # Get target position (marker 3)
    target_x_px, target_y_px = marker_centers[3]
    target_x_cm, target_y_cm = pixels_to_cm(target_x_px, target_y_px, pixel_to_cm_ratio)
    
    print(f"Current robot position: ({robot_x_cm:.2f} cm, {robot_y_cm:.2f} cm)")
    print(f"Target position (marker 3): ({target_x_cm:.2f} cm, {target_y_cm:.2f} cm)")
    
    # Generate movement commands
    if robot_orientation is not None:
        commands = generate_movement_commands(
            robot_x_cm, robot_y_cm, robot_orientation, target_x_cm, target_y_cm)
        
        print("Movement commands:")
        print(commands)
        
        # Automatically send the commands without asking for confirmation
        print("Sending commands to robot to navigate to marker 3...")
        success = await send_commands_ble(commands)
        if success:
            print("Commands sent successfully!")
        else:
            print("Failed to send commands.")
    else:
        print("Could not determine robot orientation. Cannot generate movement commands.")

# Main function with asyncio support
async def main():
    generate_aruco_markers()
    
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
        # press m to automatically navigate to marker 3
        elif key == ord('m'):
            # Use latest available data
            await enter_movement_mode(latest_marker_centers, pixel_to_cm_ratio, latest_robot_orientation)

    cap.release()
    cv2.destroyAllWindows()

# Entry point with asyncio
if __name__ == "__main__":
    asyncio.run(main())
