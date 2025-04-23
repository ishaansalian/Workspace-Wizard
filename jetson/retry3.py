from ultralytics import YOLO
import cv2
import numpy as np
import math
import heapq
import copy
import time

import cv2.aruco as aruco
import os

import asyncio
import re
from bleak import BleakClient, BleakScanner

# SEND COMMANDS

CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
VALID_COMMANDS = {"F", "B", "L", "R", "N", "M", "O", "P"}
CUSTOM_MESSAGES = {"finished", "stuck"}
COMMAND_DELAY = 1.0                         # seconds between sending commands
MOVEMENT_UNIT = 0.5                         # cm 


def ground_control(unmoved_contours, unmoved_boxes, fixed_contours, fixed_boxes, failed_contours, failed_boxes, moved_boxes, path, robot_path, robot_pos):
    import asyncio
    
    if len(robot_path) < 2:
        return []
    
    # Extract turning points from the robot's path
    robot_turning_points = [robot_path[0]]
    
    for i in range(1, len(robot_path) - 1):
        prev_dx = robot_path[i][0] - robot_path[i-1][0]
        prev_dy = robot_path[i][1] - robot_path[i-1][1]
        next_dx = robot_path[i+1][0] - robot_path[i][0]
        next_dy = robot_path[i+1][1] - robot_path[i][1]
        
        # If direction changes, it's a turning point
        if (prev_dx, prev_dy) != (next_dx, next_dy):
            robot_turning_points.append(robot_path[i])
    
    # Always add the last point
    robot_turning_points.append(robot_path[-1])

    print(f"Robot turning points: {robot_turning_points}")
    
    # Current robot position and orientation
    x, y, w, h, angle = robot_pos
    current_pos = (x, y)
    current_orientation = angle
    
    # Show the turning points
    for i in range(len(robot_turning_points)-1):
        robot_move = [robot_turning_points[i], robot_turning_points[i+1]]
        
        # Calculate movement commands for this segment
        start_pixel = robot_turning_points[i]
        end_pixel = robot_turning_points[i+1]
        
        # Convert pixels to cm
        start_cm = pixel_to_cm(start_pixel)
        end_cm = pixel_to_cm(end_pixel)
        
        print(f"Moving from {start_pixel} to {end_pixel} (pixels)")
        print(f"Moving from {start_cm} to {end_cm} (cm)")
        
        # Generate commands for this segment
        commands = generate_movement_commands(
            start_cm[0], start_cm[1], 
            current_orientation, 
            end_cm[0], end_cm[1]
        )
        
        print(f"Commands for this segment: {commands}")
        
        # Send commands to the robot
        try:
            # Run in event loop if it exists, otherwise create a new one
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(send_commands_ble(commands))
            else:
                loop.run_until_complete(send_commands_ble(commands))
        except Exception as e:
            print(f"Error sending commands: {e}")
        
        # Update robot's current position and orientation for next segment
        # The robot path already contains top-left coordinates, so use them directly
        current_pos = end_pixel
        
        # Update orientation based on movement direction
        dx = end_pixel[0] - start_pixel[0]
        dy = end_pixel[1] - start_pixel[1]
        if abs(dx) > abs(dy):  # Horizontal movement dominates
            current_orientation = 0 if dx > 0 else 180  # East or West
        else:  # Vertical movement dominates
            current_orientation = 90 if dy > 0 else 270  # South or North
        
        # The robot_cur_pos includes size and orientation
        robot_cur_pos = (end_pixel[0], end_pixel[1], w, h, current_orientation)

        # Visualize the movement
        visualize(unmoved_contours, unmoved_boxes, fixed_contours, fixed_boxes, 
                 failed_contours, failed_boxes, moved_boxes, path, 
                 robot_move, robot_cur_pos)

        # Update the image for the next visualization
        cropped_image, robot_position_in_crop, cal_request = cropper.get_cropped_image(save_image=False)
        global image
        image = cropped_image

    return robot_turning_points


def generate_movement_commands(current_x_cm, current_y_cm, current_orientation, target_x_cm, target_y_cm):

    commands = []
    dx = target_x_cm - current_x_cm
    dy = target_y_cm - current_y_cm

    EAST = 0
    SOUTH = 90
    WEST = 180
    NORTH = 270

    if abs(dx) > 0.1:

        target_orientation = EAST if dx > 0 else WEST
        rotation_commands = get_rotation_commands(current_orientation, target_orientation)
        commands.extend(rotation_commands)

        units = abs(dx) / MOVEMENT_UNIT
        num_units = round(units)
        if num_units > 0:
            commands.append(f"{num_units}F")

    elif abs(dy) > 0.1:

        target_orientation = SOUTH if dy > 0 else NORTH
        rotation_commands = get_rotation_commands(current_orientation, target_orientation)
        commands.extend(rotation_commands)

        units = abs(dy) / MOVEMENT_UNIT
        num_units = round(units)
        if num_units > 0:
            commands.append(f"{num_units}F")
    
    return commands


def get_rotation_commands(current_orientation, target_orientation):

    commands = []

    # Calculate the smallest angle to rotate
    diff = (target_orientation - current_orientation) % 360
    if diff > 180:
        diff -= 360
    
    # Determine rotation direction
    if diff == 0:
        return commands
    elif diff == 90 or diff == -270:
        commands.append("R")
    elif diff == -90 or diff == 270:
        commands.append("L")
    elif abs(diff) == 180:
        commands.append("R")
        commands.append("R")
    
    return commands


async def send_commands_ble(commands):
    ADDRESS = await find_device("Workspace_Wizard")  # Adjust to match your device's name
    if not ADDRESS:
        print("ESP32 device not found!")
        return
    """Send a list of commands over BLE with a delay between each"""
    try:
        async with BleakClient(ADDRESS) as client:
            if await client.is_connected():
                print(f"Connected to {ADDRESS}")
                
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

def pixel_to_cm(pixel_coords):
    """Convert pixel coordinates to cm coordinates"""
    global PIXELS_PER_CM
    if PIXELS_PER_CM is None:
        print("Warning: PIXELS_PER_CM not set. Using default value of 10")
        PIXELS_PER_CM = 10
    
    x_cm = pixel_coords[0] / PIXELS_PER_CM
    y_cm = pixel_coords[1] / PIXELS_PER_CM
    return (x_cm, y_cm)


def normalize_orientation(angle_deg):
    """
    Convert any angle to the standard orientation format:
    0 = East, 90 = South, 180 = West, 270 = North
    """
    # Normalize to 0-360
    normalized = angle_deg % 360
    
    # Round to nearest cardinal direction
    directions = [0, 90, 180, 270]
    return min(directions, key=lambda x: abs((normalized - x) % 360))


def smooth_turning_points(turning_points):
    """
    Make sure the robot only moves in cardinal directions by adjusting turning points.
    """
    smoothed = [turning_points[0]]
    
    for i in range(1, len(turning_points)):
        prev = smoothed[-1]
        curr = turning_points[i]
        
        # If the move isn't purely horizontal or vertical, add an intermediate point
        if prev[0] != curr[0] and prev[1] != curr[1]:
            # Add intermediate point that makes this a right-angle turn
            intermediate = (prev[0], curr[1])  # You could also use (curr[0], prev[1])
            smoothed.append(intermediate)
        
        smoothed.append(curr)
    
    return smoothed



model = YOLO("yolov8m-seg.pt")
restricted_items = ["knife"]
path_finding_timeout = 20
failed_path_finding_timeout = 20
timer_on = True

# Cache for robot fits
robot_fit_cache = {}

def clear_robot_fit_cache():
    """Clear the cache when the occupancy grid changes"""
    global robot_fit_cache
    robot_fit_cache.clear()

# visualize code (unchanged)
def visualize(unmoved_contours, unmoved_boxes, fixed_contours, fixed_boxes, failed_contours, failed_boxes, moved_boxes, path, rob_path, robot_pos):
    # visualization colors
    unmoved_color = (0, 0, 128)         # maroon
    fixed_color = (0, 255, 255)         # yellow
    failed_color = (0, 0, 255)          # red
    moved_color = (0, 255, 0)           # green
    path_color = (255, 0, 0)            # blue
    robot_color = (0, 0, 0)             # black

    # visual of change
    img_copy = image.copy()

    # visualizing unmoved objects
    for contour in unmoved_contours.values():
        cv2.fillPoly(img_copy, [contour.reshape(-1, 2)], unmoved_color)
    for _, box in unmoved_boxes.items():
        cv2.rectangle(img_copy, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), unmoved_color, 3)

    # visualizing fixed objects
    for contour in fixed_contours.values():
        cv2.fillPoly(img_copy, [contour.reshape(-1, 2)], fixed_color)
    for _, box in fixed_boxes.items():
        cv2.rectangle(img_copy, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), fixed_color, 3)

    # visualizing failed objects
    for contour in failed_contours.values():
        cv2.fillPoly(img_copy, [contour.reshape(-1, 2)], failed_color)
    for _, box in failed_boxes.items():
        cv2.rectangle(img_copy, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), failed_color, 3)

    # visualizing moved objects
    for _, box in moved_boxes.items():
        cv2.rectangle(img_copy, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), moved_color, 3)

    # visualizing path
    for i in range(len(path) - 1):
        cv2.line(img_copy, path[i], path[i+1], path_color, 2)

    # visualize robot path:
    for i in range(len(rob_path) - 1):
        cv2.line(img_copy, rob_path[i], rob_path[i+1], robot_color, 2)
    
    # visualizing robot
    x, y, w, h, angle = robot_pos
    # For visualization, show the robot's actual square position
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), robot_color, -1)  # Fill
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 165, 255), 2)  # Orange border for visibility

    cv2.imshow("image", img_copy)
    cv2.waitKey(0)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

DIRS = {
    0: ( 0,-1),  # up
    1: ( 0, 1),  # down
    2: (-1, 0),  # left
    3: ( 1, 0),  # right
}

def build_occ(obstacles, desk_dim):
    """Return a numpy array occ[y,x]=True if blocked."""
    W, H = desk_dim
    occ = np.zeros((H, W), dtype=bool)
    for x,y in obstacles:
        if 0 <= x < W and 0 <= y < H:
            occ[y, x] = True
    return occ

def robot_fits_optimized(obj_pos, obj_dim, robot_state, move_dir, occ, desk_dim):
    """Optimized version of robot_fits with caching and numpy operations"""
    global robot_fit_cache
    
    # Create a cache key
    cache_key = (obj_pos, obj_dim, move_dir)
    if cache_key in robot_fit_cache:
        return robot_fit_cache[cache_key]
    
    x0, y0 = obj_pos
    ow, oh = obj_dim
    _, _, rw, rh, _ = robot_state

    if move_dir == 0:    # object moves up → robot goes below
        rx = x0 + ow//2 - rw//2
        ry = y0 + oh
    elif move_dir == 1:  # object moves down → robot goes above
        rx = x0 + ow//2 - rw//2
        ry = y0 - rh
    elif move_dir == 2:  # object moves left → robot goes right
        rx = x0 + ow
        ry = y0 + oh//2 - rh//2
    elif move_dir == 3:  # object moves right → robot goes left
        rx = x0 - rw
        ry = y0 + oh//2 - rh//2
    else:
        robot_fit_cache[cache_key] = (None, False)
        return None, False

    # Check bounds
    W, H = desk_dim
    if rx < 0 or ry < 0 or rx + rw > W or ry + rh > H:
        robot_fit_cache[cache_key] = (None, False)
        return None, False
    
    # Vectorized occupancy check
    robot_area = occ[ry:ry+rh, rx:rx+rw]
    if np.any(robot_area):
        robot_fit_cache[cache_key] = (None, False)
        return None, False

    result = ((rx, ry, rw, rh, move_dir), True)
    robot_fit_cache[cache_key] = result
    return result

def precompute_robot_accessibility(robot_dim, occ, desk_dim):
    """Precompute where the robot can fit in the given occupancy grid"""
    rw, rh = robot_dim
    W, H = desk_dim
    
    # Create accessibility map - mark positions where top-left corner of robot can be placed
    accessible = np.zeros((H, W), dtype=bool)
    
    # Only check valid areas where robot can fit
    for y in range(0, H - rh + 1):
        for x in range(0, W - rw + 1):
            if not np.any(occ[y:y+rh, x:x+rw]):
                accessible[y, x] = True  # Only mark the top-left position as accessible
                
    return accessible

def robot_astar_optimized(start, goal, robot_dim, occ, desk_dim):
    """Optimized A* for robot pathfinding with straight line prioritization"""
    W, H = desk_dim
    rw, rh = robot_dim
    start_time = time.time()
    
    # Precompute where robot's top-left corner can be placed
    accessible = precompute_robot_accessibility(robot_dim, occ, desk_dim)
    
    # Quick check if start or goal position valid
    if not (0 <= start[0] < W-rw+1 and 0 <= start[1] < H-rh+1) or not (0 <= goal[0] < W-rw+1 and 0 <= goal[1] < H-rh+1):
        return [(-2, -2)]
    
    if not accessible[start[1], start[0]] or not accessible[goal[1], goal[0]]:
        return [(-2, -2)]
    
    # Initialize with start position and None as previous direction
    open_set = [(heuristic(start, goal), 0, start, None)]
    came_from = {}
    g_score = {start: 0}
    
    # Also store the incoming direction for each node
    direction_to = {}
    
    # Use a set for fast lookup of visited nodes
    closed_set = set()

    while open_set:
        if timer_on and (time.time() - start_time > path_finding_timeout):
            return [(-2, -2)]
        
        _, _, current, prev_dir = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        node_key = current
        if node_key in closed_set:
            continue
        closed_set.add(node_key)

        gx = g_score[current]
        for dir_name, (dx, dy) in DIRS.items():
            nx, ny = current[0] + dx, current[1] + dy
            
            # Check bounds considering robot size
            if not (0 <= nx < W-rw+1 and 0 <= ny < H-rh+1):
                continue

            # Check if robot can fit at this position
            if not accessible[ny, nx]:
                continue

            # Calculate direction change penalty
            # If prev_dir is None (starting point) or same as current direction, no penalty
            # Otherwise, add a small penalty for changing direction
            direction_penalty = 0
            curr_dir = dir_name
            
            if prev_dir is not None and prev_dir != curr_dir:
                direction_penalty = 0.5  # Penalty for changing direction
            
            ng = gx + 1 + direction_penalty
            neighbor = (nx, ny)
            
            if neighbor in closed_set:
                continue
                
            if ng < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = ng
                came_from[neighbor] = current
                direction_to[neighbor] = curr_dir
                f = ng + heuristic(neighbor, goal)
                # Add current direction to the state
                heapq.heappush(open_set, (f, ng, neighbor, curr_dir))

    return [(-2, -2)]

def object_path_astar_optimized(start, goal, obj_dim, occ, robot_state, desk_dim):
    """Optimized object pathfinding with numpy operations and straight line prioritization"""
    W, H = desk_dim
    ow, oh = obj_dim
    start_time = time.time()
    
    # Clear cache when occupancy grid changes
    clear_robot_fit_cache()

    # Helper: can object sit at (x,y)?
    def obj_ok(x, y):
        if x < 0 or y < 0 or x+ow > W or y+oh > H:
            return False
        return not np.any(occ[y:y+oh, x:x+ow])

    # Initialize with start position and None as previous direction
    init = (start, None)
    open_set = [(abs(start[0]-goal[0]) + abs(start[1]-goal[1]), 0, init)]
    came_from = {}
    g_score = {init: 0}
    
    while open_set:
        if timer_on and (time.time() - start_time > path_finding_timeout):
            return [(-2, -2)], []
        
        _, _, (cur, last_dir) = heapq.heappop(open_set)
        if cur == goal:
            # Reconstruct path
            path = []
            st = (cur, last_dir)
            while st in came_from:
                path.append(st[0])
                st = came_from[st]
            path.append(start)
            path = path[::-1]
            
            # Get turning points
            if len(path) <= 1:
                return path, []
            
            turning_points = [path[0]]
            dirs = []
            
            # Compute initial direction
            if len(path) > 1:
                dx = path[1][0] - path[0][0]
                dy = path[1][1] - path[0][1]
                prev_dir = -1
                if dx == 0 and dy < 0: prev_dir = 0      # up
                elif dx == 0 and dy > 0: prev_dir = 1    # down
                elif dx < 0 and dy == 0: prev_dir = 2    # left
                elif dx > 0 and dy == 0: prev_dir = 3    # right
                
                dirs.append(prev_dir)
            
                for i in range(1, len(path)):
                    if i < len(path) - 1:
                        nx = path[i+1][0] - path[i][0]
                        ny = path[i+1][1] - path[i][1]
                        
                        curr_dir = -1
                        if nx == 0 and ny < 0: curr_dir = 0      # up
                        elif nx == 0 and ny > 0: curr_dir = 1    # down
                        elif nx < 0 and ny == 0: curr_dir = 2    # left
                        elif nx > 0 and ny == 0: curr_dir = 3    # right
                        
                        if curr_dir != prev_dir:
                            turning_points.append(path[i])
                            dirs.append(curr_dir)
                            prev_dir = curr_dir
            
            # Always add the goal (final destination)
            if path[-1] not in turning_points:
                turning_points.append(path[-1])
                if len(dirs) > 0:
                    dirs.append(prev_dir)  # Last direction remains the same
                else:
                    dirs.append(0)  # Default direction if no movement
                
            return path, list(zip(turning_points, dirs))
            
        # Try each direction
        for new_dir, (dx, dy) in DIRS.items():
            nb = (cur[0]+dx, cur[1]+dy)
            
            # Object must fit at new position
            if not obj_ok(*nb):
                continue
            
            # Create occupancy grid with object at new position
            occ2 = occ.copy()
            occ2[nb[1]:nb[1]+oh, nb[0]:nb[0]+ow] = True
            
            # Check if robot can fit at the pushing position for this move
            if not robot_fits_optimized(nb, obj_dim, robot_state, new_dir, occ2, desk_dim)[1]:
                continue
            
            # Direction change penalty
            direction_penalty = 0
            if last_dir is not None and last_dir != new_dir:
                direction_penalty = 0.5  # Penalty for changing direction
                
            # Push new state
            new_state = (nb, new_dir)
            tg = g_score[(cur, last_dir)] + 1 + direction_penalty
            if tg < g_score.get(new_state, float('inf')):
                g_score[new_state] = tg
                came_from[new_state] = (cur, last_dir)
                f = tg + abs(nb[0]-goal[0]) + abs(nb[1]-goal[1])
                heapq.heappush(open_set, (f, tg, new_state))
                
    return [(-2, -2)], []

def get_robot_pushing_position(obj_pos, obj_dim, move_dir, robot_dim, desk_dim):
    """
    Calculate where the robot should be to push the object
    
    obj_pos: (x,y) of object
    obj_dim: (w,h) of object  
    move_dir: direction (0=up, 1=down, 2=left, 3=right)
    robot_dim: (w,h) of robot
    desk_dim: (W,H) desk dimensions
    
    Returns: (rx, ry) position for robot's top-left corner
    """
    x, y = obj_pos
    w, h = obj_dim
    rw, rh = robot_dim
    
    if move_dir == 0:  # object moves up, robot pushes from below
        return (x + w//2 - rw//2, y + h)
    elif move_dir == 1:  # object moves down, robot pushes from above
        return (x + w//2 - rw//2, y - rh)
    elif move_dir == 2:  # object moves left, robot pushes from right
        return (x + w, y + h//2 - rh//2)
    elif move_dir == 3:  # object moves right, robot pushes from left
        return (x - rw, y + h//2 - rh//2)
    else:
        return None

def plan_robot_path_for_pushing(robot_pos, obj_turning_points_with_dirs, obj_dim, occ, desk_dim):
    """Plan robot path between turning points with correct pushing positions"""
    robot_path = []
    current_robot_pos = (robot_pos[0], robot_pos[1])  # Just x,y position for A*
    robot_size = (robot_pos[2], robot_pos[3])
    complete_robot_path = []
    
    # For each segment of the object path
    for i in range(len(obj_turning_points_with_dirs)):
        current_point, current_dir = obj_turning_points_with_dirs[i]
        
        # Build occupancy grid with object at current position
        occ2 = occ.copy()
        ow, oh = obj_dim
        occ2[current_point[1]:current_point[1]+oh, current_point[0]:current_point[0]+ow] = True
        
        # Calculate where robot needs to be to push object
        robot_target_pos = get_robot_pushing_position(current_point, obj_dim, current_dir, robot_size, desk_dim)
        
        # Plan path from current robot position to pushing position
        robot_path_segment = robot_astar_optimized(current_robot_pos, robot_target_pos, robot_size, occ2, desk_dim)
        
        if robot_path_segment[0] == (-2, -2):
            return [(-2, -2)]  # Path finding failed
        
        # Add all points from this segment
        complete_robot_path.extend(robot_path_segment)
        
        # Update current robot position for next segment
        current_robot_pos = robot_target_pos
    
    return complete_robot_path

def path_finder(robot_pos, cur_pos, new_pos, unmoved_contours, unmoved_boxes, moved_boxes, desk_dim):
    """Optimized path finder with numpy operations and caching"""
    W, H = desk_dim
    # Build numpy occupancy grid
    occ = np.zeros((H, W), dtype=bool)

    # Mark moved_boxes as occupied
    for mb_x, mb_y, mb_w, mb_h in moved_boxes.values():
        occ[mb_y:mb_y+mb_h, mb_x:mb_x+mb_w] = True

    # Mark unmoved_contours via a mask
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask,
                 [cnt.reshape(-1, 2) for cnt in unmoved_contours.values()],
                 255)
    occ = occ | (mask > 0)  # Use logical OR for boolean operations
    
    # IMPORTANT: Create a copy of the occupancy grid without the robot's current position
    # This allows the object path planner to consider paths where the robot moves out of the way
    occ_without_robot = occ.copy()
    rx, ry, rw, rh, _ = robot_pos
    # Clear the robot's current footprint from the occupancy grid
    occ_without_robot[ry:ry+rh, rx:rx+rw] = False

    # Extract start, goal, and object dimensions
    start = (cur_pos[1][0], cur_pos[1][1])
    goal = (new_pos[0], new_pos[1])
    obj_dim = (cur_pos[1][2], cur_pos[1][3])

    # First find object path with turning points using the grid without robot
    obj_path, obj_turning_points_with_dirs = object_path_astar_optimized(start, goal, obj_dim, occ_without_robot, robot_pos, desk_dim)
    
    if obj_path[0] == (-2, -2) or not obj_turning_points_with_dirs:
        return unmoved_contours, unmoved_boxes, moved_boxes, [(-2, -2)], []
    
    # Then plan robot path through turning points with correct pushing positions
    # Use the original occupancy grid for robot path planning
    rob_path = plan_robot_path_for_pushing(robot_pos, obj_turning_points_with_dirs, obj_dim, occ, desk_dim)
    
    if rob_path[0] == (-2, -2):
        return unmoved_contours, unmoved_boxes, moved_boxes, [(-2, -2)], []

    # If we got a valid object path, record the move
    moved_boxes[cur_pos[0]] = (new_pos[0], new_pos[1], new_pos[2], new_pos[3])

    return unmoved_contours, unmoved_boxes, moved_boxes, obj_path, rob_path

# is_valid_pos optimized with numpy operations
def is_valid_pos(check_pos, unmoved_contours, moved_boxes, desk_dim, robot_state=None):
    check_x, check_y, obj_w, obj_h = check_pos
    desk_width, desk_height = desk_dim
    
    # checking desk boundaries
    if check_x < 0 or check_y < 0 or check_x+obj_w > desk_width or check_y+obj_h > desk_height:
        return False
    
    # checking moved box boundaries
    for mb_x, mb_y, mb_w, mb_h in moved_boxes.values():
        if not (check_x+obj_w < mb_x or check_x > mb_x+mb_w or check_y+obj_h < mb_y or check_y > mb_y+mb_h):
            return False
    
    # checking unmoved contour boundaries - optimized with mask
    obj_mask = np.zeros((desk_height, desk_width), dtype=np.uint8)
    cv2.rectangle(obj_mask, (check_x, check_y), (check_x+obj_w, check_y+obj_h), 255, -1)
    for contour in unmoved_contours.values():
        contour_mask = np.zeros((desk_height, desk_width), dtype=np.uint8)
        cv2.fillPoly(contour_mask, [contour.reshape(-1, 2)], 255)
        # Check for intersection
        if np.any(obj_mask & contour_mask):
            return False
    
    # NEW: Check if the robot can push the object from at least one direction
    if robot_state is not None:
        # Extract robot dimensions from robot_state (assuming format is consistent with robot_fits_optimized)
        _, _, rw, rh, _ = robot_state
        robot_dim = (rw, rh)
        obj_pos = (check_x, check_y)
        obj_dim = (obj_w, obj_h)
        
        # Create an occupancy grid to check robot fit
        occ = np.zeros((desk_height, desk_width), dtype=np.uint8)
        
        # Mark moved boxes as occupied
        for mb_x, mb_y, mb_w, mb_h in moved_boxes.values():
            cv2.rectangle(occ, (mb_x, mb_y), (mb_x+mb_w, mb_y+mb_h), 1, -1)
        
        # Mark unmoved contours as occupied
        for contour in unmoved_contours.values():
            cv2.fillPoly(occ, [contour.reshape(-1, 2)], 1)
        
        # Check each direction (0=up, 1=down, 2=left, 3=right)
        robot_can_push = False
        for move_dir in range(4):
            robot_pos = get_robot_pushing_position(obj_pos, obj_dim, move_dir, robot_dim, desk_dim)
            rx, ry = robot_pos
            
            # Check robot boundaries
            if rx < 0 or ry < 0 or rx + rw > desk_width or ry + rh > desk_height:
                continue
            
            # Check for collision with other objects
            robot_area = occ[ry:ry+rh, rx:rx+rw]
            if np.any(robot_area):
                continue
            
            # If we reach here, the robot can push from this direction
            robot_can_push = True
            break
        
        if not robot_can_push:
            return False
    
    return True

# finding closest organized location bounding box for given object
def nearest_pos(robot_pos, cur_pos, target_pos, unmoved_contours, unmoved_boxes, moved_boxes, desk_dim):
    target_x, target_y = target_pos
    desk_w, desk_h = desk_dim

    for dx in range(0, desk_w):
        for dy in range(0, desk_h):
            new_x = target_x + dx if target_x == 0 else target_x - dx
            new_y = target_y + dy if target_y == 0 else target_y - dy

            new_pos = (new_x, new_y, cur_pos[1][2], cur_pos[1][3])

            if is_valid_pos(new_pos, unmoved_contours, moved_boxes, desk_dim, robot_pos):
                print("valid_pos found")
                return path_finder(robot_pos, cur_pos, new_pos + (0,), unmoved_contours, unmoved_boxes, moved_boxes, desk_dim)

    return unmoved_contours, unmoved_boxes, moved_boxes, [-2, -2], []           # no path/pos found

# find distance to closest corner
def find_targets(box, desk_dim):
    x, y, w, h = box
    hx, hy = x+w/2, y+h/2
    dhw, dhh = desk_dim[0]/2, desk_dim[1]/2

    # finding closet corner for given object
    if hx<=dhw and hy<=dhh:                                 # nw
        target_x, target_y = 0, 0
        dist = math.hypot(x - 0, y - 0)  # Distance from actual corner of box

    elif hx>dhw and hy>dhh:                                 # se
        target_x, target_y = desk_dim[0]-w, desk_dim[1]-h
        dist = math.hypot(desk_dim[0]-w - x, desk_dim[1]-h - y)

    elif hx>dhw and hy<=dhh:                               # ne
        target_x, target_y = desk_dim[0]-w, 0
        dist = math.hypot(desk_dim[0]-w - x, 0 - y)

    else:                                                   # sw
        target_x, target_y = 0, desk_dim[1]-h
        dist = math.hypot(0 - x, desk_dim[1]-h - y)
        
    return [(target_x, target_y), dist]

# organizing objects
def knoll_loc(result, robot_pos):
    img = result.orig_img
    desk_height, desk_width, _ = img.shape
    desk_dim = (desk_width, desk_height)

    # populating contours and boxes
    contours = result.masks.xy
    boxes = [tuple(map(int, (x1,y1,x2-x1,y2-y1))) for x1,y1,x2,y2 in result.boxes.xyxy.tolist()]

    fixed_contours = {}
    failed_contours = {}
    unmoved_contours = {}

    fixed_boxes = {}
    failed_boxes = {}
    unmoved_boxes = {}
    moved_boxes = {}

    fixed_key = []
    failed_key = []
    failed_run_keys = []
    failed_org_keys = []
    failed_run = False

    for i, contour in enumerate(contours):
        class_id = int(result.boxes.cls[i])
        label = model.names[class_id]
        contour_arr = np.array(contour, dtype=np.int32)

        if label in restricted_items:
            fixed_contours[i] = contour_arr
            fixed_boxes[i] = copy.deepcopy(boxes[i])
            fixed_key.append(i)
        else:
            unmoved_contours[i] = contour_arr
            unmoved_boxes[i] = copy.deepcopy(boxes[i])

    # main function
    while unmoved_boxes | failed_boxes:
        visualize(unmoved_contours, unmoved_boxes, fixed_contours, fixed_boxes, failed_contours, failed_boxes, moved_boxes, [], [], robot_pos)

        targets = {}
        for key, box in unmoved_boxes.items():
            targets[key] = find_targets(box, desk_dim)

        targets = dict(sorted(targets.items(), key=lambda kv: (kv[1][0][0], kv[1][0][1], kv[1][1])))

        if len(unmoved_boxes) == 0:
            if failed_run == False:
                failed_run = True
                failed_org_keys = failed_key
                global path_finding_timeout
                path_finding_timeout = failed_path_finding_timeout

            unmoved_boxes[failed_key[0]] = failed_boxes.pop(failed_key[0])
            unmoved_contours[failed_key[0]] = failed_contours.pop(failed_key[0])
            targets[failed_key[0]] = find_targets(unmoved_boxes[failed_key[0]], desk_dim)
            failed_key.pop(0)

        cur_key = next(iter(targets))
        cur_pos = unmoved_boxes.pop(cur_key)
        cur_cont = unmoved_contours.pop(cur_key)
        target_pos = targets[cur_key][0]

        obstruction_contours = unmoved_contours | failed_contours | fixed_contours
        obstruction_boxes = unmoved_boxes | failed_boxes | fixed_boxes

        # Clear robot fit cache before each move attempt
        clear_robot_fit_cache()

        # moving object
        if failed_run:
            unmoved_contours, unmoved_boxes, moved_boxes, path, rob_path = nearest_pos(robot_pos, (cur_key, cur_pos, cur_cont), target_pos, obstruction_contours, obstruction_boxes, moved_boxes, desk_dim)
            target_options = [(0,0), (desk_dim[0]-cur_pos[2], desk_dim[1]-cur_pos[3]), (desk_dim[0]-cur_pos[2], 0), (0, desk_dim[1]-cur_pos[3])]
            targets_tried = 1
            next_traget_idx = 0

            # try all position locations
            while (path[0] == (-2,-2)) and (targets_tried<=4):
                next_traget_idx = next_traget_idx if target_options[next_traget_idx] != target_pos else next_traget_idx+1
                new_target_pos = target_options[next_traget_idx]
                unmoved_contours, unmoved_boxes, moved_boxes, path, rob_path = nearest_pos(robot_pos, (cur_key, cur_pos, cur_cont), new_target_pos, obstruction_contours, obstruction_boxes, moved_boxes, desk_dim)
                targets_tried += 1

        else:
            unmoved_contours, unmoved_boxes, moved_boxes, path, rob_path = nearest_pos(robot_pos, (cur_key, cur_pos, cur_cont), target_pos, obstruction_contours, obstruction_boxes, moved_boxes, desk_dim)

        # redistribute obstructions
        unmoved_contour_keys = list(unmoved_contours.keys())
        for key in unmoved_contour_keys:
            if key in fixed_key:
                unmoved_contours.pop(key)
                unmoved_boxes.pop(key)
            if key in failed_key:
                unmoved_contours.pop(key)
                unmoved_boxes.pop(key)

        # if object too heavy, add to fixed keys
        if path[0] == (-1,-1):
            fixed_boxes[cur_key] = cur_pos
            fixed_contours[cur_key] = cur_cont
            fixed_key.append(cur_key)

        # if path finding failed
        if path[0] == (-2,-2):
            print("Unable to find path for one object. Moving onto another.")
            failed_boxes[cur_key] = cur_pos
            failed_contours[cur_key] = cur_cont
            failed_key.append(cur_key)

            if failed_run == True:
                failed_run_keys.append(cur_key)
                if set(failed_run_keys) == set(failed_org_keys):
                    print("Unable to find paths for {} objects.".format(len(failed_org_keys)))
                    print("I tried my best! Now it's your turn! :)")
                    return
                
            continue

        # not a failed run
        failed_run = False
        failed_run_keys = []
        failed_org_keys = []

        if len(rob_path) > 0:
            # Update robot position to the last position in the robot path
            last_point = rob_path[-1]
            # The last point is the robot's top-left corner coordinates
            robot_pos = (last_point[0], last_point[1], robot_pos[2], robot_pos[3], 0)

        ground_control(unmoved_contours, unmoved_boxes, fixed_contours, fixed_boxes, failed_contours, failed_boxes, moved_boxes, path, rob_path, robot_pos)

    print("Knolled {} objects.".format(len(moved_boxes)))
    print("Knolling Finished. Yay.")
    return

# inputting picture
def input_picture():

    cropped_image, robot_position, cal_request = cropper.get_cropped_image(save_image=False)

    while cal_request:
        cropper.recalibrate()
        cropped_image, robot_position, cal_request = cropper.get_cropped_image(save_image=False)

    global image
    image = cropped_image
    conf = 0.4

    result = model.predict(image, conf=conf)[0]

    print("")
    print("STARTING TO KNOLL")
    print("")
    input("Press ENTER to continue")

    while result.masks is None:
        result = model.predict(image, conf=conf)[0]

    knoll_loc(result, robot_position)



# SEND COMMANDS - MARY

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
MARKER_IDS = [0, 1, 2]                          # number of aruco markers
CALIBRATION_DIR = "calibration_images"
CHESSBOARD_SIZE = (11, 7)                       # chessboard size
SQUARE_SIZE = 2.1                               # cm
MARKER_LENGTH = 5                               # cm
PIXELS_PER_CM = None
NUM_CALIBRATION_IMGS = 15
CALIBRATION_FILE = "camera_calibration.npz"
CAMERA_INDEX = 0

def calibrate_camera():

    print("")
    print("CALIBRATING CAMERA!")
    print("")

    if os.path.exists(CALIBRATION_FILE):
        print("Loading existing camera calibration...")
        data = np.load(CALIBRATION_FILE)
        camera_matrix, dist_coeffs = data['camera_matrix'], data['dist_coeffs']
        return camera_matrix, dist_coeffs
    
    print(f"Please show a {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} chessboard pattern to the camera.")
    print(f"Need to capture {NUM_CALIBRATION_IMGS} images.")

    os.makedirs(CALIBRATION_DIR)

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE
    objpoints = []  # 3D points
    imgpoints = []  # 2D points in image
    cap = cv2.VideoCapture(CAMERA_INDEX)    
    img_count = 0

    while img_count < NUM_CALIBRATION_IMGS:

        ret, frame = cap.read()

        if not ret:
            input("Could not access camera. Press ENTER to try again.")
            continue

        flipped_frame = cv2.flip(frame, -1)                 # flipped for easy viewing
        gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        display_frame = flipped_frame.copy()
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners2, ret)
            cv2.putText(display_frame, f"Image {img_count+1}/{NUM_CALIBRATION_IMGS}: Press 'y' to capture, anything else to retake.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            cv2.putText(display_frame, "Chessboard not detected.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Camera Calibration", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # when spacebar pressed, take pic and store points
        if (key == ord('y') or key == ord('Y')) and ret:
            img_name = os.path.join(CALIBRATION_DIR, f"calibration_{img_count}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Saved {img_name}")
            objpoints.append(objp)
            imgpoints.append(corners2)
            img_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("Calibrating camera, please wait...")
    
    # calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
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
        return None
    
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    return undistorted

def detect_aruco_markers(frame):

    flipped_frame = cv2.flip(frame, -1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT)
    marker_corners = {}
    marker_centers = {}
    robot_corners = None
    robot_orientation = None
    failure = False

    if ids is None:
        print("No Aruco Markers Detected")
        cv2.putText(flipped_frame, "No Aruco Markers Detected. Enter any key to try again.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Aruco Failure", flipped_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return marker_corners, marker_centers, robot_corners, robot_orientation, True

    undetected_ids = copy.deepcopy(MARKER_IDS)
    for id in ids.flatten():
        if id in undetected_ids:
            undetected_ids.remove(id)
    if 0 in undetected_ids:
        undetected_ids.remove(0)
        print("Robot not detected.")
        failure = True
    if len(undetected_ids) > 0:
        print("IDs not detected:", undetected_ids)
        failure = True

    if failure:
        cv2.putText(flipped_frame, "Aruco Markers Missing. Enter any key to try again.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Aruco Failure", flipped_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return marker_corners, marker_centers, robot_corners, robot_orientation, failure

    for i, marker_id in enumerate(ids.flatten()):

        corner_points = corners[i].reshape(4, 2)
        marker_corners[marker_id] = corner_points
        
        # center
        center_x = int(np.mean(corner_points[:, 0]))
        center_y = int(np.mean(corner_points[:, 1]))
        marker_centers[marker_id] = (center_x, center_y)

        # if robot, store corners
        if marker_id == 0:
            robot_corners = [(int(x), int(y)) for x, y in corner_points]
            dx = corner_points[1][0] - corner_points[0][0]
            dy = corner_points[1][1] - corner_points[0][1]
            angle_rad = np.arctan2(dy, dx)
            robot_orientation = (np.degrees(angle_rad) + 360) % 360
    
    get_pixel_to_cm(robot_corners)
    print("Robot and all Arucos Successfully Detected.")
    return marker_corners, marker_centers, robot_corners, robot_orientation, failure

def get_pixel_to_cm(robot_corners):

    global PIXELS_PER_CM

    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    side_lengths = [
        dist(robot_corners[0], robot_corners[1]),
        dist(robot_corners[1], robot_corners[2]),
        dist(robot_corners[2], robot_corners[3]),
        dist(robot_corners[3], robot_corners[0]),
    ]

    avg_side = np.mean(side_lengths)
    PIXELS_PER_CM = avg_side / MARKER_LENGTH
    return

def get_crop_corners(marker_corners):

    center_1 = np.mean(marker_corners[1], axis=0)
    center_2 = np.mean(marker_corners[2], axis=0)

    if center_1[0] < center_2[0]:
        top_left = marker_corners[1]      # marker on the left (marker 1)
        bottom_right = marker_corners[2]  # marker on the right (marker 2)
    else:
        top_left = marker_corners[2]
        bottom_right = marker_corners[1]
    
    # convert to points
    br_1 = max(top_left, key=lambda pt: np.linalg.norm(pt))        # closest to origin
    tl_2 = min(bottom_right, key=lambda pt: np.linalg.norm(pt))
    
    x1, y1 = int(br_1[0]), int(br_1[1])
    x2, y2 = int(tl_2[0]), int(tl_2[1])
    
    return (x1, y1, x2, y2)

def expand_robot_bbox(robot_corners, extra_width_cm=3.3, extra_height_cm=2.7):
    """
    Expand the ArUco marker bounding box to approximate the full robot size.
    
    Args:
        robot_corners: list of 4 (x, y) tuples, clockwise starting from top-left
        pixels_per_cm: float, pixels per centimeter
        extra_width_cm: extra amount to add to each side along marker's width (left-right)
        extra_height_cm: extra amount to add to each side along marker's height (top-bottom)
    
    Returns:
        expanded_corners: list of 4 expanded corner points (x, y) in pixels
    """

    # Convert to np array
    pts = np.array(robot_corners, dtype=np.float32)

    # Get directional vectors
    vec_x = pts[1] - pts[0]  # marker width (top edge)
    vec_y = pts[3] - pts[0]  # marker height (left edge)

    # Normalize
    unit_x = vec_x / np.linalg.norm(vec_x)
    unit_y = vec_y / np.linalg.norm(vec_y)

    # Expansion in pixels
    dx = unit_x * (extra_width_cm * PIXELS_PER_CM)
    dy = unit_y * (extra_height_cm * PIXELS_PER_CM)

    # Expand each corner accordingly
    expanded_corners = [
        pts[0] - dx - dy,  # top-left
        pts[1] + dx - dy,  # top-right
        pts[2] + dx + dy,  # bottom-right
        pts[3] - dx + dy   # bottom-left
    ]

    return [tuple(map(int, pt)) for pt in expanded_corners]


class CameraCropper:

    def __init__(self, camera_index=CAMERA_INDEX, show_feed=True):
        """Initialize the camera cropper.
        
        Args:
            camera_index: Index of the camera to use
            show_feed: Whether to show live camera feed
        """
        self.camera_index = camera_index
        self.show_feed = show_feed
        self.cap = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.is_initialized = False
        self.counter = 0                # Counter for saved images
        
    def initialize(self):

        # generate_aruco_markers()
        self.camera_matrix, self.dist_coeffs = calibrate_camera()
        while (self.camera_matrix is None) or (self.dist_coeffs is None):
            self.camera_matrix, self.dist_coeffs = calibrate_camera()

        self.cap = cv2.VideoCapture(self.camera_index)
        while True:
            if not self.cap.isOpened():
                input("Could not access camera. Press ENTER to try again.")
                self.cap = cv2.VideoCapture(self.camera_index)
            else:
                break
            
        self.is_initialized = True
        print("Camera Initialized!!")
        return True
        
    def get_cropped_image(self, save_image=False, filename=None):
        """Capture an image and return the cropped version with robot position.
        
        Args:
            save_image: Whether to save the cropped image
            filename: Optional filename for the saved image (if None, generate a name)
            
        Returns:
            tuple: (cropped_image, robot_position, robot_corners)
                - cropped_image: The cropped image if markers were detected, None otherwise
                - robot_position: (x, y) of robot center if detected, None otherwise
                - robot_corners: List of robot corner points if detected, None otherwise
        """

        if not self.is_initialized:
            print("Camera not intialized and requesting cropped image. Will attempt intialization.")
            self.initialize()

        failure = True

        while failure:

            ret, frame = self.cap.read()
            while True:
                if not ret:
                    input("Could not access camera. Press ENTER to try again.")
                    ret, frame = self.cap.read()
                else:
                    break

            # Only works if initialized properly! Check done previously.
            print("Undistorting Image.")
            frame = undistort_image(frame, self.camera_matrix, self.dist_coeffs)
            if frame is None:
                print("Checkpoint 1: camera_matrix is None or dist_coeffs is None for undistort image. Should not happen if intialized properly. Please debug!")
                self.release()
                input("Press ENTER to exit program.")
                exit(0)
                
            # Detect markers
            print("Finding Aruco Markers.")
            marker_corners, marker_centers, robot_corners, robot_orientation, failure = detect_aruco_markers(frame)
        
        crop_coords = get_crop_corners(marker_corners)
            
        # Create display frame with visualizations
        display_frame = frame.copy()
        if marker_corners:
            for marker_id, corners in marker_corners.items():
                cv2.polylines(display_frame, [corners.astype(int)], True, (0, 255, 0), 2)
                if marker_id in marker_centers:
                    center_x, center_y = marker_centers[marker_id]
                    cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(display_frame, f"ID: {marker_id}", (center_x + 10, center_y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw crop rectangle
        x1, y1, x2, y2 = crop_coords
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
        # Crop the image if possible
        print(x1, y1, x2, y2)
        cropped_image = frame[y1:y2, x1:x2]
        cropped_display = cropped_image.copy()
        
        # Update robot position relative to cropped image
        robot_position = expand_robot_bbox(robot_corners)
        robot_position_in_crop = [(x-x1, y-y1) for x, y in robot_position]
        
        # Draw robot position on cropped display
        cv2.polylines(cropped_display, [np.array(robot_position_in_crop, dtype=np.int32)], True, (133, 30, 130), thickness=2)       # purple

        # Calculate the center of the robot polygon
        robot_center = np.mean(np.array(robot_position_in_crop), axis=0).astype(int)
        center_x, center_y = robot_center
        # Define the label and font
        label = "Robot"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        # Get the size of the text box
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        # Coordinates for the white background rectangle
        rect_top_left = (center_x - text_w // 2 - 4, center_y - text_h // 2 - 4)
        rect_bottom_right = (center_x + text_w // 2 + 4, center_y + text_h // 2 + 4)
        # Draw white background rectangle
        cv2.rectangle(cropped_display, rect_top_left, rect_bottom_right, (255, 255, 255), cv2.FILLED)
        # Draw the label text in black (or any color you want)
        text_org = (center_x - text_w // 2, center_y + text_h // 2 - baseline)
        cv2.putText(cropped_display, label, text_org, font, font_scale, (0, 0, 0), thickness)

        # Save the cropped image if requested
        if save_image:
            if filename is None:
                filename = f"cropped_desk_{self.counter}.jpg"
            cv2.imwrite(filename, cropped_image)
            print(f"Saved {filename}")
            self.counter += 1
            
            # Also save a version with the robot marker highlighted
            cv2.imwrite(f"cropped_desk_robot_{self.counter-1}.jpg", cropped_display)

        display_frame = cv2.flip(display_frame, -1)
        cropped_display = cv2.flip(cropped_display, -1)
        cv2.imshow("Camera Feed with Markers", display_frame)
        cv2.imshow("Cropped Desk", cropped_display)
                
        # Process keyboard input
        cal_request = False
        #print("Any Key for Continuing, 'r' for Recalibrating")
        key = ord('n')
        cv2.destroyAllWindows()
        
        # Handle keyboard commands
        if key == ord('r') or key == ord('R'):
            cal_request = True
        
        robot_position_in_crop.append(robot_orientation)

        (x1, y1), (x2, y2), (x3, y3), (x4, y4), orientation_deg = robot_position_in_crop
        width = int(math.hypot(x2 - x1, y2 - y1))
        height = int(math.hypot(x3 - x2, y3 - y2))
        width = max(width, height)
        robot_position_in_crop = (x1, y1, width, width, orientation_deg)
                
        return cropped_image, robot_position_in_crop, cal_request
    
    def recalibrate(self):
        """Force recalibration of the camera."""

        if os.path.exists(CALIBRATION_FILE):
            os.remove(CALIBRATION_FILE)

        self.camera_matrix, self.dist_coeffs = calibrate_camera()
        while (self.camera_matrix is None) or (self.dist_coeffs is None):
            self.camera_matrix, self.dist_coeffs = calibrate_camera()

        return True
        
    def release(self):
        """Release the camera resources."""

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.is_initialized = False
        print("Camera resources released.")


# main function
if __name__ == '__main__':

    cropper = CameraCropper(show_feed=True)
    cropper.initialize()
    input_picture()