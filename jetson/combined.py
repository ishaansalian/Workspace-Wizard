from ultralytics import YOLO
import cv2
import networkx as nx
import numpy as np
import math
import heapq
import copy
import time

import cv2.aruco as aruco
import os

import asyncio
from bleak import BleakClient, BleakScanner
import re


model = YOLO("yolov8m-seg.pt")
restricted_items = ["knife", "scissors"]
path_finding_timeout = 300           # in sec
failed_path_finding_timeout = 600
timer_on = False
IMAGE = None


def ground_control(unmoved_contours, unmoved_boxes, fixed_contours, fixed_boxes, failed_contours, failed_boxes, moved_boxes, path, robot_path, robot_pos):

    if len(robot_path) < 2:
        return []
    
    robot_path = [point for sublist in robot_path for point in sublist]

    cleaned = [robot_path[0]]
    for point in robot_path[1:]:
        if point != cleaned[-1]:
            cleaned.append(point)

    robot_path = copy.deepcopy(cleaned)
    
    robot_turning_points = [robot_path[0]]
    prev_dx = robot_path[1][0] - robot_path[0][0]
    prev_dy = robot_path[1][1] - robot_path[0][1]

    for i in range(2, len(robot_path)):
        curr_dx = robot_path[i][0] - robot_path[i-1][0]
        curr_dy = robot_path[i][1] - robot_path[i-1][1]

        if (curr_dx, curr_dy) != (prev_dx, prev_dy):
            robot_turning_points.append(robot_path[i-1])
        prev_dx, prev_dy = curr_dx, curr_dy

    for i in range(len(robot_turning_points)-1):

        x, y, w, h, angle = robot_pos
        
        move_robot(robot_turning_points[i][0], robot_turning_points[i][1], angle, robot_turning_points[i+1][0], robot_turning_points[i+1][1])

        robot_move = [robot_turning_points[i], robot_turning_points[i+1]]
        robot_cur_pos = (math.ceil(robot_move[1][0]-w/2), math.ceil(robot_move[1][1]-h/2), w, h, angle)
        visualize(unmoved_contours, unmoved_boxes, fixed_contours, fixed_boxes, failed_contours, failed_boxes, moved_boxes, path, robot_move, robot_cur_pos)

        cropped_image, robot_position, robot_corners = cropper.get_cropped_image(save_image=False)
        while cropped_image is None:
            print("Image issue... None!")
            cropped_image, robot_position, robot_corners = cropper.get_cropped_image(save_image=False)
 
        global IMAGE
        IMAGE = cropped_image
    
    return


# visualize code
def visualize(unmoved_contours, unmoved_boxes, fixed_contours, fixed_boxes, failed_contours, failed_boxes, moved_boxes, path, rob_path, robot_pos):

    # visualization colors
    unmoved_color = (0, 0, 128)         # maroon
    fixed_color = (0, 255, 255)         # yellow
    failed_color = (0, 0, 255)          # red

    moved_color = (0, 255, 0)           # green

    path_color = (255, 0, 0)            # blue
    robot_color = (43, 64, 6)           # black

    # visual of change
    global IMAGE
    img_copy = IMAGE.copy()

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
    center = (x + w/2, y + h/2)
    rect = (center, (w, h), angle)
    box = np.intp(cv2.boxPoints(rect))
    cv2.fillPoly(img_copy, [box], robot_color)

    cv2.imshow("image", img_copy)
    cv2.waitKey(0)


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def robot_fits(obj_pos, obj_dim, robot_pos, move_dir, obstacles, desk_dim):

    if move_dir == 0:                                                   # Object up, robot down
        robot_w = robot_pos[2]
        robot_h = robot_pos[3]
        robot_x = int(obj_pos[0] + obj_dim[0]/2 - robot_w/2)            # obj_hw - robot_hw
        robot_y = obj_pos[1] + obj_dim[1] + 1                           # obj_y + obj_h + 1
    elif move_dir == 1:                                                 # Object down, robot up
        robot_w = robot_pos[2]
        robot_h = robot_pos[3]
        robot_x = int(obj_pos[0] + obj_dim[0]/2 - robot_w/2)            # obj_hw - robot_hw
        robot_y = obj_pos[1] - robot_pos[3] - 1                         # obj_y - robot_h - 1
    elif move_dir == 2:                                                 # Object left, robot right
        robot_w = robot_pos[3]
        robot_h = robot_pos[2]
        robot_x = obj_pos[0] + obj_dim[0] + 1                           # obj_x + obj_w + 1
        robot_y = int(obj_pos[1] + obj_dim[1]/2 - robot_h/2)            # obj_hh - robot_hh
    elif move_dir == 3:                                                 # Object right, robot left
        robot_w = robot_pos[3]
        robot_h = robot_pos[2]
        robot_x = obj_pos[0] - robot_w - 1                              # obj_x - robot_w - 1
        robot_y = int(obj_pos[1] + obj_dim[1]/2 - robot_h/2)            # obj_hh - robot_hh
    else:
        return False

    # Check if the entire robot fits
    return (robot_x, robot_y, robot_w, robot_h, move_dir), object_fits((robot_x, robot_y), (robot_w, robot_h), obstacles, desk_dim)


def object_fits(cur_pos, obj_dim, obstacles, desk_dim):

    x1 = cur_pos[0]
    x2 = cur_pos[0]+obj_dim[0]
    y1 = cur_pos[1]
    y2 = cur_pos[1]+obj_dim[1]

    if x1<0 or x2>desk_dim[0] or y1<0 or y2>desk_dim[1]:
        return False

    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            if (x, y) in obstacles:
                return False

    return True


# for robot astar 
def add_back_nodes(G, nodes, desk_dim):

    for node in nodes:

        G.add_node(node)
        x, y = node

        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for n in neighbors:
            if 0 <= n[0] < desk_dim[0] and 0 <= n[1] < desk_dim[1]:
                if n in G:
                    G.add_edge(node, n)


def robot_astar(start, goal, robot_dim, G, obstacles, desk_dim, obj_pos, obj_dim):

    cur_obstacle = set()
    for ob_x in range(obj_pos[0], obj_pos[0]+obj_dim[0]):
        for ob_y in range(obj_pos[1], obj_pos[1]+obj_dim[1]):
            cur_obstacle.add((ob_x, ob_y))

    new_obstacles = obstacles | cur_obstacle
    G.remove_nodes_from(cur_obstacle)

    start_time = time.time()

    initial_state = start
    open_set = []
    heapq.heappush(open_set, (0, initial_state))
    
    came_from = {}
    g_score = {initial_state: 0}
    f_score = {initial_state: heuristic(start, goal)}
    
    # Assume a global path_finding_timeout is defined
    while open_set:

        if timer_on and (time.time() - start_time > path_finding_timeout):

            add_back_nodes(G, cur_obstacle, desk_dim)
            return [(-2, -2)]
        
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            add_back_nodes(G, cur_obstacle, desk_dim)
            adjusted_path = [(int(x+robot_dim[0]/2), int(y+robot_dim[1]/2)) for (x, y) in path]
            return adjusted_path
        
        if current not in G:
            continue
        neighbors = list(copy.deepcopy(G[current]))
        for neighbor in neighbors:
            # Check if the robot fits at the neighbor cell using our simplified robot_fits_at
            if not object_fits(neighbor, robot_dim, new_obstacles, desk_dim):
                continue
            
            tentative_g = g_score[current] + 1  # uniform cost for each move
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    add_back_nodes(G, cur_obstacle, desk_dim)
    return [(-2, -2)]


def custom_astar(start, goal, obj_dim, G, obstacles, robot_pos, desk_dim):

    robot_size = robot_pos[2] if robot_pos[2] > robot_pos[3] else robot_pos[3]
    start_time = time.time()                                # starting time

    initial_state = (start, None, robot_pos)                # ((x, y) (x_old, y_old) (x, y, w, h, direction))

    open_set = []                                           # priority queue (min-heap)
    heapq.heappush(open_set, (0, initial_state))            # (f-cost, node)

    came_from = {}                                          # node tracker for each node

    g_score = {initial_state: 0}                            # cost from start to node
    f_score = {initial_state: heuristic(start, goal)}       # estimated cost to goal

    while open_set:

        if timer_on and (time.time() - start_time > path_finding_timeout):
            return [(-2, -2)], []

        _, current_state = heapq.heappop(open_set)          # best node next (lowest f score)
        (current_node, last_move_dir, current_robot_state) = current_state

        if current_node == goal:                            # goal reached

            obj_path = []
            robot_paths = []
            state = current_state

            while state in came_from:
                parent, robot_seg = came_from[state]
                obj_path.append(state[0])
                if robot_seg:
                    robot_paths.insert(0, robot_seg)        # prepend
                state = parent

            obj_path.append(start)
            obj_path.reverse()
            adjusted_path = [(int(x+obj_dim[0]/2), int(y + obj_dim[1]/2)) for (x, y) in obj_path]
            return adjusted_path, robot_paths
        
        neighbors = list(copy.deepcopy(G[current_node]))
        for neighbor in neighbors:                    # looping thorough all possible neighbors

            new_dir = 0 if neighbor[1] < current_node[1] else 1 if neighbor[1] > current_node[1] else 2 if neighbor[0] < current_node[0] else 3

            new_robot_state, robot_fit_check = robot_fits(neighbor, obj_dim, current_robot_state, new_dir, obstacles, desk_dim)

            # making sure object and robot can fit in new location
            if not object_fits(neighbor, obj_dim, obstacles, desk_dim) or not robot_fit_check:
                continue
                
            tentative_g = g_score[current_state] + 1
            new_state = (neighbor, new_dir, new_robot_state)

            robot_subpath = robot_astar(current_robot_state[:2], new_robot_state[:2], (robot_size, robot_size),  G, obstacles, desk_dim, neighbor, obj_dim)

            if robot_subpath[0] == (-2, -2):
                continue
            
            if tentative_g < g_score.get(new_state, float('inf')):
                came_from[new_state] = (current_state, robot_subpath)
                g_score[new_state] = tentative_g
                f_score[new_state] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[new_state], new_state))

    return [(-2,-2)], []


def path_finder(robot_pos, cur_pos, new_pos, unmoved_contours, unmoved_boxes, moved_boxes, desk_dim):

    G = nx.grid_2d_graph(desk_dim[0], desk_dim[1])

    # Add moved_boxes points to obstacles
    obstacle_set = set()
    for mb_x, mb_y, mb_w, mb_h in moved_boxes.values():
        for ob_x in range(mb_x, mb_x+mb_w+1):
            for ob_y in range(mb_y, mb_y+mb_h+1):
                obstacle_set.add((ob_x, ob_y))

    # Add unmoved_contours points to obstacles
    mask = np.zeros((desk_dim[1], desk_dim[0]), dtype=np.uint8)
    cv2.fillPoly(mask, [contour.reshape(-1, 2) for contour in unmoved_contours.values()], 255)
    y_indices, x_indices = np.nonzero(mask)
    obstacle_set.update(zip(x_indices, y_indices))

    # Remove obstacles from the graph
    for node in list(G.nodes):
        if node in obstacle_set:
            G.remove_node(node)
            continue

    # Restrict movement to only 90-degree turns (no diagonals)
    for node in list(G.nodes):
        x, y = node
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for n in list(G.neighbors(node)):
            if n not in neighbors:
                G.remove_edge(node, n)
    
    # find path
    start = (cur_pos[1][0], cur_pos[1][1])
    end = (new_pos[0], new_pos[1])
    obj_dim = (cur_pos[1][2], cur_pos[1][3])

    obj_path, rob_path = custom_astar(start, end, obj_dim, G, obstacle_set, robot_pos, desk_dim)

    if obj_path[0] != (-1,-1) and obj_path[0] != (-2,-2):
        moved_boxes[cur_pos[0]] = new_pos

    return unmoved_contours, unmoved_boxes, moved_boxes, obj_path, rob_path


# checking if organized location bounding box is valid
def is_valid_pos(check_pos, unmoved_contours, moved_boxes, desk_dim):

    check_x, check_y, obj_w, obj_h = check_pos
    desk_width, desk_height = desk_dim

    # checking desk boundaries
    if check_x < 0 or check_y < 0 or check_x+obj_w > desk_width or check_y+obj_h > desk_height:
        return False
    
    # checking moved box boundaries
    for mb_x, mb_y, mb_w, mb_h in moved_boxes.values():
        if not (check_x+obj_w < mb_x or check_x > mb_x+mb_w or check_y+obj_h < mb_y or check_y > mb_y+mb_h):
            return False
        
    # checking unmoved contour boundaries
    for contour in unmoved_contours.values():
        obj_rect = np.array([[check_x, check_y], [check_x+obj_w, check_y], [check_x+obj_w, check_y+obj_h], [check_x, check_y+obj_h]], dtype=np.int32)
        intersection_area, _ = cv2.intersectConvexConvex(contour, obj_rect)
        if intersection_area != 0:
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

            if is_valid_pos(new_pos, unmoved_contours, moved_boxes, desk_dim):
                return path_finder(robot_pos, cur_pos, new_pos, unmoved_contours, unmoved_boxes, moved_boxes, desk_dim)

    return unmoved_contours, unmoved_boxes, moved_boxes, [-2, -2], []           # not path/pos found


# find distance to closest corner
def find_targets(box, desk_dim):

    x, y, w, h = box
    hx, hy = x+w/2, y+h/2
    dhw, dhh = desk_dim[0]/2, desk_dim[1]/2

    # finding closet corner for given object
    if hx<=dhw and hy<=dhh:                                 # nw
        target_x, target_y = 0, 0
        dist = math.hypot(hx, hy)

    elif hx>dhw and hy>dhh:                                 # se
        target_x, target_y = desk_dim[0]-w, desk_dim[1]-h
        dist = math.hypot(desk_dim[0]-hx, hy)

    elif hx>dhw and hy<=desk_dim[1]:                        # ne
        target_x, target_y = desk_dim[0]-w, 0
        dist = math.hypot(hx, desk_dim[0]-hy)

    else:                                                   # sw
        target_x, target_y = 0, desk_dim[1]-h
        dist = math.hypot(desk_dim[0]-hx, desk_dim[1]-hy)
        
    return [(target_x, target_y), dist]


# organizing objects
def knoll_loc(result, robot_pos):

    img = result.orig_img
    desk_height, desk_width, _ = img.shape
    desk_dim = (desk_width, desk_height)

    # populating contours and boxes
    contours = result.masks.xy
    boxes = [tuple(map(int, (x1,y1,x2-x1,y2-y1))) for x1,y1,x2,y2 in result.boxes.xyxy.tolist()]

    fixed_contours = {}                 # {1: array([[x1,y1],[x2,y2], dtype]),2:...}
    failed_contours = {}
    unmoved_contours = {}

    fixed_boxes = {}                    # {1:(x1,y1,w,h),2:...}
    failed_boxes = {}
    unmoved_boxes = {}
    moved_boxes = {}

    fixed_key = []                      # keys
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
        cur_pos = unmoved_boxes.pop(cur_key)            # pops out dictionary item based on target, distance
        cur_cont = unmoved_contours.pop(cur_key)
        target_pos = targets[cur_key][0]

        obstruction_contours = unmoved_contours | failed_contours | fixed_contours
        obstruction_boxes = unmoved_boxes | failed_boxes | fixed_boxes

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

        robot_pos = (rob_path[-1][-1][0], rob_path[-1][-1][1], robot_pos[2], robot_pos[3], 0)
        robot_pos = (math.ceil(robot_pos[0]-robot_pos[2]/2), math.ceil(robot_pos[1]-robot_pos[3]/2), robot_pos[2], robot_pos[3], 0)

        ground_control(unmoved_contours, unmoved_boxes, fixed_contours, fixed_boxes, failed_contours, failed_boxes, moved_boxes, path, rob_path, robot_pos)


    print("Knolled {} objects.".format(len(moved_boxes)))
    print("Knolling Finished. Yay.")
    return


# inputting picture
def input_picture():

    cropped_image, robot_position, robot_orientation, cal_request = cropper.get_cropped_image(save_image=False)

    while cal_request:
        cropper.recalibrate()
        cropped_image, robot_position, robot_orientation, cal_request = cropper.get_cropped_image(save_image=False)

    global IMAGE
    IMAGE = cropped_image
    conf = 0.4

    result = model.predict(IMAGE, conf=conf)[0]
    robot_position.append(robot_orientation)

    print("")
    print("STARTING TO KNOLL")
    print("")
    print(robot_position)
    input("Press ENTER to continue")

    knoll_loc(result, robot_position)


# SEND COMMANDS - ISHAAN

CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
VALID_COMMANDS = {"F", "B", "L", "R", "N", "M", "O", "P"}
CUSTOM_MESSAGES = {"finished", "stuck"}
COMMAND_DELAY = 1.0                         # seconds between sending commands
MOVEMENT_UNIT = 1.0                         # cm 


async def move_robot(current_x, current_y, current_orientation, target_x, target_y):

    robot_x_cm, robot_y_cm = pixels_to_cm(current_x, current_y, PIXEL_TO_CM_RATIO)
    target_x_cm, target_y_cm = pixels_to_cm(target_x, target_y, PIXEL_TO_CM_RATIO)
    commands = generate_movement_commands(robot_x_cm, robot_y_cm, current_orientation, target_x_cm, target_y_cm)

    print("Sending commands to robot to navigate to marker 3...")
    success = await send_commands_ble(commands)
    if success:
        print("Commands sent successfully!")
    else:
        print("Failed to send commands.")


def calculate_pixel_to_cm_ratio(marker_corners):

    global PIXEL_TO_CM_RATIO
    
    if not all(marker_id in marker_corners for marker_id in [1, 2, 3, 4]):
        return None
    
    top_left = marker_corners[1][2]     # bottom right of marker 1
    top_right = marker_corners[2][3]    # bottom left of marker 2
    
    pixel_distance = np.sqrt((top_right[0] - top_left[0])**2 + (top_right[1] - top_left[1])**2)
    known_distance_cm = 50.0            # adjust this value based on your actual setup
    
    PIXEL_TO_CM_RATIO = known_distance_cm/pixel_distance
    return PIXEL_TO_CM_RATIO


def pixels_to_cm(x_px, y_px, ratio):

    if ratio is None:
        return None, None
    
    x_cm = x_px * ratio
    y_cm = y_px * ratio
    return x_cm, y_cm


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
    ADDRESS = await find_device("ESP32_Robot")  # Adjust to match your device's name
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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT)
    marker_corners = {}
    marker_centers = {}
    robot_corners = None
    robot_orientation = None
    failure = False

    if ids is None:
        print("No Aruco Markers Detected")
        cv2.putText(frame, "No Aruco Markers Detected. Enter any key to try again.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Aruco Failure", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return marker_corners, marker_centers, robot_corners, robot_orientation, True

    undetected_ids = copy.deepcopy(MARKER_IDS)
    for id in ids.flatten():
        if id in undetected_ids:
            undetected_ids.remove(id)
    if 0 in undetected_ids:
        undetected_ids.remove(id)
        print("Robot not detected.")
        failure = True
    if len(undetected_ids) > 0:
        print("IDs not detected:", undetected_ids)
        failure = True

    if failure:
        cv2.putText(frame, "Aruco Markers Missing. Enter any key to try again.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Aruco Failure", frame)
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

    top_left = marker_corners[1]            # marker 1
    bottom_right = marker_corners[2]        # marker 2
    
    # convert to points
    br_1 = max(top_left, key=lambda pt: (pt[1], pt[0]))         # bottom-most right corner of marker 1
    tl_2 = min(bottom_right, key=lambda pt: (pt[1], pt[0]))     # top-most left corner of marker 2
    
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
        x1, y1, x2, y2 = crop_coords
        cropped_image = frame[y1:y2, x1:x2]
        cropped_display = cropped_image.copy()
        
        # Update robot position relative to cropped image
        robot_position = expand_robot_bbox(robot_corners)
        robot_position_in_crop = [(x-x1, y-y1) for x, y in robot_position]
        
        # Draw robot position on cropped display
        cv2.polylines(cropped_display, [np.array(robot_position_in_crop, dtype=np.int32)], True, (133, 30, 130), thickness=2)

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

        cv2.imshow("Camera Feed with Markers", display_frame)
        cv2.imshow("Cropped Desk", cropped_display)
                
        # Process keyboard input
        cal_request = False
        print("Any Key for Continuing, 'r' for Recalibrating")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Handle keyboard commands
        if key == ord('r') or key == ord('R'):
            cal_request = True
                
        return cropped_image, robot_position_in_crop, robot_orientation, cal_request
    
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