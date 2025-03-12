from ultralytics import YOLO
import cv2
import networkx as nx
import numpy as np
import math
import heapq
import copy
import time

model = YOLO("yolov8m-seg.pt")
restricted_items = ["knife"]
path_finding_timeout = 300           # in sec
failed_path_finding_timeout = 600
timer_on = True


def send_mary_path(robot_path):

    if len(robot_path) < 2:
        return []
    
    robot_path = [point for sublist in robot_path for point in sublist]

    cleaned = [robot_path[0]]
    for point in robot_path[1:]:
        if point != cleaned[-1]:
            cleaned.append(point)

    robot_path = copy.deepcopy(cleaned)

    print(robot_path)
    print()
    
    robot_turning_points = []
    prev_dx = robot_path[1][0] - robot_path[0][0]
    prev_dy = robot_path[1][1] - robot_path[0][1]

    for i in range(2, len(robot_path)):
        curr_dx = robot_path[i][0] - robot_path[i-1][0]
        curr_dy = robot_path[i][1] - robot_path[i-1][1]

        if (curr_dx, curr_dy) != (prev_dx, prev_dy):
            robot_turning_points.append(robot_path[i-1])
        prev_dx, prev_dy = curr_dx, curr_dy

    print(robot_turning_points)
    
    return robot_turning_points


# visualize code
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
    rob_path = [point for sublist in rob_path for point in sublist]
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

        visualize(unmoved_contours, unmoved_boxes, fixed_contours, fixed_boxes, failed_contours, failed_boxes, moved_boxes, path, rob_path, robot_pos)
        robot_pos = (rob_path[-1][-1][0], rob_path[-1][-1][1], robot_pos[2], robot_pos[3], 0)
        send_mary_path(rob_path)


    print("Knolled {} objects.".format(len(moved_boxes)))
    print("Knolling Finished. Yay.")
    return


# inputting picture
def input_picture():

    global image 
    image = cv2.imread("desk3.jpg")
    conf = 0.4
    result = model.predict(image, conf=conf)[0]
    
    robot_pos = (800, 600, 10, 10, 0)             # (x1, y1, w, h, direction)
    knoll_loc(result, robot_pos)


# main function
if __name__ == '__main__':
    input_picture()