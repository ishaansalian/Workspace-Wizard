import pygame
import heapq
from pygame.locals import *

# Constants
WIDTH, HEIGHT = 800, 600
ROWS, COLS = 30, 40
GRID_COLOR = (200, 200, 200)
WALL_COLOR = (0, 0, 0)
START_COLOR = (0, 255, 0)
END_COLOR = (255, 0, 0)
PATH_COLOR = (255, 255, 0)
OPEN_COLOR = (173, 216, 230)
CLOSED_COLOR = (128, 128, 128)

# Directions for neighbors (up, down, left, right)
DIRECTIONS = [
    (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal directions
    (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal directions
]

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Pathfinding Visualizer")

# Load the background image
background = pygame.image.load("desk.jpg")
background = pygame.transform.scale(background, (WIDTH, HEIGHT))

# Cell size
CELL_WIDTH = WIDTH // COLS
CELL_HEIGHT = HEIGHT // ROWS

# Create grid
grid = [["empty" for _ in range(COLS)] for _ in range(ROWS)]

# Heuristic function (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Draw the grid and background
def draw_grid():
    screen.blit(background, (0, 0))
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            if grid[row][col] == "wall":
                pygame.draw.rect(screen, WALL_COLOR, rect)
            elif grid[row][col] == "start":
                pygame.draw.rect(screen, START_COLOR, rect)
            elif grid[row][col] == "end":
                pygame.draw.rect(screen, END_COLOR, rect)
            elif grid[row][col] == "path":
                pygame.draw.rect(screen, PATH_COLOR, rect)
            elif grid[row][col] == "open":
                pygame.draw.rect(screen, OPEN_COLOR, rect)
            elif grid[row][col] == "closed":
                pygame.draw.rect(screen, CLOSED_COLOR, rect)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)  # Grid lines
    pygame.display.update()

# Reconstruct path
def reconstruct_path(came_from, current):
    while current in came_from:
        current = came_from[current]
        grid[current[0]][current[1]] = "path"
        draw_grid()

# A* algorithm
def a_star(start, end):
    count = 0
    open_set = []
    heapq.heappush(open_set, (0, count, start))
    came_from = {}
    g_score = { (row, col): float("inf") for row in range(ROWS) for col in range(COLS) }
    g_score[start] = 0
    f_score = { (row, col): float("inf") for row in range(ROWS) for col in range(COLS) }
    f_score[start] = heuristic(start, end)

    open_set_hash = {start}

    while open_set:
        current = heapq.heappop(open_set)[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, current)
            return True

        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < ROWS and 0 <= neighbor[1] < COLS):
                continue
            if grid[neighbor[0]][neighbor[1]] == "wall":
                continue

            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor, end)
                if neighbor not in open_set_hash:
                    count += 1
                    heapq.heappush(open_set, (f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
        if grid[current[0]][current[1]] not in ("start", "end"):
            grid[current[0]][current[1]] = "closed"
        draw_grid()

    return False

# Main loop
def main():
    start = None
    end = None
    running = True

    while running:
        draw_grid()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if pygame.mouse.get_pressed()[0]:  # Left mouse button
                pos = pygame.mouse.get_pos()
                row, col = pos[1] // CELL_HEIGHT, pos[0] // CELL_WIDTH

                if not start and grid[row][col] == "empty":
                    grid[row][col] = "start"
                    start = (row, col)
                elif not end and grid[row][col] == "empty":
                    grid[row][col] = "end"
                    end = (row, col)
                elif grid[row][col] == "empty":
                    grid[row][col] = "wall"

            elif pygame.mouse.get_pressed()[2]:  # Right mouse button
                pos = pygame.mouse.get_pos()
                row, col = pos[1] // CELL_HEIGHT, pos[0] // CELL_WIDTH
                if grid[row][col] == "wall":
                    grid[row][col] = "empty"
                elif grid[row][col] == "start":
                    grid[row][col] = "empty"
                    start = None
                elif grid[row][col] == "end":
                    grid[row][col] = "empty"
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    a_star(start, end)
                if event.key == pygame.K_c:
                    start, end = None, None
                    for row in range(ROWS):
                        for col in range(COLS):
                            grid[row][col] = "empty"

    pygame.quit()

if __name__ == "__main__":
    main()
