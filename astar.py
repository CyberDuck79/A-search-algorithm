import pygame
from queue import PriorityQueue
from typing import List, Tuple, Callable

### A refaire en C++ avec minilibx ?

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption('A* path finding')
COLOR = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0),
    "grey": (128, 128, 128),
    "turquoise": (64, 224, 208)
}

class Node:
    def __init__(self, row: int, col: int, width: int):
        self._row = row
        self._col = col
        self._draw_info = (row * width, col * width, width, width)
        self._color = COLOR["white"]
        self.neighbors = []
        self.g_score = float("inf")
        self.f_score = float(0)
        self.came_from = None

    def get_pos(self) -> Tuple[int, int]:
        return self._row, self._col

    def is_barrier(self) -> bool:
        return self._color == COLOR["black"]

    def reset(self) -> None:
        self._color = COLOR["white"]

    def set_close(self) -> None:
        self._color = COLOR["red"]

    def set_open(self) -> None:
        self._color = COLOR["green"]

    def set_barrier(self) -> None:
        self._color = COLOR["black"]

    def set_start(self) -> None:
        self._color = COLOR["orange"]

    def set_end(self) -> None:
        self._color = COLOR["turquoise"]

    def set_path(self) -> None:
        self._color = COLOR["purple"]

    def draw(self, win: pygame.Surface) -> None:
        pygame.draw.rect(win, self._color, self._draw_info)

    def update_neighbors(self, grid: List[list]) -> None:
        self.neighbors.clear()
        if self._row < len(grid) - 1:
            neighbor = grid[self._row + 1][self._col]
            if not neighbor.is_barrier():
                self.neighbors.append(neighbor)
        if self._row:
            neighbor = grid[self._row - 1][self._col]
            if not neighbor.is_barrier():
                self.neighbors.append(neighbor)
        if self._col < len(grid) - 1:
            neighbor = grid[self._row][self._col + 1]
            if not neighbor.is_barrier():
                self.neighbors.append(neighbor)
        if self._col:
            neighbor = grid[self._row][self._col - 1]
            if not neighbor.is_barrier():
                self.neighbors.append(neighbor)

def h(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def make_grid(rows: int, width: int) -> List[List[Node]]:
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            grid[i].append(Node(i, j, gap))
    return grid

def draw_grid(win: pygame.Surface, rows: int, width: int) -> None:
    gap = width // rows
    for i in range(rows):
        x_pos = i * gap
        pygame.draw.line(win, COLOR["grey"], (0, x_pos), (width, x_pos))
        for j in range(rows):
            y_pos = j * gap
            pygame.draw.line(win, COLOR["grey"], (y_pos, 0), (y_pos, width))

def draw(win: pygame.Surface, grid: List[List[Node]], rows: int, width: int) -> None:
    for row in grid:
        for node in row:
            node.draw(win)
    draw_grid(win, rows, width)


def get_click_coord(pos: Tuple[int, int], rows: int, width: int) -> Tuple[int, int]:
    gap = width // rows
    return pos[0] // gap, pos[1] // gap

def reconstruct_path(current: Node, draw: Callable) -> None:
    current.set_path()
    while current.came_from:
        current = current.came_from
        current.set_path()
        draw()

def not_in_queue(item: Node, queue: List) -> bool:
    for entries in queue:
        if entries[2] == item:
            return False
    return True

def algorithm(draw: Callable, start: Node, end: Node) -> bool:
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    start.g_score = 0
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        current: Node = open_set.get()[2]
        if current == end:
            reconstruct_path(current, draw)
            return True
        current.set_close()
        for neighbor in current.neighbors:
            if current.g_score + 1 < neighbor.g_score:
                neighbor.g_score = current.g_score + 1
                neighbor.f_score = neighbor.g_score + h(neighbor.get_pos(), end.get_pos())
                neighbor.came_from = current
                if not_in_queue(neighbor, open_set.queue):
                    count += 1
                    open_set.put((neighbor.f_score, count, neighbor))
                    neighbor.set_open()
        draw()
    return False

def main(win: pygame.Surface, width: int) -> None:
    ROWS = 50
    grid = make_grid(ROWS, width)
    start = None
    end = None
    started = False
    while True:
        draw(win, grid, ROWS, width)
        if started:
            for row in grid:
                for node in row:
                    node.update_neighbors(grid)
            algorithm(lambda: draw(win, grid, ROWS, width), start, end)
            started = not started
            continue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    started = not started
                elif event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
                continue
        click = pygame.mouse.get_pressed()
        if click:
            row, col = get_click_coord(pygame.mouse.get_pos(), ROWS, width)
            node = grid[row][col]
            if click[0]:
                if node != end and not start:
                    node.set_start()
                    start = node
                elif node != start and not end:
                    node.set_end()
                    end = node
                elif node != start and node != end:
                    node.set_barrier()
            elif click[2]:
                if node == start:
                    start = None
                elif node == end:
                    end = None
                node.reset()

main(WIN, WIDTH)
