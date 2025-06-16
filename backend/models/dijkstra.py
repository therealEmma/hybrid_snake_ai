import asyncio
import platform
import random
from heapq import heappush, heappop
from typing import List, Tuple, Optional
from enum import Enum
import time
import tracemalloc
from .common import DeathCause

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE
TIMEOUT_MOVES = 5000

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class PathfindingAlgorithm:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.direction_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        return [(x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if 0 <= x + dx < self.width and 0 <= y + dy < self.height]

    def dijkstra(self, start: Tuple[int, int], goal: Tuple[int, int],
                 obstacles: set) -> List[Tuple[int, int]]:
        if start == goal or goal in obstacles:
            return []

        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            current = heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                if neighbor in obstacles:
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    heappush(open_set, (g_score[neighbor], neighbor))

        return []

    def get_safe_directions(self, head: Tuple[int, int], snake: List[Tuple[int, int]]) -> List[int]:
        safe_directions = []
        obstacles = set(snake)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for i, (dx, dy) in enumerate(directions):
            next_pos = (head[0] + dx, head[1] + dy)
            if (0 <= next_pos[0] < self.width and
                0 <= next_pos[1] < self.height and
                next_pos not in obstacles):
                safe_directions.append(i)
        return safe_directions

class SnakeGame:
    def __init__(self):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.pathfinder = PathfindingAlgorithm(self.width, self.height)
        self.start_time = 0
        self.decision_times = []
        self.path_lengths = []
        self.optimal_path_lengths = []
        self.death_cause = None
        self.reset()

    def reset(self):
        center_x, center_y = self.width // 2, self.height // 2
        self.snake = [(center_x, center_y), (center_x - 1, center_y), (center_x - 2, center_y)]
        self.direction = Direction.RIGHT
        self.food = self._generate_food()
        self.score = 0
        self.moves = 0
        self.game_over = False
        self.start_time = 0
        self.survival_time = 0.0
        self.decision_times = []
        self.path_lengths = []
        self.optimal_path_lengths = []
        self.death_cause = None

    def _generate_food(self):
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food

    def get_action(self):
        start_time = time.time()
        head = self.snake[0]
        snake_body = set(self.snake[1:])
        optimal_path = self.pathfinder.dijkstra(head, self.food, snake_body)
        self.optimal_path_lengths.append(len(optimal_path) if optimal_path else self.width + self.height)

        path = self.pathfinder.dijkstra(head, self.food, snake_body)
        self.decision_times.append(time.time() - start_time)
        self.path_lengths.append(1 if path else self.width + self.height)

        if path:
            next_pos = path[0]
            dx, dy = next_pos[0] - head[0], next_pos[1] - head[1]
            action = self.pathfinder.direction_map.get((dx, dy))
            if action is not None:
                return action

        safe_directions = self.pathfinder.get_safe_directions(head, self.snake)
        if safe_directions:
            return safe_directions[0]
        return 0

    async def step(self):
        if self.game_over:
            return

        action = self.get_action()
        self.direction = self._action_to_direction(action)
        head = self.snake[0]
        new_head = (head[0] + self.direction.value[0], head[1] + self.direction.value[1])

        if new_head[0] < 0 or new_head[0] >= self.width or new_head[1] < 0 or new_head[1] >= self.height:
            self.game_over = True
            self.death_cause = DeathCause.WALL
            self.survival_time = (asyncio.get_event_loop().time() - self.start_time)
            return
        if new_head in self.snake:
            self.game_over = True
            self.death_cause = DeathCause.SELF
            self.survival_time = (asyncio.get_event_loop().time() - self.start_time)
            return
        if self.moves >= TIMEOUT_MOVES:
            self.game_over = True
            self.death_cause = DeathCause.TIMEOUT
            self.survival_time = (asyncio.get_event_loop().time() - self.start_time)
            return

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self._generate_food()
        else:
            self.snake.pop()

        self.moves += 1

    def _action_to_direction(self, action):
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        new_direction = directions[action % 4]
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        if new_direction == opposite[self.direction]:
            return self.direction
        return new_direction

    def get_metrics(self):
        path_efficiency = self.score / self.moves if self.moves > 0 else 0
        avg_decision_time = sum(self.decision_times) / len(self.decision_times) if self.decision_times else 0
        avg_path_efficiency = sum([actual / optimal if optimal > 0 else 0 for actual, optimal in zip(self.path_lengths, self.optimal_path_lengths)]) / len(self.path_lengths) if self.path_lengths else 0
        return {
            "score": self.score,
            "survival_time": self.survival_time,
            "moves": self.moves,
            "path_efficiency": path_efficiency,
            "decision_time": avg_decision_time,
            "path_length_efficiency": avg_path_efficiency,
            "death_cause": self.death_cause.value if self.death_cause else None,
        }

    def get_state(self):
        return {
            "snake": self.snake,
            "food": self.food,
            "score": self.score,
            "game_over": self.game_over,
            "direction": self.direction.name,
            "moves": self.moves,
            "survival_time": self.survival_time,
            "death_cause": self.death_cause.value if self.death_cause else None
        }

class DijkstraSnakeAI:
    def __init__(self):
        self.pathfinder = PathfindingAlgorithm(GRID_WIDTH, GRID_HEIGHT)
        self.direction_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}

    def get_action(self, state):
        """Get the next action based on the current state"""
        snake = state["snake"]
        food = state["food"]
        head = snake[0]
        
        # Find path to food using Dijkstra's algorithm
        path = self.pathfinder.dijkstra(head, food, set(snake[1:]))
        
        if path:
            # Get the next position in the path
            next_pos = path[0]
            # Calculate the direction to move
            dx = next_pos[0] - head[0]
            dy = next_pos[1] - head[1]
            # Get the action index
            return self.direction_map.get((dx, dy), 0)
        
        # If no path found, try to find a safe direction
        safe_directions = self.pathfinder.get_safe_directions(head, snake)
        if safe_directions:
            return safe_directions[0]
        
        # If no safe direction, default to current direction
        return 0

# async def run_simulation():
#     tracemalloc.start()
#     game = SnakeGame()
#     game.start_time = asyncio.get_event_loop().time()

#     while not game.game_over:
#         start_snapshot = tracemalloc.take_snapshot()
#         await game.step()
#         end_snapshot = tracemalloc.take_snapshot()

#     stats = end_snapshot.compare_to(start_snapshot, 'lineno')
#     total_memory = sum(stat.size_diff for stat in stats)
#     game.peak_memory = total_memory
#     tracemalloc.stop()

#     return game.get_metrics()

# if platform.system() == "Emscripten":
#     asyncio.ensure_future(run_simulation())
# else:
#     asyncio.run(run_simulation())