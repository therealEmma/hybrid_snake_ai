import asyncio
import platform
import random
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

class RuleBasedSnakeAI:
    def __init__(self):
        self.direction_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT

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

    def get_action(self, state):
        """Get the next action based on the current state using simple rules"""
        snake = state["snake"]
        food = state["food"]
        head = snake[0]
        
        # Get all safe directions
        safe_directions = self.get_safe_directions(head, snake)
        if not safe_directions:
            return 0  # No safe directions, default to current direction
        
        # Calculate distances to food for each safe direction
        distances = []
        for direction in safe_directions:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction]
            next_pos = (head[0] + dx, head[1] + dy)
            distance = abs(next_pos[0] - food[0]) + abs(next_pos[1] - food[1])
            distances.append((distance, direction))
        
        # Choose the direction that gets us closest to the food
        if distances:
            return min(distances, key=lambda x: x[0])[1]
        
        # If no safe directions lead closer to food, choose any safe direction
        return safe_directions[0]

class SnakeGame:
    def __init__(self):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.start_time = 0
        self.decision_times = []
        self.path_lengths = []
        self.optimal_path_lengths = []
        self.death_cause = None
        self.pathfinder = RuleBasedSnakeAI()
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

    def _rule_based_action(self):
        start_time = time.time()
        head = self.snake[0]
        food = self.food
        safe_directions = self.pathfinder.get_safe_directions(head, self.snake)
        if not safe_directions:
            self.decision_times.append(time.time() - start_time)
            return 0

        best_distance = float('inf')
        best_action = safe_directions[0]
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for action in safe_directions:
            dx, dy = directions[action]
            next_pos = (head[0] + dx, head[1] + dy)
            distance = abs(next_pos[0] - food[0]) + abs(next_pos[1] - food[1])
            if distance < best_distance:
                best_distance = distance
                best_action = action

        self.decision_times.append(time.time() - start_time)
        return best_action

    def get_action(self):
        head = self.snake[0]
        snake_body = set(self.snake[1:])
        self.optimal_path_lengths.append(self.width + self.height)
        action = self._rule_based_action()
        self.path_lengths.append(1)
        return action

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

    def get_state(self):
        return {
            "snake": self.snake,
            "food": self.food,
            "score": self.score,
            "game_over": self.game_over,
            "direction": self.direction.name,
            "moves": self.moves,
            "survival_time": self.survival_time,
        }
    

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