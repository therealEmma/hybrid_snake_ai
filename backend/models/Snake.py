import asyncio
import platform
import random
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from typing import List, Tuple, Optional
from enum import Enum
from collections import defaultdict
import uuid
from backend.models.hybrid import SnakeGame as HybridSnakeGame, CGPGenome as HybridCGPGenome, HybridSnakeAI, CGPEvolution as HybridCGPEvolution
from CGP import SnakeGame as CGPSnakeGame, SnakeAI as CGPSnakeAI, CGPEvolution as CGPEvolution, CGPGenome
from CGP import load_genome
import time  # For decision speed
import tracemalloc  # For memory usage

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE
NUM_RUNS = 100  # Number of runs per algorithm for comparison
SCORE_THRESHOLDS = [5, 10, 15, 20]  # For survival rate analysis
TIMEOUT_MOVES = 5000 # Maximum moves before timeout

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class DeathCause(Enum):
    WALL = "Wall Collision"
    SELF = "Self Collision"
    TIMEOUT = "Timeout"

class PathfindingAlgorithm:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.direction_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}  # Precompute
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        return [(x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)] 
                if 0 <= x + dx < self.width and 0 <= y + dy < self.height]
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int], 
               obstacles: set) -> List[Tuple[int, int]]:
        if start == goal or goal in obstacles:
            return []
        
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
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
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return []
    
    def greedy_best_first(self, start: Tuple[int, int], goal: Tuple[int, int], 
                         obstacles: set) -> List[Tuple[int, int]]:
        if start == goal or goal in obstacles:
            return []
        
        open_set = []
        heappush(open_set, (self.heuristic(start, goal), start))
        came_from = {}
        visited = set()
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in obstacles or neighbor in visited:
                    continue
                came_from[neighbor] = current
                heappush(open_set, (self.heuristic(neighbor, goal), neighbor))
        
        return []
    
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
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        for i, (dx, dy) in enumerate(directions):
            next_pos = (head[0] + dx, head[1] + dy)
            if (0 <= next_pos[0] < self.width and 
                0 <= next_pos[1] < self.height and 
                next_pos not in obstacles):
                safe_directions.append(i)
        return safe_directions

class SnakeGame:
    def __init__(self, algorithm: str, hybrid_genome: Optional[HybridCGPGenome] = None, cgp_genome: Optional[CGPGenome] = None):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.algorithm = algorithm
        self.pathfinder = PathfindingAlgorithm(self.width, self.height)
        self.hybrid_ai = None
        self.cgp_ai = None
        if algorithm == "hybrid" and hybrid_genome:
            self.hybrid_ai = HybridSnakeAI(hybrid_genome, self.width, self.height)
        elif algorithm == "cgp" and cgp_genome:
            self.cgp_ai = CGPSnakeAI(cgp_genome)
        self.start_time = 0
        self.decision_times = []
        self.path_lengths = []
        self.optimal_path_lengths = []
        self.death_cause = None
        self.peak_memory = 0
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
        self.peak_memory = 0
    
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
            return 0  # Default to UP if no safe moves
        
        # Calculate Manhattan distance to food for each safe direction
        best_distance = float('inf')
        best_action = safe_directions[0]
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        
        for action in safe_directions:
            dx, dy = directions[action]
            next_pos = (head[0] + dx, head[1] + dy)
            distance = abs(next_pos[0] - food[0]) + abs(next_pos[1] - food[1])
            if distance < best_distance:
                best_distance = distance
                best_action = action
        
        self.decision_times.append(time.time() - start_time)
        return best_action
    
    def _get_cgp_state(self):
        start_time = time.time()
        head = self.snake[0]
        food_dx = self.food[0] - head[0]
        food_dy = self.food[1] - head[1]
        food_distance = np.sqrt(food_dx**2 + food_dy**2)
        
        # Calculate angle to food relative to snake direction
        dir_vector = self.direction.value
        food_vector = (food_dx, food_dy)
        dot_product = dir_vector[0] * food_vector[0] + dir_vector[1] * food_vector[1]
        dir_magnitude = np.sqrt(dir_vector[0]**2 + dir_vector[1]**2)
        food_magnitude = food_distance if food_distance > 0 else 1
        cos_angle = dot_product / (dir_magnitude * food_magnitude) if food_distance > 0 else 0
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        obstacles = []
        for dx, dy in directions:
            next_pos = (head[0] + dx, head[1] + dy)
            is_obstacle = (next_pos[0] < 0 or next_pos[0] >= self.width or
                          next_pos[1] < 0 or next_pos[1] >= self.height or
                          next_pos in self.snake)
            obstacles.append(1.0 if is_obstacle else 0.0)
        
        food_directions = [
            1.0 if food_dy < 0 else 0.0,  # Food is up
            1.0 if food_dy > 0 else 0.0,  # Food is down
            1.0 if food_dx < 0 else 0.0,  # Food is left
            1.0 if food_dx > 0 else 0.0   # Food is right
        ]
        
        body_distances = []
        for dx, dy in directions:
            distance = self.width + self.height
            x, y = head[0], head[1]
            steps = 1
            while 0 <= x + dx * steps < self.width and 0 <= y + dy * steps < self.height:
                pos = (x + dx * steps, y + dy * steps)
                if pos in self.snake:
                    distance = steps
                    break
                steps += 1
            body_distances.append(distance)
        
        body_density = 0
        x, y = head
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) in self.snake[1:]:
                    body_density += 1
        
        snake_length = len(self.snake)
        direction_encoding = [0.0] * 4
        direction_map = {Direction.UP: 0, Direction.DOWN: 1, Direction.LEFT: 2, Direction.RIGHT: 3}
        direction_encoding[direction_map[self.direction]] = 1.0
        
        state = (
            [food_dx, food_dy, food_distance, cos_angle] +
            food_directions +
            obstacles +
            body_distances +
            [body_density, snake_length] +
            direction_encoding
        )
        
        normalized_state = []
        for i, val in enumerate(state):
            if i < 4:  # Food dx, dy, distance, angle
                if i == 2:
                    normalized_state.append(val / np.sqrt(self.width**2 + self.height**2))
                elif i == 3:
                    normalized_state.append((val + 1) / 2)  # Normalize [-1, 1] to [0, 1]
                else:
                    normalized_state.append(val / max(self.width, self.height))
            elif i < 8:  # Food directions
                normalized_state.append(val)
            elif i < 12:  # Obstacles
                normalized_state.append(val)
            elif i < 16:  # Body distances
                normalized_state.append(val / (self.width + self.height))
            elif i == 16:  # Body density
                normalized_state.append(val / 24)  # Max 24 segments in 5x5 grid (excluding head)
            elif i == 17:  # Snake length
                normalized_state.append(val / (self.width * self.height))
            else:  # Direction encoding
                normalized_state.append(val)
        
        self.decision_times.append(time.time() - start_time)
        return normalized_state
    
    def _get_simple_state(self):
        start_time = time.time()
        head = self.snake[0]
        food_dx = self.food[0] - head[0]
        food_dy = self.food[1] - head[1]
        food_distance = abs(food_dx) + abs(food_dy)
        
        food_directions = [
            1.0 if food_dy < 0 else 0.0,  # Food above
            1.0 if food_dy > 0 else 0.0,  # Food below
            1.0 if food_dx < 0 else 0.0,  # Food left
            1.0 if food_dx > 0 else 0.0   # Food right
        ]
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        obstacles = []
        for dx, dy in directions:
            next_pos = (head[0] + dx, head[1] + dy)
            is_obstacle = (next_pos[0] < 0 or next_pos[0] >= self.width or
                          next_pos[1] < 0 or next_pos[1] >= self.height or
                          next_pos in self.snake)
            obstacles.append(1.0 if is_obstacle else 0.0)
        
        direction_encoding = [0.0] * 4
        direction_map = {Direction.UP: 0, Direction.DOWN: 1, Direction.LEFT: 2, Direction.RIGHT: 3}
        direction_encoding[direction_map[self.direction]] = 1.0
        
        snake_body = set(self.snake[1:])
        path_to_food = self.pathfinder.a_star(head, self.food, snake_body)
        food_path_length = len(path_to_food) if path_to_food else self.width + self.height
        safe_directions = self.pathfinder.get_safe_directions(head, self.snake)
        safe_direction_count = len(safe_directions)
        
        state = (
            [food_dx, food_dy, food_distance] +
            food_directions +
            obstacles +
            [len(self.snake)] +
            direction_encoding +
            [food_path_length, safe_direction_count]
        )
        
        max_coord = max(self.width, self.height)
        max_distance = self.width + self.height
        normalized_state = []
        for i, val in enumerate(state):
            if i < 2:  # food_dx, food_dy
                normalized_state.append(val / max_coord)
            elif i == 2:  # food_distance
                normalized_state.append(val / max_distance)
            elif i < 7:  # food_directions
                normalized_state.append(val)
            elif i < 11:  # obstacles
                normalized_state.append(val)
            elif i == 11:  # snake length
                normalized_state.append(val / (self.width * self.height))
            elif i < 16:  # direction_encoding
                normalized_state.append(val)
            else:  # A* features
                normalized_state.append(val / max_distance)
        
        self.decision_times.append(time.time() - start_time)
        return normalized_state
    
    def get_action(self):
        start_time = time.time()
        head = self.snake[0]
        
        # Calculate optimal path length for path efficiency
        snake_body = set(self.snake[1:])
        optimal_path = self.pathfinder.a_star(head, self.food, snake_body)
        self.optimal_path_lengths.append(len(optimal_path) if optimal_path else self.width + self.height)
        
        if self.algorithm == "rule_based":
            action = self._rule_based_action()
            if self.moves % 10 == 0:
                self.decision_times.append(time.time() - start_time)
            self.path_lengths.append(1)
            return action
        
        if self.algorithm == "hybrid":
            if self.hybrid_ai:
                state = self._get_simple_state()
                action = self.hybrid_ai.get_action(state, self.snake, self.food)
                if self.moves % 10 == 0:
                    self.decision_times.append(time.time() - start_time)
                self.path_lengths.append(1)
                return action
            else:
                print("Warning: No hybrid genome provided, defaulting to random action")
                action = random.randint(0, 3)
                if self.moves % 10 == 0:
                    self.decision_times.append(time.time() - start_time)
                self.path_lengths.append(1)
                return action
        
        if self.algorithm == "cgp":
            if self.cgp_ai:
                state = self._get_cgp_state()
                action = self.cgp_ai.get_action(state)
                if self.moves % 10 == 0:
                    self.decision_times.append(time.time() - start_time)
                self.path_lengths.append(1)
                return action
            else:
                print("Warning: No CGP genome provided, defaulting to random action")
                action = random.randint(0, 3)
                if self.moves % 10 == 0:
                     self.decision_times.append(time.time() - start_time)
                self.path_lengths.append(1)
                return action
        
        path = []
        if self.algorithm == "a_star":
            path = self.pathfinder.a_star(head, self.food, snake_body)
        elif self.algorithm == "greedy":
            path = self.pathfinder.greedy_best_first(head, self.food, snake_body)
        elif self.algorithm == "dijkstra":
            path = self.pathfinder.dijkstra(head, self.food, snake_body)
        
        self.decision_times.append(time.time() - start_time)
        self.path_lengths.append(1 if path else self.width + self.height)  # Assume 1 step if path exists
        
        if path:
            next_pos = path[0]
            dx, dy = next_pos[0] - head[0], next_pos[1] - head[1]
            action = self.pathfinder.direction_map.get((dx, dy))
            if action is not None:
                return action
        
        safe_directions = self.pathfinder.get_safe_directions(head, self.snake)
        if safe_directions:
            return safe_directions[0]
        return 0  # Default to UP if no safe directions
    
    async def step(self):
        if self.game_over:
            return
        
        action = self.get_action()
        self.direction = self._action_to_direction(action)
        head = self.snake[0]
        new_head = (head[0] + self.direction.value[0], head[1] + self.direction.value[1])
        
        # Check for game over conditions
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
        avg_decision_time = np.mean(self.decision_times) if self.decision_times else 0
        avg_path_efficiency = np.mean([actual / optimal if optimal > 0 else 0 for actual, optimal in zip(self.path_lengths, self.optimal_path_lengths)]) if self.path_lengths and self.optimal_path_lengths else 0
        return {
            "score": self.score,
            "survival_time": self.survival_time,
            "moves": self.moves,
            "path_efficiency": path_efficiency,
            "decision_time": avg_decision_time,
            "path_length_efficiency": avg_path_efficiency,
            "death_cause": self.death_cause,
            "peak_memory": self.peak_memory / 1024  # Convert bytes to KB
        }

async def run_simulation(algorithm: str, hybrid_genome: Optional[HybridCGPGenome] = None, cgp_genome: Optional[CGPGenome] = None):
    tracemalloc.start()  # Start memory tracking
    game = SnakeGame(algorithm=algorithm, hybrid_genome=hybrid_genome, cgp_genome=cgp_genome)
    game.start_time = asyncio.get_event_loop().time()
    
    while not game.game_over:
        start_snapshot = tracemalloc.take_snapshot()
        await game.step()
        end_snapshot = tracemalloc.take_snapshot()
    
    # Final memory snapshot
    stats = end_snapshot.compare_to(start_snapshot, 'lineno')
    total_memory = sum(stat.size_diff for stat in stats)
    game.peak_memory = total_memory  
    tracemalloc.stop()  # Stop memory tracking
    
    return game.get_metrics()

def plot_comparison(metrics: dict):
    algorithms = list(metrics.keys())
    scores = [np.mean([run["score"] for run in metrics[algo]]) for algo in algorithms]
    max_scores = [max([run["score"] for run in metrics[algo]]) for algo in algorithms]
    survival_times = [np.mean([run["survival_time"] for run in metrics[algo]]) for algo in algorithms]
    path_efficiencies = [np.mean([run["path_efficiency"] for run in metrics[algo]]) for algo in algorithms]
    decision_times = [np.mean([run["decision_time"] for run in metrics[algo]]) for algo in algorithms]
    path_length_efficiencies = [np.mean([run["path_length_efficiency"] for run in metrics[algo]]) for algo in algorithms]
    peak_memories = [np.mean([run["peak_memory"] for run in metrics[algo]]) for algo in algorithms]
    
    # Bar plot for scores
    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, scores, color='skyblue')
    plt.title('Average Score per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Score')
    plt.savefig('scores_comparison.png')
    plt.close()
    
    # Bar plot for max scores
    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, max_scores, color='lightblue')
    plt.title('Maximum Score per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Max Score')
    plt.savefig('max_scores_comparison.png')
    plt.close()
    
    # Bar plot for survival times
    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, survival_times, color='lightgreen')
    plt.title('Average Survival Time per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Survival Time (s)')
    plt.savefig('survival_time_comparison.png')
    plt.close()
    
    # Bar plot for move efficiencies
    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, path_efficiencies, color='salmon')
    plt.title('Average Move Efficiency per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Move Efficiency (Score/Moves)')
    plt.savefig('path_efficiency_comparison.png')
    plt.close()
    
    # Bar plot for decision times
    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, decision_times, color='purple')
    plt.title('Average Computational Time per Move')
    plt.xlabel('Algorithm')
    plt.ylabel('Computational Time (s)')
    plt.savefig('decision_time_comparison.png')
    plt.close()
    
    # Bar plot for path length efficiencies
    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, path_length_efficiencies, color='orange')
    plt.title('Average Path Length Efficiency per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Path Length Efficiency (Actual/Optimal)')
    plt.savefig('path_length_efficiency_comparison.png')
    plt.close()
    
    # Bar plot for peak memory usage
    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, peak_memories, color='teal')
    plt.title('Average Peak Memory Usage per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Peak Memory (KB)')
    plt.savefig('peak_memory_comparison.png')
    plt.close()

def plot_boxplots(metrics: dict):
    algorithms = list(metrics.keys())
    scores = [[run["score"] for run in metrics[algo]] for algo in algorithms]
    survival_times = [[run["survival_time"] for run in metrics[algo]] for algo in algorithms]
    path_efficiencies = [[run["path_efficiency"] for run in metrics[algo]] for algo in algorithms]
    decision_times = [[run["decision_time"] for run in metrics[algo]] for algo in algorithms]
    path_length_efficiencies = [[run["path_length_efficiency"] for run in metrics[algo]] for algo in algorithms]
    peak_memories = [[run["peak_memory"] for run in metrics[algo]] for algo in algorithms]
    
    # Box plot for scores
    plt.figure(figsize=(12, 6))
    plt.boxplot(scores, labels=algorithms, patch_artist=True, 
                boxprops=dict(facecolor='skyblue', color='black'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    plt.title('Score Distribution per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Score')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('scores_boxplot.png')
    plt.close()
    
    # Box plot for survival times with adjusted scale
    plt.figure(figsize=(14, 8))  # Increased figure size
    plt.boxplot(survival_times, labels=algorithms, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen', color='black'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    plt.title('Survival Time Distribution per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Survival Time (s)')
    plt.ylim(0, max([max(times) for times in survival_times]) * 1.2)  # Extend y-axis by 20%
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('survival_time_boxplot.png')
    plt.close()
    
    # Box plot for move efficiencies
    plt.figure(figsize=(12, 6))
    plt.boxplot(path_efficiencies, labels=algorithms, patch_artist=True, 
                boxprops=dict(facecolor='salmon', color='black'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    plt.title('Move Efficiency Distribution per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Move Efficiency (Score/Moves)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('path_efficiency_boxplot.png')
    plt.close()
    
    # Box plot for decision times
    plt.figure(figsize=(12, 6))
    plt.boxplot(decision_times, labels=algorithms, patch_artist=True, 
                boxprops=dict(facecolor='purple', color='black'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    plt.title('Computational Time per Move Distribution')
    plt.xlabel('Algorithm')
    plt.ylabel('Computational Time (s)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('decision_time_boxplot.png')
    plt.close()
    
    # Box plot for path length efficiencies
    plt.figure(figsize=(12, 6))
    plt.boxplot(path_length_efficiencies, labels=algorithms, patch_artist=True, 
                boxprops=dict(facecolor='orange', color='black'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    plt.title('Path Length Efficiency Distribution per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Path Length Efficiency (Actual/Optimal)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('path_length_efficiency_boxplot.png')
    plt.close()
    
    # Box plot for peak memory usage
    plt.figure(figsize=(12, 6))
    plt.boxplot(peak_memories, labels=algorithms, patch_artist=True, 
                boxprops=dict(facecolor='teal', color='black'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    plt.title('Peak Memory Usage Distribution per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Peak Memory (KB)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('peak_memory_boxplot.png')
    plt.close()

def plot_score_distribution(metrics: dict):
    algorithms = list(metrics.keys())
    for algo in algorithms:
        scores = [run["score"] for run in metrics[algo]]
        bins = np.arange(0, max(scores) + 5, 5)  # Bins of size 5
        hist, _ = np.histogram(scores, bins=bins)
        labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
        
        plt.figure(figsize=(8, 8))
        plt.pie(hist, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f'Score Distribution for {algo.upper()}')
        plt.savefig(f'score_distribution_{algo}.png')
        plt.close()

def plot_death_analysis(metrics: dict):
    algorithms = list(metrics.keys())
    for algo in algorithms:
        death_counts = defaultdict(int)
        for run in metrics[algo]:
            cause = run["death_cause"].value if run["death_cause"] else "Unknown"
            death_counts[cause] += 1
        
        labels = list(death_counts.keys())
        values = list(death_counts.values())
        
        plt.figure(figsize=(8, 8))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f'Death Cause Distribution for {algo.upper()}')
        plt.savefig(f'death_analysis_{algo}.png')
        plt.close()

async def main():
    algorithms = ["a_star", "greedy", "dijkstra", "rule_based", "hybrid", "cgp"]
    metrics = defaultdict(list)
    
    # Initialize real-time results file
    with open('simulation_results.txt', 'w') as f:
        f.write("Simulation Results\n")
        f.write("=" * 80 + "\n\n")
    
    # Load hybrid genome
    hybrid_genome = None
    try:
        evolution = HybridCGPEvolution()
        hybrid_genome = evolution.load_genome("best_genome.json")
        print("Loaded hybrid genome from 'best_genome.json'")
        with open('simulation_results.txt', 'a') as f:
            f.write("Loaded hybrid genome from 'best_genome.json'\n")
    except FileNotFoundError:
        print("Warning: 'best_genome.json' not found. Hybrid algorithm will use random actions.")
        with open('simulation_results.txt', 'a') as f:
            f.write("Warning: 'best_genome.json' not found. Hybrid algorithm will use random actions.\n")
    except Exception as e:
        print(f"Warning: Failed to load hybrid genome: {e}. Hybrid algorithm will use random actions.")
        with open('simulation_results.txt', 'a') as f:
            f.write(f"Warning: Failed to load hybrid genome: {e}. Hybrid algorithm will use random actions.\n")
    
    # Load CGP genome
    cgp_genome = None
    try:
        cgp_genome = load_genome("training_data/best_snake_ai.json")
        print("Loaded CGP genome from 'training_data/best_snake_ai.json'")
        with open('simulation_results.txt', 'a') as f:
            f.write("Loaded CGP genome from 'training_data/best_snake_ai.json'\n")
    except FileNotFoundError:
        print("Warning: 'training_data/best_snake_ai.json' not found. CGP algorithm will use random actions.")
        with open('simulation_results.txt', 'a') as f:
            f.write("Warning: 'training_data/best_snake_ai.json' not found. CGP algorithm will use random actions.\n")
    except Exception as e:
        print(f"Warning: Failed to load CGP genome: {e}. CGP algorithm will use random actions.")
        with open('simulation_results.txt', 'a') as f:
            f.write(f"Warning: Failed to load CGP genome: {e}. CGP algorithm will use random actions.\n")
    
    for algo in algorithms:
        print(f"Running {algo.upper()}...")
        with open('simulation_results.txt', 'a') as f:
            f.write(f"Running {algo.upper()}...\n")
        for i in range(NUM_RUNS):
            result = await run_simulation(algo, hybrid_genome, cgp_genome)
            metrics[algo].append(result)
            result_str = (f"Run {i+1}/{NUM_RUNS} completed: Score={result['score']}, "
                          f"Survival Time={result['survival_time']:.2f}s, "
                          f"Move Efficiency={result['path_efficiency']:.4f}, "
                          f"Path Length Efficiency={result['path_length_efficiency']:.4f}, "
                          f"Computational Time per Move={result['decision_time']:.6f}s, "
                          f"Peak Memory={result['peak_memory']:.2f}KB, "
                          f"Death Cause={result['death_cause'].value if result['death_cause'] else 'Unknown'}")
            print(result_str)
            with open('simulation_results.txt', 'a') as f:
                f.write(result_str + "\n")
    
    # Plot bar charts
    plot_comparison(metrics)
    print("Performance plots saved as 'scores_comparison.png', 'max_scores_comparison.png', "
          "'survival_time_comparison.png', 'path_efficiency_comparison.png', "
          "'decision_time_comparison.png', 'path_length_efficiency_comparison.png', "
          "and 'peak_memory_comparison.png'")
    with open('simulation_results.txt', 'a') as f:
        f.write("Performance plots saved as 'scores_comparison.png', 'max_scores_comparison.png', "
                "'survival_time_comparison.png', 'path_efficiency_comparison.png', "
                "'decision_time_comparison.png', 'path_length_efficiency_comparison.png', "
                "and 'peak_memory_comparison.png'\n")
    
    # Plot boxplots
    plot_boxplots(metrics)
    print("Box plots saved as 'scores_boxplot.png', 'survival_time_boxplot.png', "
          "'path_efficiency_boxplot.png', 'decision_time_boxplot.png', "
          "'path_length_efficiency_boxplot.png', and 'peak_memory_boxplot.png'")
    with open('simulation_results.txt', 'a') as f:
        f.write("Box plots saved as 'scores_boxplot.png', 'survival_time_boxplot.png', "
                "'path_efficiency_boxplot.png', 'decision_time_boxplot.png', "
                "'path_length_efficiency_boxplot.png', and 'peak_memory_boxplot.png'\n")
    
    # Plot score distribution (pie charts)
    plot_score_distribution(metrics)
    print(f"Score distribution pie charts saved as 'score_distribution_<algorithm>.png' for each algorithm")
    with open('simulation_results.txt', 'a') as f:
        f.write(f"Score distribution pie charts saved as 'score_distribution_<algorithm>.png' for each algorithm\n")
    
    # Plot death analysis (pie charts)
    plot_death_analysis(metrics)
    print(f"Death analysis pie charts saved as 'death_analysis_<algorithm>.png' for each algorithm")
    with open('simulation_results.txt', 'a') as f:
        f.write(f"Death analysis pie charts saved as 'death_analysis_<algorithm>.png' for each algorithm\n")
    
    # Calculate survival rates
    survival_rate_results = []
    algorithms = list(metrics.keys())
    for algo in algorithms:
        scores = [run["score"] for run in metrics[algo]]
        result = f"Survival Rate for {algo.upper()}:\n"
        for threshold in SCORE_THRESHOLDS:
            rate = sum(1 for score in scores if score >= threshold) / len(scores) * 100
            result += f"  Score >= {threshold}: {rate:.2f}%\n"
        result += "\n"
        survival_rate_results.append(result)
    
    # Output and save survival rate analysis
    print("\nSurvival Rate Analysis:")
    with open('simulation_results.txt', 'a') as f:
        f.write("\nSurvival Rate Analysis:\n")
        f.write("=" * 80 + "\n\n")
    for result in survival_rate_results:
        print(result)
        with open('simulation_results.txt', 'a') as f:
            f.write(result)

# if platform.system() == "Emscripten":
#     asyncio.ensure_future(main())
# else:
#     asyncio.run(main())