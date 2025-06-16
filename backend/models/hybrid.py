import random
import numpy as np
import math
from enum import Enum
from typing import List, Tuple, Optional, Callable
import json
import time
import matplotlib.pyplot as plt
import os
from collections import deque
import heapq
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

class AStarPathfinder:
    """Optimized A* pathfinding algorithm for Snake game"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        x, y = pos
        return [(x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)] 
                if 0 <= x + dx < self.width and 0 <= y + dy < self.height]
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                 obstacles: set) -> List[Tuple[int, int]]:
        """Find shortest path from start to goal avoiding obstacles"""
        if start == goal or goal in obstacles:
            return []
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
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
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []
    
    def get_safe_directions(self, head: Tuple[int, int], snake: List[Tuple[int, int]]) -> List[int]:
        """Get directions that don't lead to immediate collision"""
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
    
class CGPGenome:
    """Optimized Cartesian Genetic Programming genome"""
    
    FUNCTIONS = {
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y,
        'div': lambda x, y: x / y if abs(y) > 1e-6 else 1.0,
        'max': lambda x, y: max(x, y),
        'min': lambda x, y: min(x, y),
        'gt': lambda x, y: 1.0 if x > y else 0.0,
        'lt': lambda x, y: 1.0 if x < y else 0.0,
        'and': lambda x, y: 1.0 if x > 0.5 and y > 0.5 else 0.0,
        'or': lambda x, y: 1.0 if x > 0.5 or y > 0.5 else 0.0,
        'sigmoid': lambda x, y: 1.0 / (1.0 + math.exp(-max(min(x, 10), -10))),
        'clamp': lambda x, y: max(0.0, min(1.0, x))
    }
    
    def __init__(self, n_inputs: int, n_outputs: int, n_nodes: int):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes
        self.avg_score = 0.0
        self.avg_moves = 0.0
        self.avg_survival_time = 0.0
        self.avg_safety_overrides = 0.0
        self.function_names = list(self.FUNCTIONS.keys())
        self.nodes = []

        # Input nodes
        for i in range(n_inputs):
            self.nodes.append(CGPNode('input', [i]))

        # Function nodes
        for i in range(n_nodes):
            max_input_idx = self.n_inputs + i - 1
            if max_input_idx < self.n_inputs:
                max_input_idx = self.n_inputs - 1
            max_input_idx = min(max_input_idx, len(self.nodes) - 1)  # Cap

            input1 = random.randint(0, max_input_idx)
            input2 = random.randint(0, max_input_idx)
            function_name = random.choice(self.function_names)
            function = self.FUNCTIONS[function_name]
            self.nodes.append(CGPNode('function', [input1, input2], function))

        # Output nodes
        max_node_idx = len(self.nodes) - 1
        for i in range(n_outputs):
            output_input = random.randint(0, max_node_idx)
            self.nodes.append(CGPNode('output', [output_input]))

        self.update_active_nodes()
        self.fitness = 0.0

    def update_active_nodes(self):
        """Mark nodes that are actually used in the computation"""
        for node in self.nodes:
            node.active = False

        output_start = self.n_inputs + self.n_nodes
        active_queue = list(range(output_start, len(self.nodes)))

        while active_queue:
            node_idx = active_queue.pop(0)
            if not (0 <= node_idx < len(self.nodes)):
                continue
            if self.nodes[node_idx].active:
                continue
            self.nodes[node_idx].active = True
            for input_idx in self.nodes[node_idx].inputs:
                if 0 <= input_idx < len(self.nodes):
                    active_queue.append(input_idx)

    def evaluate(self, inputs: List[float]) -> List[float]:
        """Evaluate the CGP program with given inputs"""
        node_values = [0.0] * len(self.nodes)
        
        for i, node in enumerate(self.nodes):
            if node.active:
                node_values[i] = node.evaluate(inputs, node_values)
                node_values[i] = max(min(node_values[i], 10.0), -10.0)
        
        output_start = self.n_inputs + self.n_nodes
        return [node_values[i] for i in range(output_start, len(self.nodes))]
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate the genome"""
        function_start = self.n_inputs
        function_end = self.n_inputs + self.n_nodes
        
        for i in range(function_start, function_end):
            node = self.nodes[i]
            if random.random() < mutation_rate:
                max_input_idx = self.n_inputs + (i - self.n_inputs) - 1
                if max_input_idx < self.n_inputs:
                    max_input_idx = self.n_inputs - 1
                node.inputs[0] = random.randint(0, max_input_idx)
                node.inputs[1] = random.randint(0, max_input_idx)
            
            if random.random() < mutation_rate:
                function_name = random.choice(self.function_names)
                node.function = self.FUNCTIONS[function_name]
        
        output_start = self.n_inputs + self.n_nodes
        for i in range(output_start, len(self.nodes)):
            if random.random() < mutation_rate:
                max_node_idx = self.n_inputs + self.n_nodes - 1
                self.nodes[i].inputs[0] = random.randint(0, max_node_idx)
        
        self.update_active_nodes()
    
    def copy(self):
        """Create a copy of this genome"""
        new_genome = CGPGenome(self.n_inputs, self.n_outputs, self.n_nodes)
        for i, node in enumerate(self.nodes):
            new_genome.nodes[i].inputs = node.inputs.copy()
            new_genome.nodes[i].function = node.function
            new_genome.nodes[i].active = node.active

        new_genome.fitness = self.fitness
        new_genome.avg_score = self.avg_score
        new_genome.avg_moves = self.avg_moves
        new_genome.avg_survival_time = self.avg_survival_time
        return new_genome
    
class SnakeGame:
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT, genome: CGPGenome = None):
        self.width = width
        self.height = height
        self.genome = genome
        self.pathfinder = AStarPathfinder(width, height)
        self.ai = HybridSnakeAI(genome, width, height) if genome else None
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
        self.limit_steps = 0
        self.max_moves = self.width * self.height * 2
        self.game_over = False
        self.survival_time = 0.0
        self.start_time = time.time()
        self.death_cause = None
        self.path_lengths = []
        self.decision_times = []
        self.optimal_path_lengths = []
        return self.get_internal_state()

    def _generate_food(self):
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food

    async def step(self):
        """Async version for gameplay rendering"""
        if self.game_over:
            return self.get_state(), 0, True

        if self.ai:
            start_time = time.time()
            state = self.get_internal_state()
            path_to_food = self.pathfinder.find_path(self.snake[0], self.food, set(self.snake[1:]))
            action = self.ai.get_action(state, self.snake, self.food)
            self.decision_times.append(time.time() - start_time)
            self.path_lengths.append(1 if path_to_food else self.width + self.height)
        else:
            raise ValueError("No AI provided and no action specified")

        return self.step_with_action(action)

    def step_with_action(self, action: int):
        """Sync version for training"""
        if self.game_over:
            return self.get_internal_state(), 0, True

        self.direction = self._action_to_direction(action)
        head = self.snake[0]
        new_head = (head[0] + self.direction.value[0], head[1] + self.direction.value[1])

        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height):
            self.game_over = True
            self.death_cause = DeathCause.WALL
            self.survival_time = time.time() - self.start_time
            # return self.get_internal_state(), -100, True
            return self.get_state(), -100, True
        
        if new_head in self.snake:
            self.game_over = True
            self.death_cause = DeathCause.SELF
            self.survival_time = time.time() - self.start_time
            return self.get_state(), -100, True
        
        if self.moves >= TIMEOUT_MOVES:
            self.game_over = True
            self.death_cause = DeathCause.TIMEOUT
            self.survival_time = time.time() - self.start_time
            return self.get_state(), -50, True

        self.snake.insert(0, new_head)
        reward = -0.1

        old_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        if new_head == self.food:
            self.score += 1
            reward = 100
            self.food = self._generate_food()
        else:
            self.snake.pop()
            reward += 10 / (new_distance + 1) if new_distance < old_distance else -1.0

        path_to_food = self.pathfinder.find_path(new_head, self.food, set(self.snake[1:]))
        if path_to_food:
            reward += 20.0 / (len(path_to_food) + 1)

        self.moves += 1
        if self.limit_steps and self.moves >= self.max_moves:
            self.game_over = True
            self.survival_time = time.time() - self.start_time
            reward = -50

        return self.get_internal_state(), reward, self.game_over


    
    def get_astar_features(self, head: Tuple[int, int]) -> List[float]:
        """Get A* informed features for the state"""
        snake_body = set(self.snake[1:])
        
        path_to_food = self.pathfinder.find_path(head, self.food, snake_body)
        food_path_length = len(path_to_food) if path_to_food else self.width + self.height
        
        safe_directions = self.pathfinder.get_safe_directions(head, self.snake)
        safe_direction_count = len(safe_directions)
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        direction_path_lengths = []
        
        for dx, dy in directions:
            next_pos = (head[0] + dx, head[1] + dy)
            if (0 <= next_pos[0] < self.width and 
                0 <= next_pos[1] < self.height and 
                next_pos not in snake_body):
                path = self.pathfinder.find_path(next_pos, self.food, snake_body)
                path_len = len(path) if path else self.width + self.height
                direction_path_lengths.append(path_len)
            else:
                direction_path_lengths.append(self.width + self.height)
        
        return [
            food_path_length,
            safe_direction_count,
        ] + direction_path_lengths
    
    def get_internal_state(self):
        """Get current game state as feature vector for AI"""
        head = self.snake[0]
        food_dx = self.food[0] - head[0]
        food_dy = self.food[1] - head[1]
        food_distance = abs(food_dx) + abs(food_dy)
        
        food_directions = [
            1.0 if food_dy < 0 else 0.0,
            1.0 if food_dy > 0 else 0.0,
            1.0 if food_dx < 0 else 0.0,
            1.0 if food_dx > 0 else 0.0
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
        
        astar_features = self.get_astar_features(head)
        state = (
            [food_dx, food_dy, food_distance] +
            food_directions +
            obstacles +
            [len(self.snake)] +
            direction_encoding +
            astar_features
        )
        
        max_coord = max(self.width, self.height)
        max_distance = self.width + self.height
        
        normalized_state = []
        for i, val in enumerate(state):
            if i < 2:
                normalized_state.append(val / max_coord)
            elif i == 2:
                normalized_state.append(val / max_distance)
            elif i < 7:
                normalized_state.append(val)
            elif i == 7:
                normalized_state.append(val / (self.width * self.height))
            elif i < 12:
                normalized_state.append(val)
            else:
                normalized_state.append(val / (max_distance * 0.5))
        
        return normalized_state
    
    def _action_to_direction(self, action):
        """Convert action (0-3) to direction, preventing reverse moves"""
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
            "death_cause":self.death_cause.value if self.death_cause else None
        }
    
    
    def get_metrics(self):
        path_efficiency = self.score / self.moves if self.moves > 0 else 0
        avg_decision_time = sum(self.decision_times) / len(self.decision_times) if self.decision_times else 0
        return {
            "score": self.score,
            "survival_time": self.survival_time,
            "moves": self.moves,
            "path_efficiency": path_efficiency,
            "decision_time": avg_decision_time,
            "death_cause": self.death_cause.value if self.death_cause else None,
        }


class CGPNode:
    """A node in the Cartesian Genetic Programming graph"""

    def __init__(self, node_type: str, inputs: List[int], function: Optional[Callable] = None):
        self.type = node_type
        self.inputs = inputs
        self.function = function
        self.active = False

    def evaluate(self, input_values: List[float], node_values: List[float]) -> float:
        try:
            if self.type == 'input':
                if 0 <= self.inputs[0] < len(input_values):
                    return input_values[self.inputs[0]]
                else:
                    # print(f"⚠️ Input index {self.inputs[0]} out of bounds (len={len(input_values)})")
                    return 0.0

            elif self.type == 'function':
                func_inputs = []
                for idx in self.inputs:
                    if 0 <= idx < len(node_values):
                        func_inputs.append(node_values[idx])
                    else:
                        # print(f"⚠️ Function input index {idx} out of bounds")
                        func_inputs.append(0.0)
                return self.function(*func_inputs)

            elif self.type == 'output':
                idx = self.inputs[0]
                if 0 <= idx < len(node_values):
                    return node_values[idx]
                else:
                    return 0.0
        except Exception as e:
            return 0.0

        return 0.0


class HybridSnakeAI:
    """Optimized hybrid AI controller combining CGP with A* safety validation"""
    
    def __init__(self, genome: CGPGenome, game_width: int, game_height: int):
        self.genome = genome
        self.pathfinder = AStarPathfinder(game_width, game_height)
        self.safety_override_count = 0
    
    def get_action(self, state: List[float], snake: List[Tuple[int, int]], food: Tuple[int, int]) -> int:
        """Get action from hybrid AI with safety validation"""
        head = snake[0]

        outputs = self.genome.evaluate(state)
        cgp_preferences = [(outputs[i], i) for i in range(4)]
        cgp_preferences.sort(reverse=True)
        
        safe_directions = self.pathfinder.get_safe_directions(head, snake)
        
        if not safe_directions:
            return cgp_preferences[0][1]
        
        path_to_food = self.pathfinder.find_path(head, food, set(snake[1:]))
        preferred_action = None
        if path_to_food:
            next_pos = path_to_food[0]
            dx, dy = next_pos[0] - head[0], next_pos[1] - head[1]
            direction_map = {
                (0, -1): 0,
                (0, 1): 1,
                (-1, 0): 2,
                (1, 0): 3
            }
            preferred_action = direction_map.get((dx, dy))
        
        if preferred_action in safe_directions:
            for preference, action in cgp_preferences:
                if action == preferred_action:
                    return action
        
        for preference, action in cgp_preferences:
            if action in safe_directions:
                return action
        
        self.safety_override_count += 1
        return safe_directions[0]

class CGPEvolution:
    """Optimized evolution algorithm for hybrid CGP-A* genomes"""
    
    def __init__(self, population_size: int = 50, n_inputs: int = 18, n_outputs: int = 4,
                 n_nodes: int = 50, mutation_rate: float = 0.1, max_generations = 100, n_evaluations: int = 1,
                 log_file: Optional[str] = None):
        self.population_size = population_size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.n_evaluations = n_evaluations
        self.log_file = log_file
        self.population = [CGPGenome(n_inputs, n_outputs, n_nodes) for _ in range(population_size)]
        self.best_fitness = float('-inf')
        self.best_genome = None
        self.fitness_history = []
        self.fitness = 0.0
        self.avg_score = 0.0
        self.avg_moves = 0.0
        self.avg_survival_time = 0.0
        self.avg_safety_overrides = 0.0
        self.current_iteration = 0
    
    def log(self, message: str):
        """Log message to both console and file if specified"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(formatted_message + '\n')
                    f.flush()
            except Exception as e:
                print(f"Warning: Could not write to log file {self.log_file}: {e}")
    
    def evaluate_genome(self, genome: CGPGenome) -> float:
        """Evaluate a genome's fitness over multiple games"""
        total_fitness = 0.0
        total_scores = 0
        total_moves = 0
        total_survival_time = 0.0
        safety_overrides = 0
        
        for eval_run in range(self.n_evaluations):
            game = SnakeGame()
            ai = HybridSnakeAI(genome, game.width, game.height)
            
            state = game.reset()
            episode_reward = 0.0
            steps = 0
            max_steps = 2000
            
            while not game.game_over and steps < max_steps:
                action = ai.get_action(state, game.snake, game.food)
                state, reward, done = game.step_with_action(action)  # ✅ safe for training
                episode_reward += reward
                steps += 1
            
            score_bonus = game.score * 1000
            survival_bonus = min(steps, 1000)
            efficiency_bonus = (game.score * 100) / max(steps, 1) if steps > 0 else 0
            
            fitness = episode_reward + score_bonus + survival_bonus + efficiency_bonus
            total_fitness += fitness
            total_scores += game.score
            total_moves += steps
            total_survival_time += game.survival_time
            safety_overrides += ai.safety_override_count
        
        genome.avg_score = total_scores / self.n_evaluations
        genome.avg_moves = total_moves / self.n_evaluations
        genome.avg_survival_time = total_survival_time / self.n_evaluations
        genome.avg_safety_overrides = safety_overrides / self.n_evaluations
        
        return total_fitness / self.n_evaluations
    
    def evolve_generation(self, callback: Optional[Callable] = None ):
        """Evolve one generation using (1+λ) evolution strategy"""
        fitness_scores = []
        for genome in self.population:
            fitness = self.evaluate_genome(genome)
            genome.fitness = fitness
            fitness_scores.append(fitness)
            self.current_iteration += 1

            if callback: # A way to notify the app that training is progressing
                callback(self.current_iteration + 1, self.best_genome.avg_score if self.best_genome else 0, self.generation + 1, self.best_genome.fitness if self.best_genome else 0)
        
        
        best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[best_idx]
        best_genome = self.population[best_idx]
        
        
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_genome = best_genome.copy()
            self.save_best_genome('best_genome.json')

            self.log(f"New best fitness: {self.best_fitness:.2f} "
                   f"(Score: {best_genome.avg_score:.1f}, "
                   f"Moves: {best_genome.avg_moves:.0f}, "
                   f"Time: {best_genome.avg_survival_time:.1f}s, "
                   f"Overrides: {best_genome.avg_safety_overrides:.1f})")
        
        avg_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': current_best_fitness,
            'avg_fitness': avg_fitness,
            'std_fitness': std_fitness,
            'best_score': best_genome.avg_score,
            'best_moves': best_genome.avg_moves,
            'best_survival_time': best_genome.avg_survival_time,
            'safety_overrides': best_genome.avg_safety_overrides
        })
        
        new_population = [best_genome.copy()]
        for _ in range(self.population_size - 1):
            offspring = best_genome.copy()
            offspring.mutate(self.mutation_rate)
            new_population.append(offspring)
        
        self.population = new_population
        self.generation += 1
    
    def run_evolution(self, max_generations: int = None, callback: Optional[Callable] = None):
        if max_generations is None:
            max_generations = self.max_generations
        """Run the evolution process"""
        self.log(f"Starting hybrid CGP-A* evolution with {self.population_size} genomes")
        self.log(f"Genome structure: {self.n_inputs} inputs, {self.n_outputs} outputs, {self.n_nodes} nodes")
        self.log(f"Mutation rate: {self.mutation_rate}, Evaluations per genome: {self.n_evaluations}")
        
        start_time = time.time()
        self.generation = 0  # Initialize generation counter
        
        try:
            for gen in range(max_generations):
                gen_start = time.time()
                self.evolve_generation(callback=callback)
                gen_time = time.time() - gen_start

                if callback:
                    callback(self.current_iteration + 1, self.best_genome.avg_score if self.best_genome else 0, self.generation, self.best_genome.fitness)
        
        
        except KeyboardInterrupt:
            self.log("\nEvolution interrupted by user.")
        
        total_time = time.time() - start_time
        self.log(f"\nEvolution completed in {total_time:.2f}s")
        self.log(f"Best fitness achieved: {self.best_fitness:.2f}")
        
        if self.best_genome:
            self.save_best_genome('best_genome.json')
            self.save_evolution_history('evolution_history.json')
        return self.best_genome
    
    def save_best_genome(self, filename: str):
        """Save the best genome to a JSON file"""
        if not self.best_genome:
            return
        
        genome_data = {
            'fitness': self.best_fitness,
            'generation': self.generation,
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'n_nodes': self.n_nodes,
            'nodes': []
        }
        
        for node in self.best_genome.nodes:
            node_data = {
                'type': node.type,
                'inputs': node.inputs,
                'active': node.active
            }
            if node.function:
                for name, func in CGPGenome.FUNCTIONS.items():
                    if func == node.function:
                        node_data['function'] = name
                        break
            genome_data['nodes'].append(node_data)
        
        with open(filename, 'w') as f:
            json.dump(genome_data, f, indent=2)
        
        self.log(f"Best genome saved to {filename}")
    
    @staticmethod
    def load_genome(filename: str) -> CGPGenome:
        """Load a genome from a JSON file"""
        with open(filename, 'r') as f:
            genome_data = json.load(f)
        
        genome = CGPGenome(genome_data['n_inputs'], genome_data['n_outputs'], 
                          genome_data['n_nodes'])
        
        for i, node_data in enumerate(genome_data['nodes']):
            genome.nodes[i].inputs = node_data['inputs']
            genome.nodes[i].active = node_data['active']
            if 'function' in node_data:
                genome.nodes[i].function = CGPGenome.FUNCTIONS[node_data['function']]
        
        genome.fitness = genome_data['fitness']
        return genome
    
    def save_evolution_history(self, filename: str):
        """Save evolution history to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.fitness_history, f, indent=2)
        self.log(f"Evolution history saved to {filename}")