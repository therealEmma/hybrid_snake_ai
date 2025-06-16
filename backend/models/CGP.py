import platform
import pygame
import random
import numpy as np
import math
from enum import Enum
from typing import List, Tuple, Optional, Callable
import json
import time
import matplotlib.pyplot as plt
import os
import tracemalloc
from .common import DeathCause

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class CGPGenome:
    """Simplified Cartesian Genetic Programming genome representation"""
    
    FUNCTIONS = {
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y,
        'div': lambda x, y: x / y if abs(y) > 1e-6 else 1.0,
        'tanh': lambda x, y: math.tanh(x),
        'max': lambda x, y: max(x, y),
        'min': lambda x, y: min(x, y),
        'gt': lambda x, y: 1.0 if x > y else 0.0,
        'lt': lambda x, y: 1.0 if x < y else 0.0
    }
    
    def __init__(self, n_inputs: int, n_outputs: int, n_nodes: int):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes
        self.function_names = list(self.FUNCTIONS.keys())
        
        self.nodes = []
        for i in range(n_inputs):
            self.nodes.append(CGPNode('input', [i]))
        
        for i in range(n_nodes):
            max_input_idx = self.n_inputs + i - 1
            if max_input_idx < self.n_inputs:
                max_input_idx = self.n_inputs - 1
            input1 = random.randint(0, max_input_idx)
            input2 = random.randint(0, max_input_idx)
            function_name = random.choice(self.function_names)
            function = self.FUNCTIONS[function_name]
            self.nodes.append(CGPNode('function', [input1, input2], function))
        
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
            if self.nodes[node_idx].active:
                continue
            self.nodes[node_idx].active = True
            for input_idx in self.nodes[node_idx].inputs:
                if not self.nodes[input_idx].active:
                    active_queue.append(input_idx)
    
    def evaluate(self, inputs: List[float]) -> List[float]:
        """Evaluate the CGP program with given inputs"""
        node_values = [0.0] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            if node.active:
                node_values[i] = node.evaluate(inputs, node_values)
        output_start = self.n_inputs + self.n_nodes
        return [node_values[i] for i in range(output_start, len(self.nodes))]
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate the genome"""
        function_start = self.n_inputs
        function_end = self.n_inputs + self.n_nodes
        
        for i in range(function_start, function_end):
            node = self.nodes[i]
            for j in range(len(node.inputs)):
                if random.random() < mutation_rate:
                    max_input_idx = self.n_inputs + (i - self.n_inputs) - 1
                    if max_input_idx < self.n_inputs:
                        max_input_idx = self.n_inputs - 1
                    node.inputs[j] = random.randint(0, max_input_idx)
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
        return new_genome


class SnakeGame:
    """Snake game environment optimized for AI training"""
    
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT, genome: CGPGenome = None):
        tracemalloc.start()
        self.width = width
        self.height = height
        self.start_snapshot = None
        self.end_snapshot = None
        self.genome = genome
        self.optimal_path_lengths = []
        self.decision_times = []
        self.death_cause = None
        self.path_lengths = []
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        center_x, center_y = self.width // 2, self.height // 2
        self.snake = [(center_x, center_y), (center_x - 1, center_y), (center_x - 2, center_y)]
        self.direction = Direction.RIGHT
        self.food = self._generate_food()
        self.score = 0
        self.moves = 0
        self.max_moves = self.width * self.height * 2
        self.game_over = False
        self.start_time = time.time()
        self.survival_time = 0.0
        self.path_lengths = []
        self.decision_times = []
        self.death_cause = None
        return self.get_state()
    
    def _generate_food(self):
        """Generate food at random position not occupied by snake"""
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food
    
    def get_action(self) -> int:
        """Get action from AI given current state"""
        outputs = self.genome.evaluate(self.get_internal_state())
        self.path_lengths.append(1)
        self.optimal_path_lengths.append(self.width + self.height)
        return outputs.index(max(outputs)) % 4

    async def step(self):
        """Execute one game step with given action"""
        self.start_snapshot = tracemalloc.take_snapshot()
        if self.game_over:
            stats = self.end_snapshot.compare_to(self.start_snapshot, 'lineno')
            tracemalloc.stop()
            return self.get_state(), 0, True
        
        action = self.get_action()
        self.direction = self._action_to_direction(action)
        head = self.snake[0]
        new_head = (head[0] + self.direction.value[0], head[1] + self.direction.value[1])
        
        if (new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            self.game_over = True
            self.death_cause = DeathCause.WALL
            self.survival_time = time.time() - self.start_time
            return self.get_state(), -100, True
        
        if new_head in self.snake:
            self.game_over = True
            self.death_cause = DeathCause.SELF
            self.survival_time = time.time() - self.start_time
            return self.get_state(), -100, True
        
        self.snake.insert(0, new_head)
        reward = -0.1  # Base cost per move
        old_distance = math.sqrt((head[0] - self.food[0])**2 + (head[1] - self.food[1])**2)
        new_distance = math.sqrt((new_head[0] - self.food[0])**2 + (new_head[1] - self.food[1])**2)
        
        if new_head == self.food:
            self.score += 1
            reward = 100
            self.food = self._generate_food()
        else:
            self.snake.pop()
            reward += 10 / (new_distance + 1) if new_distance < old_distance else -1.0
        
        # Penalize moves that bring head closer to body
        body_distances = self._get_body_distances(new_head)
        min_body_distance = min([d for d in body_distances if d > 0], default=self.width)
        if min_body_distance < 5:
            penalty = 10 * (len(self.snake) / 10) / (min_body_distance + 1)
            reward -= penalty
        
        self.moves += 1
        if self.moves >= self.max_moves:
            self.game_over = True
            self.death_cause = DeathCause.TIMEOUT
            self.survival_time = time.time() - self.start_time
            reward = -50
        
        self.end_snapshot = tracemalloc.take_snapshot()
        return self.get_state(), reward, self.game_over
    
    def _get_body_distances(self, head):
        """Calculate distance to nearest body segment in each direction"""
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        distances = []
        for dx, dy in directions:
            distance = self.width + self.height  # Max possible distance
            x, y = head[0], head[1]
            steps = 1
            while 0 <= x + dx * steps < self.width and 0 <= y + dy * steps < self.height:
                pos = (x + dx * steps, y + dy * steps)
                if pos in self.snake:  # Skip head
                    distance = steps
                    break
                steps += 1
            distances.append(distance)
        return distances
    
    def _get_body_density(self, head):
        """Count body segments within a 5x5 grid around the head"""
        count = 0
        x, y = head
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) in self.snake[1:]:
                    count += 1
        return count
    
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
            "death_cause": self.death_cause.value if self.death_cause else None,
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

    def get_internal_state(self):
        """Get current game state as feature vector for AI"""
        head = self.snake[0]
        food_dx = self.food[0] - head[0]
        food_dy = self.food[1] - head[1]
        food_distance = math.sqrt(food_dx**2 + food_dy**2)
        
        # Calculate angle to food relative to snake direction
        dir_vector = self.direction.value
        food_vector = (food_dx, food_dy)
        dot_product = dir_vector[0] * food_vector[0] + dir_vector[1] * food_vector[1]
        dir_magnitude = math.sqrt(dir_vector[0]**2 + dir_vector[1]**2)
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
        
        body_distances = self._get_body_distances(head)
        body_density = self._get_body_density(head)
        
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
                    normalized_state.append(val / math.sqrt(self.width**2 + self.height**2))
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

        self.decision_times.append(time.time() - self.start_time)
        return normalized_state

class CGPNode:
    """A node in the Cartesian Genetic Programming graph"""
    
    def __init__(self, node_type: str, inputs: List[int], function: Optional[Callable] = None):
        self.type = node_type
        self.inputs = inputs
        self.function = function
        self.value = 0.0
        self.active = False
    
    def evaluate(self, input_values: List[float], node_values: List[float]) -> float:
        """Evaluate this node given input and node values"""
        if self.type == 'input':
            return input_values[self.inputs[0]]
        elif self.type == 'function':
            func_inputs = [node_values[i] for i in self.inputs]
            return self.function(*func_inputs)
        elif self.type == 'output':
            return node_values[self.inputs[0]]
        return 0.0

class CGPEvolution:
    """Evolution algorithm for CGP genomes"""
    
    def __init__(self, population_size: int = 100, n_inputs: int = 22, n_outputs: int = 4,
                 n_nodes: int = 50, mutation_rate: float = 0.1, n_evaluations: int = 5,
                 log_file: Optional[str] = None):
        self.population_size = population_size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes
        self.mutation_rate = mutation_rate
        self.n_evaluations = n_evaluations
        self.log_file = log_file
        self.population = [CGPGenome(n_inputs, n_outputs, n_nodes) for _ in range(population_size)]
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_genome = None
        self.fitness_history = []
    
    def log(self, message: str):
        """Log message to both terminal and file"""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
                f.flush()
                
    
    def evaluate_genome(self, genome: CGPGenome) -> Tuple[float, float, float, float]:
        """Evaluate a genome's fitness by playing Snake, returning fitness, avg score, avg steps, avg survival time"""
        total_fitness = 0.0
        total_score = 0
        total_steps = 0
        total_survival_time = 0.0
        for _ in range(self.n_evaluations):
            game = SnakeGame(render=False)
            ai = SnakeAI(genome)
            state = game.reset()
            total_reward = 0
            steps = 0
            while not game.game_over and steps < 2000:
                action = ai.get_action(state)
                state, reward, done = game.step()
                total_reward += reward
                steps += 1
                if done:
                    break
            score_fitness = game.score * 1000
            survival_fitness = steps * 0.1 + game.survival_time * 0.5  # Include survival time
            efficiency_bonus = game.score * 10 if steps > 0 else 0
            episode_fitness = score_fitness + survival_fitness + efficiency_bonus
            total_fitness += episode_fitness
            total_score += game.score
            total_steps += steps
            total_survival_time += game.survival_time
            game.close()
        return (
            total_fitness / self.n_evaluations,
            total_score / self.n_evaluations,
            total_steps / self.n_evaluations,
            total_survival_time / self.n_evaluations
        )
    
    def evaluate_population(self):
        """Evaluate fitness for all genomes in population"""
        self.log(f"Evaluating generation {self.generation}...")
        avg_score = 0
        avg_steps = 0
        avg_survival_time = 0.0
        for i, genome in enumerate(self.population):
            fitness, score, steps, survival_time = self.evaluate_genome(genome)
            genome.fitness = fitness
            avg_score += score
            avg_steps += steps
            avg_survival_time += survival_time
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome.copy()
            if (i + 1) % 10 == 0:
                self.log(f"  Evaluated {i + 1}/{len(self.population)} genomes")
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        avg_fitness = sum(g.fitness for g in self.population) / len(self.population)
        avg_score /= len(self.population)
        avg_steps /= len(self.population)
        avg_survival_time /= len(self.population)
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': avg_fitness,
            'worst_fitness': self.population[-1].fitness,
            'avg_score': avg_score,
            'avg_steps': avg_steps,
            'avg_survival_time': avg_survival_time
        })
        self.log(f"Generation {self.generation}: Best Fitness={self.best_fitness:.2f}, "
                 f"Avg Fitness={avg_fitness:.2f}, Worst Fitness={self.population[-1].fitness:.2f}, "
                 f"Avg Score={avg_score:.2f}, Avg Steps={avg_steps:.2f}, "
                 f"Avg Survival Time={avg_survival_time:.2f}s")
    
    def evolve_generation(self):
        """Evolve one generation using (1+λ) evolution strategy"""
        best_genome = self.population[0].copy()
        new_population = [best_genome]
        for _ in range(self.population_size - 1):
            offspring = best_genome.copy()
            offspring.mutate(self.mutation_rate)
            new_population.append(offspring)
        self.population = new_population
        self.generation += 1
    
    def plot_fitness_history(self):
        """Plot fitness, food eaten, and survival time history using Matplotlib"""
        generations = [data['generation'] for data in self.fitness_history]
        avg_fitness = [data['avg_fitness'] for data in self.fitness_history]
        avg_score = [data['avg_score'] for data in self.fitness_history]
        avg_survival_time = [data['avg_survival_time'] for data in self.fitness_history]
        
        os.makedirs('training_data', exist_ok=True)
        
        # Plot 1: Average Fitness
        plt.figure(figsize=(10, 6))
        plt.plot(generations, avg_fitness, 'b-', label='Average Fitness')
        plt.title('Average Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness')
        plt.grid(True)
        plt.legend()
        plt.savefig('training_data/fitness_plot.png')
        plt.close()
        
        # Plot 2: Average Food Eaten
        plt.figure(figsize=(10, 6))
        plt.plot(generations, avg_score, 'g-', label='Average Food Eaten')
        plt.title('Average Food Eaten Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Average Food Eaten')
        plt.grid(True)
        plt.legend()
        plt.savefig('training_data/food_eaten_plot.png')
        plt.close()
        
        # Plot 3: Average Survival Time
        plt.figure(figsize=(10, 6))
        plt.plot(generations, avg_survival_time, 'r-', label='Average Survival Time')
        plt.title('Average Survival Time Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Average Survival Time (seconds)')
        plt.grid(True)
        plt.legend()
        plt.savefig('training_data/survival_time_plot.png')
        plt.close()
    
    def run_evolution(self, n_generations: int):
        """Run evolution for specified number of generations"""
        self.log(f"Starting evolution for {n_generations} generations...")
        self.log(f"Population size: {self.population_size}")
        self.log(f"CGP parameters: {self.n_nodes} nodes")
        self.log(f"Mutation rate: {self.mutation_rate}")
        self.log("-" * 50)
        for generation in range(n_generations):
            self.evaluate_population()
            if generation < n_generations - 1:
                self.evolve_generation()
        self.plot_fitness_history()
        self.log("\nEvolution completed!")
        self.log(f"Best fitness achieved: {self.best_fitness:.2f}")
        return self.best_genome
    
    def save_best_genome(self, filename: str):
        """Save the best genome to a file"""
        if self.best_genome is None:
            self.log("No best genome to save!")
            return
        genome_data = {
            'n_inputs': self.best_genome.n_inputs,
            'n_outputs': self.best_genome.n_outputs,
            'n_nodes': self.best_genome.n_nodes,
            'fitness': self.best_genome.fitness,
            'nodes': []
        }
        for node in self.best_genome.nodes:
            node_data = {
                'type': node.type,
                'inputs': node.inputs,
                'active': node.active
            }
            if node.function is not None:
                for name, func in CGPGenome.FUNCTIONS.items():
                    if func == node.function:
                        node_data['function_name'] = name
                        break
            genome_data['nodes'].append(node_data)
        genome_data['fitness_history'] = self.fitness_history
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(genome_data, f, indent=2)
        self.log(f"Best genome saved to {filename}")

def load_genome(filename: str) -> CGPGenome:
        """Load a genome from a file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            genome = CGPGenome(data['n_inputs'], data['n_outputs'], data['n_nodes'])
            for i, node_data in enumerate(data['nodes']):
                genome.nodes[i].type = node_data['type']
                genome.nodes[i].inputs = node_data['inputs']
                genome.nodes[i].active = node_data['active']
                if 'function_name' in node_data:
                    function_name = node_data['function_name']
                    genome.nodes[i].function = CGPGenome.FUNCTIONS[function_name]
            genome.fitness = data['fitness']
            return genome
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return None
        except Exception as e:
            print(f"Error loading genome: {e}")
            return None