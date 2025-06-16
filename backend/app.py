from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import json
import os
import asyncio
import threading
import time
from datetime import datetime
import uuid
import shutil
from collections import defaultdict
import numpy as np
from models.A_star import SnakeGame
from models.greedy import SnakeGame as GreedySnakeAI
from models.dijkstra import SnakeGame as DijkstraSnakeAI
from models.rule_based import RuleBasedSnakeAI
from models.hybrid import SnakeGame as HybridSnakeGame, CGPEvolution as evolution, HybridSnakeAI
from models.CGP import SnakeGame as CGPSnakeGame, CGPGenome, load_genome as cgp_load_genome 
from models.rule_based import SnakeGame as RuleBasedSnakeAI
import tracemalloc
from models.common import DeathCause

# # Create mock classes for development
# class SnakeGame:
#     def __init__(self, algorithm, **kwargs):
#         self.algorithm = algorithm
#         self.snake = [(20, 15), (19, 15), (18, 15)]
#         self.food = (25, 15)
#         self.score = 0
#         self.game_over = False
#         self.direction = "RIGHT"
    
#     def get_state(self):
#         return {
#             "snake": self.snake,
#             "food": self.food,
#             "score": self.score,
#             "game_over": self.game_over,
#             "direction": self.direction
#         }

class Direction:
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

# async def run_simulation(algorithm, **kwargs):
#     # Mock simulation results
#     import random
#     await asyncio.sleep(0.1)  # Simulate processing time
#     return {
#         "score": random.randint(1, 25),
#         "survival_time": random.uniform(1.0, 30.0),
#         "moves": random.randint(50, 500),
#         "path_efficiency": random.uniform(0.01, 0.1),
#         "decision_time": random.uniform(0.001, 0.01),
#         "path_length_efficiency": random.uniform(0.5, 1.0),
#         "death_cause": random.choice([DeathCause.WALL, DeathCause.SELF, DeathCause.TIMEOUT]),
#         "peak_memory": random.uniform(10, 100)
#     }

app = Flask(__name__, static_folder='../dist', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})


# Global variables for game state and training
current_game = None
training_status = {"status": "idle", "progress": 0, "generation": 0, "best_score": 0}
comparison_results = {}

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("training_data", exist_ok=True)
os.makedirs("static/charts", exist_ok=True)

def get_algorithm_instance(algorithm_name):
    """Create and return an instance of the selected algorithm"""
    if algorithm_name == 'a_star':
        return SnakeGame()
    elif algorithm_name == 'greedy':
        return GreedySnakeAI()
    elif algorithm_name == 'dijkstra':
        return DijkstraSnakeAI()
    elif algorithm_name == 'rule_based':
        return RuleBasedSnakeAI()
    elif algorithm_name == 'hybrid':
        try:
            hybrid_genome = evolution.load_genome("models/best_genome.json")
            return HybridSnakeGame(genome=hybrid_genome)
        except Exception as e:
            print(f"Error loading hybrid genome: {e}")
            return HybridSnakeGame()
    elif algorithm_name == 'cgp':
        # try:
        cgp_genome = cgp_load_genome("models/best_snake_ai.json")
        return CGPSnakeGame(genome=cgp_genome)
        # except:x``
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/game/start', methods=['POST'])
def start_game():
    global current_game
    data = request.get_json()
    algorithm = data.get('algorithm', 'hybrid')
    
    try:
        # Create algorithm instance
        current_game = get_algorithm_instance(algorithm)
        return jsonify({
            "success": True,
            "game_state": current_game.get_state() if hasattr(current_game, 'get_state') else {
                "snake": current_game.snake,
                "food": current_game.food,
                "score": current_game.score,
                "game_over": current_game.game_over,
                "direction": current_game.direction.name if hasattr(current_game.direction, 'name') else str(current_game.direction)
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/game/step', methods=['POST'])
def game_step():
    global current_game
    if not current_game:
        return jsonify({"success": False, "error": "No active game"}), 400
    
    try:
        # Run game step in async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(current_game.step())
        
        loop.close()
        
        return jsonify({
            "success": True,
            "game_state": current_game.get_state() if hasattr(current_game, 'get_state') else {
                "snake": current_game.snake,
                "food": current_game.food,
                "score": current_game.score,
                "game_over": current_game.game_over,
                "direction": current_game.direction.name if hasattr(current_game.direction, 'name') else str(current_game.direction)
            },
            "metrics": current_game.get_metrics() if hasattr(current_game, 'get_metrics') else {}
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/game/state', methods=['GET'])
def get_game_state():
    global current_game
    if not current_game:
        return jsonify({"success": False, "error": "No active game"}), 400
    
    return jsonify({
        "success": True,
        "game_state": current_game.get_state() if hasattr(current_game, 'get_state') else {
            "snake": current_game.snake,
            "food": current_game.food,
            "score": current_game.score,
            "game_over": current_game.game_over,
            "direction": current_game.direction.name if hasattr(current_game.direction, 'name') else str(current_game.direction)
        }
    })

@app.route('/api/training/start', methods=['POST'])
def start_training():
    global training_status
    data = request.get_json()

    try:
        generations = int(data.get('generations', 50))
        population_size = int(data.get('population_size', 100))
        mutation_rate = float(data.get('mutation_rate', 0.1))
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid parameters: {e}"}), 400

    if training_status.get("status") == "training":
        return jsonify({"success": False, "error": "Training already in progress"}), 400

    def train_model(generations, population_size, mutation_rate):
        global training_status
        training_status.update({
            "status": "training", "progress": 0, "generation": 0, "best_score": 0
        })

        try:
            print("🚀 Starting training for Hybrid AI")
            evo = evolution(
                population_size=population_size,
                max_generations=generations,
                mutation_rate=mutation_rate
            )

            generation_fitness_and_best_score = {}

            def callback(current_iteration, best_score, generation, fitness = 0):
                training_status.update({
                    "generation": generation, 
                    "progress": (current_iteration / (generations * population_size)) * 100,
                    "best_score": best_score,
                    "fitness": fitness
                })

                generation_fitness_and_best_score.update({ generation: { "score": best_score, "fitness": fitness} })
                print("Generation and fitness", generation_fitness_and_best_score)
                if training_status["status"] == "stopped":
                    raise Exception("Training stopped by user")

            evo.run_evolution(callback=callback)

            with open("training_data/fitness_log.json", "w") as f:
                fitness_data = extract_fitness_data(fitness_dict=generation_fitness_and_best_score)
                json.dump(fitness_data, f, indent=2)

            training_status.update({"status": "completed"})

        except Exception as e:
            import traceback
            print("Training Error:", e)
            traceback.print_exc()
            training_status.update({
                "status": "error",
                "error": str(e),
                "progress": 0,
                "generation": 0,
                "best_score": 0
            })

    training_thread = threading.Thread(
        target=train_model,
        args=(generations, population_size, mutation_rate)
    )
    training_thread.daemon = True
    training_thread.start()

    return jsonify({"success": True, "message": "Training started"})


def extract_fitness_data(fitness_dict: dict):
    fitness_data = []
    for k, v in fitness_dict.items():
        fitness_data.append({"generation": k, **v })
    return fitness_data
    

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    return jsonify(training_status)

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    global training_status
    if training_status["status"] == "training":
        training_status["status"] = "stopped"
        return jsonify({"success": True, "message": "Training stopped"})
    return jsonify({"success": False, "error": "No training in progress"}), 400

@app.route('/api/training/fitness_log', methods=['GET'])
def get_fitness_log():
    try:
        with open("training_data/fitness_log.json", "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


async def run_simulation(algorithm):
    # Initialize the game instance for the given algorithm
    game = get_algorithm_instance(algorithm)

    tracemalloc.start()
    game.start_time = asyncio.get_event_loop().time()
    
    while not game.game_over:
        await game.step()
    
    # Get peak memory usage
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    metrics = game.get_metrics()
    # Add peak memory to metrics in MB
    metrics['peak_memory'] = peak_memory_bytes / (1024 * 1024)
    
    return metrics

@app.route('/api/comparison/status', methods=['GET'])
def get_comparison_status():
    return jsonify(comparison_results)

def process_comparison_results(metrics):
    algorithms = list(metrics.keys())
    death_cause_types = [e.value for e in DeathCause]
    
    # Calculate averages for each metric
    avg_scores = [np.mean([run["score"] for run in metrics[algo]]) for algo in algorithms]
    max_scores = [max([run["score"] for run in metrics[algo]]) for algo in algorithms]
    
    # Process time metrics - ensure they're in reasonable ranges
    avg_survival_times = []
    avg_decision_times = []
    for algo in algorithms:
        # Filter out any unreasonable survival times (e.g., > 1000 seconds)
        survival_times = [min(run["survival_time"], 1000) for run in metrics[algo]]
        avg_survival_times.append(np.mean(survival_times))
        
        # Filter out any unreasonable decision times (e.g., > 1 second)
        decision_times = [min(run["decision_time"], 1.0) for run in metrics[algo]]
        avg_decision_times.append(np.mean(decision_times))
    
    avg_path_efficiency = [np.mean([run["path_efficiency"] for run in metrics[algo]]) for algo in algorithms]
    avg_peak_memory = [np.mean([run.get("peak_memory", 0) for run in metrics[algo]]) for algo in algorithms]

    # Process death causes for pie charts
    death_causes_data = {}
    for algo in algorithms:
        counts = {cause: 0 for cause in death_cause_types}
        total_runs = len(metrics[algo])
        
        for run in metrics[algo]:
            cause = run.get("death_cause")
            if cause in counts:
                counts[cause] += 1
        
        # Calculate percentages and create labels with percentages
        labels = []
        for cause in death_cause_types:
            count = counts[cause]
            percentage = (count / total_runs * 100) if total_runs > 0 else 0
            labels.append(f"{cause} ({percentage:.1f}%)")
        
        death_causes_data[algo] = {
            "labels": labels,
            "datasets": [{
                "data": list(counts.values()),
                "backgroundColor": ['#ef4444', '#f59e0b', '#8b5cf6'],
                "hoverBackgroundColor": ['#ef4444', '#f59e0b', '#8b5cf6']
            }]
        }

    # Prepare data for Chart.js
    chart_data = {
        "scores": {
            "labels": algorithms,
            "datasets": [{
                "label": "Average Score",
                "data": avg_scores,
                "backgroundColor": "rgba(54, 162, 235, 0.6)",
                "borderColor": "rgba(54, 162, 235, 1)",
                "borderWidth": 1
            }]
        },
        "max_scores": {
            "labels": algorithms,
            "datasets": [{
                "label": "Maximum Score",
                "data": max_scores,
                "backgroundColor": "rgba(255, 99, 132, 0.6)",
                "borderColor": "rgba(255, 99, 132, 1)",
                "borderWidth": 1
            }]
        },
        "survival_times": {
            "labels": algorithms,
            "datasets": [{
                "label": "Average Survival Time (s)",
                "data": avg_survival_times,
                "backgroundColor": "rgba(75, 192, 192, 0.6)",
                "borderColor": "rgba(75, 192, 192, 1)",
                "borderWidth": 1
            }]
        },
        "decision_times": {
            "labels": algorithms,
            "datasets": [{
                "label": "Average Decision Time (s)",
                "data": avg_decision_times,
                "backgroundColor": "rgba(153, 102, 255, 0.6)",
                "borderColor": "rgba(153, 102, 255, 1)",
                "borderWidth": 1
            }]
        },
        "path_efficiency": {
            "labels": algorithms,
            "datasets": [{
                "label": "Path Efficiency",
                "data": avg_path_efficiency,
                "backgroundColor": "rgba(255, 159, 64, 0.6)",
                "borderColor": "rgba(255, 159, 64, 1)",
                "borderWidth": 1
            }]
        },
        "peak_memory": {
            "labels": algorithms,
            "datasets": [{
                "label": "Average Peak Memory (MB)",
                "data": avg_peak_memory,
                "backgroundColor": "rgba(255, 206, 86, 0.6)",
                "borderColor": "rgba(255, 206, 86, 1)",
                "borderWidth": 1
            }]
        },
        "death_causes": death_causes_data
    }
    
    return chart_data

@app.route('/api/comparison/run', methods=['POST'])
def run_comparison():
    global comparison_results
    data = request.get_json()
    algorithms = data.get('algorithms', ['a_star', 'greedy', 'dijkstra', 'rule_based', 'hybrid', 'cgp'])
    num_runs = data.get('num_runs', 10)

    async def run_comparison_async():
        global comparison_results
        # Only reset if we're not already running
        if comparison_results.get("status") != "running":
            comparison_results = {"status": "running", "progress": 0, "results": {}}
        try:
            metrics = defaultdict(list)
            total_runs = len(algorithms) * num_runs
            completed_runs = 0

            for algo in algorithms:
                for i in range(num_runs):
                    result = await run_simulation(algo)
                    metrics[algo].append(result)
                    completed_runs += 1
                    progress = (completed_runs / total_runs) * 100
                    # Update progress while preserving other state
                    comparison_results.update({
                        "status": "running",
                        "progress": progress
                    })

            chart_data = process_comparison_results(metrics)
            comparison_results.update({
                "status": "completed",
                "progress": 100,
                "results": chart_data
            })
        except Exception as e:
            comparison_results.update({
                "status": "error",
                "error": str(e),
                "progress": 0,
                "results": {}
            })

    def run_comparison_sync():
        asyncio.run(run_comparison_async())

    comparison_thread = threading.Thread(target=run_comparison_sync)
    comparison_thread.daemon = True
    comparison_thread.start()

    return jsonify({"success": True, "message": "Comparison started"})



@app.route('/api/models/list', methods=['GET'])
def list_models():
    models = []
    if os.path.exists("models"):
        for filename in os.listdir("models"):
            if filename.endswith('.json'):
                filepath = os.path.join("models", filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    models.append({
                        "filename": filename,
                        "timestamp": data.get("timestamp", "Unknown"),
                        "best_score": data.get("best_score", "Unknown"),
                        "generation": data.get("generation", "Unknown")
                    })
                except:
                    models.append({
                        "filename": filename,
                        "timestamp": "Unknown",
                        "best_score": "Unknown",
                        "generation": "Unknown"
                    })
    return jsonify({"models": models})

@app.route('/api/models/load/<filename>', methods=['POST'])
def load_model(filename):
    filepath = os.path.join("models", filename)
    if not os.path.exists(filepath):
        return jsonify({"success": False, "error": "Model not found"}), 404
    
    try:
        # Copy to active model location
        shutil.copy(filepath, "models/best_genome.json")
        return jsonify({"success": True, "message": f"Model {filename} loaded successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)