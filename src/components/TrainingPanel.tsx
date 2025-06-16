import React, { useState, useEffect } from "react";
import { Play, Square, Settings, TrendingUp, Clock, Award } from "lucide-react";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";
import FitnessChart from "./FitnessChart";
import ScoreChart from "./ScoreChart";

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend
);

interface TrainingStatus {
  status: string;
  progress: number;
  generation: number;
  best_score: number;
  error?: string;
  fitness: number;
}

const TrainingPanel: React.FC = () => {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    status: "idle",
    progress: 0,
    generation: 0,
    best_score: 0,
    fitness: 0,
  });
  const [trainingConfig, setTrainingConfig] = useState({
    generations: 50,
    population_size: 100,
    mutation_rate: 0.1,
  });

  useEffect(() => {
    const interval = setInterval(async () => {
      if (trainingStatus.status === "training") {
        try {
          const response = await fetch(
            "http://localhost:5000/api/training/status"
          );
          const data = await response.json();
          setTrainingStatus(data);
        } catch (error) {
          console.error("Failed to fetch training status:", error);
        }
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [trainingStatus.status]);

  const startTraining = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(trainingConfig),
      });

      const data = await response.json();
      if (data.success) {
        setTrainingStatus({
          status: "training",
          progress: 0,
          generation: 0,
          best_score: 0,
          fitness: 0,
        });
      }
    } catch (error) {
      console.error("Failed to start training:", error);
    }
  };

  const stopTraining = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/training/stop", {
        method: "POST",
      });

      const data = await response.json();
      if (data.success) {
        setTrainingStatus((prev) => ({ ...prev, status: "stopped" }));
      }
    } catch (error) {
      console.error("Failed to stop training:", error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "training":
        return "text-blue-400";
      case "completed":
        return "text-green-400";
      case "error":
        return "text-red-400";
      case "stopped":
        return "text-yellow-400";
      default:
        return "text-slate-400";
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case "training":
        return "Training in Progress";
      case "completed":
        return "Training Completed";
      case "error":
        return "Training Failed";
      case "stopped":
        return "Training Stopped";
      default:
        return "Ready to Train";
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">
          AI Training Laboratory
        </h2>
        <p className="text-slate-300">
          Train new AI agents using Cartesian Genetic Programming
        </p>
      </div>

      {/* Training Configuration */}
      <div className="bg-slate-700/50 rounded-lg p-6">
        <div className="flex items-center gap-2 text-white mb-4">
          <Settings size={20} />
          <h3 className="text-lg font-semibold">Training Configuration</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-slate-300 text-sm font-medium mb-2">
              Generations
            </label>
            <input
              type="number"
              value={trainingConfig.generations}
              onChange={(e) =>
                setTrainingConfig((prev) => ({
                  ...prev,
                  generations: Number(e.target.value),
                }))
              }
              className="w-full bg-slate-600 text-white px-3 py-2 rounded-lg border border-slate-500 focus:border-purple-500 focus:outline-none"
              disabled={trainingStatus.status === "training"}
              min="1"
              max="1000"
            />
          </div>

          <div>
            <label className="block text-slate-300 text-sm font-medium mb-2">
              Population Size
            </label>
            <input
              type="number"
              value={trainingConfig.population_size}
              onChange={(e) =>
                setTrainingConfig((prev) => ({
                  ...prev,
                  population_size: Number(e.target.value),
                }))
              }
              className="w-full bg-slate-600 text-white px-3 py-2 rounded-lg border border-slate-500 focus:border-purple-500 focus:outline-none"
              disabled={trainingStatus.status === "training"}
              min="10"
              max="500"
            />
          </div>

          <div>
            <label className="block text-slate-300 text-sm font-medium mb-2">
              Mutation Rate
            </label>
            <input
              type="number"
              step="0.01"
              value={trainingConfig.mutation_rate}
              onChange={(e) =>
                setTrainingConfig((prev) => ({
                  ...prev,
                  mutation_rate: Number(e.target.value),
                }))
              }
              className="w-full bg-slate-600 text-white px-3 py-2 rounded-lg border border-slate-500 focus:border-purple-500 focus:outline-none"
              disabled={trainingStatus.status === "training"}
              min="0"
              max="1"
            />
          </div>

          <div></div>
        </div>
      </div>

      {/* Training Controls */}
      <div className="flex justify-center">
        {trainingStatus.status !== "training" ? (
          <button
            onClick={startTraining}
            className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-8 py-3 rounded-lg font-medium transition-colors text-lg"
          >
            <Play size={24} />
            Start Training
          </button>
        ) : (
          <button
            onClick={stopTraining}
            className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white px-8 py-3 rounded-lg font-medium transition-colors text-lg"
          >
            <Square size={24} />
            Stop Training
          </button>
        )}
      </div>

      {/* Training Status */}
      <div className="bg-slate-700/50 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Training Status</h3>
          <span
            className={`font-medium ${getStatusColor(trainingStatus.status)}`}
          >
            {getStatusText(trainingStatus.status)}
          </span>
        </div>

        {trainingStatus.status === "training" && (
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm text-slate-300 mb-1">
                <span>Progress</span>
                <span>{trainingStatus.progress.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-slate-600 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${trainingStatus.progress}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="flex items-center justify-center gap-2 text-blue-400 mb-1">
                  <Clock size={16} />
                  <span className="text-sm font-medium">Generation</span>
                </div>
                <p className="text-white text-xl font-bold">
                  {trainingStatus.generation}
                </p>
              </div>

              <div className="text-center">
                <div className="flex items-center justify-center gap-2 text-green-400 mb-1">
                  <Award size={16} />
                  <span className="text-sm font-medium">Best Score</span>
                </div>
                <p className="text-white text-xl font-bold">
                  {trainingStatus.best_score}
                </p>
              </div>

              <div className="text-center">
                <div className="flex items-center justify-center gap-2 text-purple-400 mb-1">
                  <TrendingUp size={16} />
                  <span className="text-sm font-medium">Fitness</span>
                </div>
                <p className="text-white text-xl font-bold">
                  {trainingStatus.fitness}
                </p>
              </div>
            </div>
          </div>
        )}

        {trainingStatus.status === "completed" && (
          <div className="text-center py-4">
            <div className="text-green-400 mb-2">
              <Award size={48} className="mx-auto" />
            </div>
            <p className="text-white text-lg font-semibold">
              Training completed successfully!
            </p>
            <p className="text-slate-300">
              Best score achieved: {trainingStatus.best_score}
            </p>
          </div>
        )}

        {trainingStatus.status === "completed" && (
          <div className="bg-slate-700/50 rounded-lg p-6 mt-4">
            <h3 className="text-lg font-semibold text-white mb-4">
              Score Progress Over Generations
            </h3>
            <ScoreChart />
          </div>
        )}

        {trainingStatus.status === "completed" && (
          <div className="bg-slate-700/50 rounded-lg p-6 mt-4">
            <h3 className="text-lg font-semibold text-white mb-4">
              Fitness Progress Over Generations
            </h3>
            <FitnessChart />
          </div>
        )}

        {trainingStatus.status === "error" && trainingStatus.error && (
          <div className="text-center py-4">
            <p className="text-red-400 font-medium">
              Error: {trainingStatus.error}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingPanel;
