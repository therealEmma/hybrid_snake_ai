import React, { useState, useEffect } from "react";
import { BarChart3, Play, Download, RefreshCw, PieChart } from "lucide-react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  LogarithmicScale,
  ArcElement,
} from "chart.js";
import { Bar, Pie } from "react-chartjs-2";
import type { ChartOptions } from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LogarithmicScale,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

interface ComparisonStatus {
  status: string;
  progress: number;
  results: any;
  error?: string;
}

const ComparisonPanel: React.FC = () => {
  const [comparisonStatus, setComparisonStatus] = useState<ComparisonStatus>({
    status: "idle",
    progress: 0,
    results: {},
  });
  const [selectedAlgorithms, setSelectedAlgorithms] = useState([
    "hybrid",
    "cgp",
    "a_star",
    "greedy",
    "dijkstra",
    "rule_based",
  ]);
  const [numRuns, setNumRuns] = useState(10);
  const [activeChart, setActiveChart] = useState("scores");
  const isLogScale = ["decision_times", "survival_times"].includes(activeChart);

  const algorithms = [
    { id: "hybrid", name: "Hybrid CGP", color: "#8b5cf6" },
    { id: "cgp", name: "Pure CGP", color: "#06b6d4" },
    { id: "a_star", name: "A* Pathfinding", color: "#10b981" },
    { id: "greedy", name: "Greedy Best-First", color: "#f59e0b" },
    { id: "dijkstra", name: "Dijkstra", color: "#ef4444" },
    { id: "rule_based", name: "Rule-Based", color: "#6b7280" },
  ];

  const chartTypes = [
    { id: "scores", name: "Average Scores", icon: BarChart3 },
    { id: "max_scores", name: "Maximum Scores", icon: BarChart3 },
    { id: "survival_times", name: "Survival Times", icon: BarChart3 },
    { id: "decision_times", name: "Decision Times", icon: BarChart3 },
    { id: "path_efficiency", name: "Path Efficiency", icon: BarChart3 },
    { id: "peak_memory", name: "Peak Memory (MB)", icon: BarChart3 },
    { id: "death_causes", name: "Death Causes", icon: PieChart },
  ];

  // Fetch initial status when component mounts
  useEffect(() => {
    const fetchInitialStatus = async () => {
      try {
        const response = await fetch("http://localhost:5000/api/comparison/status");
        const data = await response.json();
        setComparisonStatus(data);
      } catch (error) {
        console.error("Failed to fetch initial comparison status:", error);
      }
    };
    fetchInitialStatus();
  }, []);

  // Poll for status updates
  useEffect(() => {
    let isMounted = true;
    const interval = setInterval(async () => {
      if (comparisonStatus.status === "running" && isMounted) {
        try {
          const response = await fetch("http://localhost:5000/api/comparison/status");
          const data = await response.json();
          if (isMounted) {
            setComparisonStatus(data);
          }
        } catch (error) {
          console.error("Failed to fetch comparison status:", error);
        }
      }
    }, 1000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [comparisonStatus.status]);

  const startComparison = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/comparison/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          algorithms: selectedAlgorithms,
          num_runs: numRuns,
        }),
      });

      const data = await response.json();
      if (data.success) {
        // Don't reset the status here, let the polling update it
        // This prevents the progress bar from resetting when switching tabs
        await fetch("http://localhost:5000/api/comparison/status")
          .then(res => res.json())
          .then(data => setComparisonStatus(data))
          .catch(error => console.error("Failed to fetch comparison status:", error));
      }
    } catch (error) {
      console.error("Failed to start comparison:", error);
    }
  };

  const toggleAlgorithm = (algorithmId: string) => {
    setSelectedAlgorithms((prev) =>
      prev.includes(algorithmId)
        ? prev.filter((id) => id !== algorithmId)
        : [...prev, algorithmId]
    );
  };

  const chartOptions: ChartOptions<'bar'> = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#e2e8f0'
        }
      },
      title: {
        display: true,
        text: chartTypes.find(c => c.id === activeChart)?.name || 'Comparison',
        color: '#e2e8f0',
        font: {
          size: 16
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const value = Number(context.raw);
            const label = context.dataset.label || '';
            
            // Format time metrics
            if (activeChart === 'survival_times') {
              return `${label}: ${value.toFixed(2)} seconds`;
            } else if (activeChart === 'decision_times') {
              // Convert to milliseconds for decision times
              return `${label}: ${(value * 1000).toFixed(2)} ms`;
            }
            
            // Format other metrics
            if (activeChart === 'peak_memory') {
              return `${label}: ${value.toFixed(2)} MB`;
            }
            
            return `${label}: ${value.toFixed(1)}`;
          }
        }
      }
    },
    scales: {
      x: {
        ticks: {
          color: '#94a3b8'
        },
        grid: {
          color: '#374151'
        }
      },
      y: {
        type: isLogScale ? 'logarithmic' : 'linear',
        ticks: {
          color: '#94a3b8',
          callback: function(value) {
            const numValue = Number(value);
            
            // Format y-axis labels for time metrics
            if (activeChart === 'survival_times') {
              return `${numValue.toFixed(1)}s`;
            } else if (activeChart === 'decision_times') {
              // Convert to milliseconds for decision times
              return `${(numValue * 1000).toFixed(0)}ms`;
            }
            
            // Format other metrics
            if (activeChart === 'peak_memory') {
              return `${numValue.toFixed(1)}MB`;
            }
            
            return numValue.toLocaleString();
          }
        },
        grid: {
          color: '#374151'
        }
      }
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">
          Algorithm Comparison
        </h2>
        <p className="text-slate-300">
          Compare the performance of different AI algorithms
        </p>
      </div>

      {/* Configuration */}
      <div className="bg-slate-700/50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">
          Comparison Settings
        </h3>

        <div className="space-y-4">
          <div>
            <label className="block text-slate-300 text-sm font-medium mb-2">
              Number of Runs per Algorithm
            </label>
            <input
              type="number"
              value={numRuns}
              onChange={(e) => setNumRuns(Number(e.target.value))}
              className="w-32 bg-slate-600 text-white px-3 py-2 rounded-lg border border-slate-500 focus:border-purple-500 focus:outline-none"
              disabled={comparisonStatus.status === "running"}
              min="1"
              max="100"
            />
          </div>

          <div>
            <label className="block text-slate-300 text-sm font-medium mb-3">
              Select Algorithms to Compare
            </label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {algorithms.map((algo) => (
                <label
                  key={algo.id}
                  className="flex items-center gap-2 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={selectedAlgorithms.includes(algo.id)}
                    onChange={() => toggleAlgorithm(algo.id)}
                    disabled={comparisonStatus.status === "running"}
                    className="w-4 h-4 text-purple-600 bg-slate-600 border-slate-500 rounded focus:ring-purple-500"
                  />
                  <span className="text-slate-300">{algo.name}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Start Comparison Button */}
      <div className="flex justify-center">
        <button
          onClick={startComparison}
          disabled={
            comparisonStatus.status === "running" ||
            selectedAlgorithms.length === 0
          }
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white px-8 py-3 rounded-lg font-medium transition-colors text-lg"
        >
          {comparisonStatus.status === "running" ? (
            <>
              <RefreshCw size={24} className="animate-spin" />
              Running Comparison
            </>
          ) : (
            <>
              <Play size={24} />
              Start Comparison
            </>
          )}
        </button>
      </div>

      {/* Progress */}
      {comparisonStatus.status === "running" && (
        <div className="bg-slate-700/50 rounded-lg p-6">
          <div className="flex justify-between text-sm text-slate-300 mb-2">
            <span>Progress</span>
            <span>{comparisonStatus.progress.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-slate-600 rounded-full h-3">
            <div
              className="bg-blue-500 h-3 rounded-full transition-all duration-300"
              style={{ width: `${comparisonStatus.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Results */}
      {comparisonStatus.status === "completed" && comparisonStatus.results && (
        <div className="space-y-6">
          {/* Chart Type Selector */}
          <div className="flex flex-wrap gap-2 justify-center">
            {chartTypes.map((chart) => (
              <button
                key={chart.id}
                onClick={() => setActiveChart(chart.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  activeChart === chart.id
                    ? "bg-purple-600 text-white"
                    : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                }`}
              >
                <chart.icon size={16} />
                {chart.name}
              </button>
            ))}
          </div>

          {/* Chart */}
          <div className="bg-slate-700/50 rounded-lg p-6">
            {activeChart !== 'death_causes' && comparisonStatus.results[activeChart] && (
              <Bar
                data={comparisonStatus.results[activeChart]}
                options={chartOptions}
              />
            )}
             {activeChart === "death_causes" && comparisonStatus.results.death_causes && (
              <div>
                <h3 className="text-xl font-semibold text-white mb-4 text-center">Death Cause Analysis</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                  {Object.entries(comparisonStatus.results.death_causes).map(([algo, data]: [string, any]) => (
                    <div key={algo}>
                      <h4 className="text-lg font-semibold text-slate-300 mb-2 text-center capitalize">{algo.replace(/_/g, ' ')}</h4>
                      <Pie 
                        data={data} 
                        options={{
                          plugins: {
                            legend: {
                              position: 'bottom',
                              labels: {
                                color: '#e2e8f0',
                                font: {
                                  size: 12
                                },
                                padding: 20
                              }
                            },
                            tooltip: {
                              callbacks: {
                                label: function(context) {
                                  const label = context.label || '';
                                  const value = context.raw || 0;
                                  return `${label}: ${value} runs`;
                                }
                              }
                            }
                          }
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Download Results */}
          <div className="flex justify-center">
            <button
              onClick={() => {
                const dataStr = JSON.stringify(
                  comparisonStatus.results,
                  null,
                  2
                );
                const dataBlob = new Blob([dataStr], {
                  type: "application/json",
                });
                const url = URL.createObjectURL(dataBlob);
                const link = document.createElement("a");
                link.href = url;
                link.download = "comparison_results.json";
                link.click();
                URL.revokeObjectURL(url);
              }}
              className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg font-medium transition-colors"
            >
              <Download size={20} />
              Download Results
            </button>
          </div>
        </div>
      )}

      {comparisonStatus.status === "error" && comparisonStatus.error && (
        <div className="bg-red-900/50 border border-red-700 rounded-lg p-4">
          <p className="text-red-400 font-medium">
            Error: {comparisonStatus.error}
          </p>
        </div>
      )}
    </div>
  );
};

export default ComparisonPanel;
