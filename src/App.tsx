import  { useState } from 'react';
import { Play, Square, BarChart3, Brain, Settings, Zap } from 'lucide-react';
import GameVisualization from './components/GameVisualization';
import TrainingPanel from './components/TrainingPanel';
import ComparisonPanel from './components/ComparisonPanel';
import ModelManager from './components/ModelManager';

interface GameState {
  snake: [number, number][];
  food: [number, number];
  score: number;
  game_over: boolean;
  direction: string;
}

function App() {
  const [activeTab, setActiveTab] = useState('game');
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isGameRunning, setIsGameRunning] = useState(false);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('hybrid');

  const tabs = [
    { id: 'game', label: 'Game Visualization', icon: Play },
    { id: 'training', label: 'AI Training', icon: Brain },
    { id: 'comparison', label: 'Algorithm Comparison', icon: BarChart3 },
    { id: 'models', label: 'Model Manager', icon: Settings }
  ];

  const algorithms = [
    { id: 'hybrid', name: 'Hybrid CGP' },
    { id: 'cgp', name: 'Pure CGP' },
    { id: 'a_star', name: 'A* Pathfinding' },
    { id: 'greedy', name: 'Greedy Best-First' },
    { id: 'dijkstra', name: 'Dijkstra' },
    { id: 'rule_based', name: 'Rule-Based' }
  ];

  const startGame = async () => {
    try {
      console.log('Starting game with algorithm:', selectedAlgorithm);
      const response = await fetch('http://localhost:5000/api/game/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm: selectedAlgorithm })
      });
      
      const data = await response.json();
      if (data.success) {
        console.log(data.game_state)
        setGameState(data.game_state);
        setIsGameRunning(true);
      }
    } catch (error) {
      console.error('Failed to start game:', error);
    }
  };

  const stopGame = () => {
    setIsGameRunning(false);
    setGameState(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            <Zap className="text-yellow-400" />
            Snake AI Algorithm
          </h1>
          <p className="text-slate-300 text-lg">
            Advanced AI agents playing Snake using Cartesian Genetic Programming
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                  activeTab === tab.id
                    ? 'bg-purple-600 text-white shadow-lg shadow-purple-600/25'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white'
                }`}
              >
                <Icon size={20} />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Content Area */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
          {activeTab === 'game' && (
            <div>
              <div className="flex flex-col sm:flex-row gap-4 items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                  <select
                    value={selectedAlgorithm}
                    onChange={(e) => setSelectedAlgorithm(e.target.value)}
                    className="bg-slate-700 text-white px-4 py-2 rounded-lg border border-slate-600 focus:border-purple-500 focus:outline-none"
                    disabled={isGameRunning}
                  >
                    {algorithms.map((algo) => (
                      <option key={algo.id} value={algo.id}>
                        {algo.name}
                      </option>
                    ))}
                  </select>
                  
                  {!isGameRunning ? (
                    <button
                      onClick={startGame}
                      className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg font-medium transition-colors"
                    >
                      <Play size={20} />
                      Start Game
                    </button>
                  ) : (
                    <button
                      onClick={stopGame}
                      className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-lg font-medium transition-colors"
                    >
                      <Square size={20} />
                      Stop Game
                    </button>
                  )}
                </div>
                
                {gameState && (
                  <div className="text-white">
                    <span className="text-2xl font-bold">Score: {gameState.score}</span>
                  </div>
                )}
              </div>
              
              <GameVisualization 
                gameState={gameState}
                isRunning={isGameRunning}
                algorithm={selectedAlgorithm}
                setGameState={setGameState}
              />
            </div>
          )}

          {activeTab === 'training' && <TrainingPanel />}
          {activeTab === 'comparison' && <ComparisonPanel />}
          {activeTab === 'models' && <ModelManager />}
        </div>
      </div>
    </div>
  );
}

export default App;