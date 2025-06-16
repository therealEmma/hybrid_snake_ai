import React, { useState, useEffect } from 'react';
import { Download, Upload, Trash2, CheckCircle, Clock, Award } from 'lucide-react';

interface Model {
  filename: string;
  timestamp: string;
  best_score: number | string;
  generation: number | string;
}

const ModelManager: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/models/list');
      const data = await response.json();
      setModels(data.models || []);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadModel = async (filename: string) => {
    try {
      const response = await fetch(`http://localhost:5000/api/models/load/${filename}`, {
        method: 'POST'
      });
      
      const data = await response.json();
      if (data.success) {
        setSelectedModel(filename);
        // Show success message
        setTimeout(() => setSelectedModel(null), 3000);
      }
    } catch (error) {
      console.error('Failed to load model:', error);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    if (timestamp === 'Unknown') return 'Unknown';
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">Model Manager</h2>
        <p className="text-slate-300">
          Manage your trained AI models and load the best performers
        </p>
      </div>

      {/* Upload Section */}
      <div className="bg-slate-700/50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Upload size={20} />
          Upload Model
        </h3>
        <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center">
          <Upload size={48} className="mx-auto text-slate-400 mb-4" />
          <p className="text-slate-300 mb-2">Drag and drop a model file here</p>
          <p className="text-slate-500 text-sm">or click to browse</p>
          <input
            type="file"
            accept=".json"
            className="hidden"
            onChange={(e) => {
              // Handle file upload
              console.log('File selected:', e.target.files?.[0]);
            }}
          />
        </div>
      </div>

      {/* Models List */}
      <div className="bg-slate-700/50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Download size={20} />
          Saved Models
        </h3>

        {models.length === 0 ? (
          <div className="text-center py-8">
            <Award size={48} className="mx-auto text-slate-400 mb-4" />
            <p className="text-slate-300">No models found</p>
            <p className="text-slate-500 text-sm">Train an AI agent to create your first model</p>
          </div>
        ) : (
          <div className="space-y-3">
            {models.map((model, index) => (
              <div
                key={index}
                className="bg-slate-600/50 rounded-lg p-4 flex items-center justify-between"
              >
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h4 className="text-white font-medium">{model.filename}</h4>
                    {selectedModel === model.filename && (
                      <div className="flex items-center gap-1 text-green-400 text-sm">
                        <CheckCircle size={16} />
                        Loaded
                      </div>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div className="flex items-center gap-2 text-slate-300">
                      <Clock size={14} />
                      <span>{formatTimestamp(model.timestamp)}</span>
                    </div>
                    
                    <div className="flex items-center gap-2 text-slate-300">
                      <Award size={14} />
                      <span>Score: {model.best_score}</span>
                    </div>
                    
                    <div className="flex items-center gap-2 text-slate-300">
                      <span>Gen: {model.generation}</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => loadModel(model.filename)}
                    disabled={selectedModel === model.filename}
                    className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg font-medium transition-colors"
                  >
                    Load
                  </button>
                  
                  <button
                    onClick={() => {
                      // Handle download
                      console.log('Download model:', model.filename);
                    }}
                    className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                  >
                    <Download size={16} />
                  </button>
                  
                  <button
                    onClick={() => {
                      // Handle delete
                      console.log('Delete model:', model.filename);
                    }}
                    className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Model Performance Summary */}
      {models.length > 0 && (
        <div className="bg-slate-700/50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Performance Summary</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400 mb-1">
                {models.length}
              </div>
              <div className="text-slate-300 text-sm">Total Models</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400 mb-1">
                {Math.max(...models.map(m => typeof m.best_score === 'number' ? m.best_score : 0))}
              </div>
              <div className="text-slate-300 text-sm">Best Score</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-400 mb-1">
                {Math.max(...models.map(m => typeof m.generation === 'number' ? m.generation : 0))}
              </div>
              <div className="text-slate-300 text-sm">Max Generation</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelManager;