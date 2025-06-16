import { useState, useEffect } from "react";
import { Line } from "react-chartjs-2";


const FitnessChart: React.FC = () => {
  const [fitnessData, setFitnessData] = useState<{ generation: number, fitness: number }[]>([]);

  useEffect(() => {
    fetch("http://localhost:5000/api/training/fitness_log")
      .then(res => res.json())
      .then(res => {
        if (res.success) {
          setFitnessData(res.data);
        }
      });
  }, []);

  const data = {
    labels: fitnessData.map(d => `Gen ${d.generation}`),
    datasets: [
      {
        label: 'Best Fitness',
        data: fitnessData.map(d => d.fitness),
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.3,
        pointRadius: 3
      }
    ]
  };

  const options = {
    responsive: true,
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Fitness Score'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Generation'
        }
      }
    }
  };

  return <Line data={data} options={options} />;
};

export default FitnessChart