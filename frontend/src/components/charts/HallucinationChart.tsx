import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

interface HallucinationChartProps {
  percentage: number;
}

const HallucinationChart: React.FC<HallucinationChartProps> = ({ percentage }) => {
  const data = [
    { name: 'Hallucinated', value: percentage },
    { name: 'Accurate', value: 100 - percentage },
  ];

  const colors = ['#ef4444', '#22c55e'];

  return (
    <div className="flex flex-col items-center">
      <h3 className="text-xl font-semibold mb-6 text-white">Hallucination Analysis</h3>
      
      <div className="relative">
        <ResponsiveContainer width={280} height={280}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={120}
              startAngle={90}
              endAngle={450}
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={colors[index]}
                  stroke="none"
                />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        
        {/* Center Text */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="text-4xl font-bold text-white">{percentage}%</div>
            <div className="text-sm text-gray-400">Hallucinated</div>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex space-x-6 mt-6">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded-full bg-red-500"></div>
          <span className="text-sm text-gray-300">Hallucinated ({percentage}%)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded-full bg-green-500"></div>
          <span className="text-sm text-gray-300">Accurate ({100 - percentage}%)</span>
        </div>
      </div>
    </div>
  );
};

export default HallucinationChart;