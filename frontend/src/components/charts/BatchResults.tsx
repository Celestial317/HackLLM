import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';

interface BatchResultsProps {
  data: Array<{
    type: string;
    percentage: number;
    count: number;
  }>;
}

const BatchResults: React.FC<BatchResultsProps> = ({ data }) => {
  return (
    <div className="bg-gradient-card rounded-2xl p-8 border border-dark-600">
      <h2 className="text-2xl font-bold text-white mb-8 text-center">
        Hallucination Types Breakdown
      </h2>
      
      {/* Chart */}
      <div className="mb-8">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
            <XAxis 
              dataKey="type" 
              stroke="#94a3b8"
              fontSize={12}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis stroke="#94a3b8" fontSize={12} />
            <Bar 
              dataKey="percentage" 
              fill="#3b82f6"
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed List */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white mb-4">Detailed Breakdown</h3>
        {data.map((item, index) => (
          <div
            key={index}
            className="flex items-center justify-between p-4 bg-dark-800 rounded-lg border border-dark-600 hover:border-dark-500 transition-colors duration-200"
          >
            <div className="flex items-center space-x-4">
              <div className="w-4 h-4 rounded-full bg-primary-500"></div>
              <span className="text-white font-medium">{item.type}</span>
            </div>
            <div className="text-right">
              <div className="text-white font-semibold">{item.percentage}%</div>
              <div className="text-gray-400 text-sm">{item.count} instances</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BatchResults;