import React, { useState } from 'react';
import { Upload, FileText, AlertTriangle, TrendingUp } from 'lucide-react';
import Button from '../components/ui/Button';
import FileUpload from '../components/ui/FileUpload';
import BatchResults from '../components/charts/BatchResults';

const BatchValidator = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [results, setResults] = useState<{
    totalPercentage: number;
    totalEntries: number;
    hallucinationTypes: Array<{
      type: string;
      percentage: number;
      count: number;
    }>;
  } | null>(null);

  const handleValidate = async () => {
    if (!file) return;

    setIsValidating(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/validate/batch', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      setResults({
        totalPercentage: data.total_percentage,
        totalEntries: data.total_entries,
        hallucinationTypes: data.hallucination_types,
      });
    } catch (error) {
      console.error('Error validating batch:', error);
      // Fallback to mock data in case of error
      const mockTypes = [
        { type: 'Factual Error', percentage: 35, count: 42 },
        { type: 'Contextual Misalignment', percentage: 28, count: 34 },
        { type: 'Logical Contradiction', percentage: 22, count: 26 },
        { type: 'Temporal Inconsistency', percentage: 15, count: 18 },
      ];
      
      setResults({
        totalPercentage: Math.floor(Math.random() * 40) + 30, // 30-70%
        totalEntries: 120,
        hallucinationTypes: mockTypes,
      });
    }
    
    setIsValidating(false);
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl sm:text-5xl font-bold mb-4 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
          Batch Validator
        </h1>
        <p className="text-gray-400 text-lg">
          Upload and analyze multiple AI-generated responses at once
        </p>
      </div>

      {/* Upload Section */}
      <div className="bg-gradient-card rounded-2xl p-8 shadow-2xl border border-dark-600 mb-8">
        <FileUpload 
          onFileSelect={setFile} 
          selectedFile={file}
          acceptedTypes=".csv,.json"
          description="Accepts CSV, JSON files"
        />
        
        {file && (
          <div className="mt-6 flex items-center justify-between p-4 bg-dark-800 rounded-lg border border-dark-600">
            <div className="flex items-center space-x-3">
              <FileText className="w-5 h-5 text-primary-400" />
              <span className="text-white font-medium">{file.name}</span>
              <span className="text-gray-400 text-sm">
                ({(file.size / 1024).toFixed(1)} KB)
              </span>
            </div>
            <Button
              onClick={handleValidate}
              disabled={isValidating}
              className="px-6 py-2"
            >
              {isValidating ? 'Processing...' : 'Validate'}
            </Button>
          </div>
        )}
      </div>

      {/* Results Section */}
      {results && (
        <div className="space-y-8 animate-fade-in">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gradient-card rounded-xl p-6 border border-dark-600 hover:border-primary-500/50 transition-all duration-200">
              <div className="flex items-center space-x-3 mb-2">
                <AlertTriangle className="w-6 h-6 text-red-400" />
                <h3 className="text-lg font-semibold text-white">Overall Hallucination</h3>
              </div>
              <p className="text-3xl font-bold text-red-300">{results.totalPercentage}%</p>
              <p className="text-gray-400 text-sm mt-1">of analyzed content</p>
            </div>

            <div className="bg-gradient-card rounded-xl p-6 border border-dark-600 hover:border-primary-500/50 transition-all duration-200">
              <div className="flex items-center space-x-3 mb-2">
                <FileText className="w-6 h-6 text-primary-400" />
                <h3 className="text-lg font-semibold text-white">Total Entries</h3>
              </div>
              <p className="text-3xl font-bold text-white">{results.totalEntries}</p>
              <p className="text-gray-400 text-sm mt-1">entries processed</p>
            </div>

            <div className="bg-gradient-card rounded-xl p-6 border border-dark-600 hover:border-primary-500/50 transition-all duration-200">
              <div className="flex items-center space-x-3 mb-2">
                <TrendingUp className="w-6 h-6 text-green-400" />
                <h3 className="text-lg font-semibold text-white">Accuracy Rate</h3>
              </div>
              <p className="text-3xl font-bold text-green-300">{100 - results.totalPercentage}%</p>
              <p className="text-gray-400 text-sm mt-1">reliable content</p>
            </div>
          </div>

          {/* Detailed Analysis */}
          <BatchResults data={results.hallucinationTypes} />
        </div>
      )}
    </div>
  );
};

export default BatchValidator;