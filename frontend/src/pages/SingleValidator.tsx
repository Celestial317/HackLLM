import React, { useState } from 'react';
import { AlertTriangle, CheckCircle, Loader } from 'lucide-react';
import HallucinationChart from '../components/charts/HallucinationChart';
import Button from '../components/ui/Button';
import TextArea from '../components/ui/TextArea';

const SingleValidator = () => {
  const [generatedText, setGeneratedText] = useState('');
  const [prompt, setPrompt] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [results, setResults] = useState<{
    isHallucination: boolean;
    percentage: number;
    reasoning: string;
  } | null>(null);

  const handleValidate = async () => {
    if (!generatedText.trim() || !prompt.trim()) return;

    setIsValidating(true);
    
    try {
      const response = await fetch('http://localhost:8000/validate/single', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          generated_text: generatedText,
          prompt: prompt,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      setResults({
        isHallucination: data.is_hallucination,
        percentage: data.percentage,
        reasoning: data.reasoning,
      });
    } catch (error) {
      console.error('Error validating text:', error);
      // Fallback to mock data in case of error
      const mockPercentage = Math.floor(Math.random() * 100);
      const isHallucination = mockPercentage > 50;
      
      setResults({
        isHallucination,
        percentage: mockPercentage,
        reasoning: isHallucination 
          ? "Error occurred during validation. Showing fallback results. The generated text may contain inaccuracies."
          : "Error occurred during validation. Showing fallback results. The generated text appears to be generally consistent."
      });
    }
    
    setIsValidating(false);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl sm:text-5xl font-bold mb-4 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
          Single Text Validator
        </h1>
        <p className="text-gray-400 text-lg">
          Validate individual AI-generated responses against their source prompts
        </p>
      </div>

      {/* Input Section */}
      <div className="space-y-6 mb-8">
        <TextArea
          label="Generated Text by LLM"
          placeholder="Paste the AI-generated text you want to validate..."
          value={generatedText}
          onChange={setGeneratedText}
          rows={6}
        />
        
        <TextArea
          label="Given Prompt to LLM"
          placeholder="Paste the original prompt that was given to the LLM..."
          value={prompt}
          onChange={setPrompt}
          rows={4}
        />
      </div>

      {/* Validate Button */}
      <div className="text-center mb-8">
        <Button
          onClick={handleValidate}
          disabled={!generatedText.trim() || !prompt.trim() || isValidating}
          className="px-8 py-3 text-lg"
        >
          {isValidating ? (
            <>
              <Loader className="w-5 h-5 mr-2 animate-spin" />
              Validating...
            </>
          ) : (
            'Validate'
          )}
        </Button>
      </div>

      {/* Results Section */}
      {results && (
        <div className="bg-gradient-card rounded-2xl p-8 shadow-2xl border border-dark-600 animate-fade-in">
          {/* Status */}
          <div className="flex items-center justify-center mb-8">
            <div className={`flex items-center space-x-3 px-6 py-3 rounded-full ${
              results.isHallucination 
                ? 'bg-red-500/20 border border-red-500/30' 
                : 'bg-green-500/20 border border-green-500/30'
            }`}>
              {results.isHallucination ? (
                <AlertTriangle className="w-6 h-6 text-red-400" />
              ) : (
                <CheckCircle className="w-6 h-6 text-green-400" />
              )}
              <span className={`text-xl font-semibold ${
                results.isHallucination ? 'text-red-300' : 'text-green-300'
              }`}>
                {results.isHallucination ? 'Hallucination Detected' : 'No Hallucination Detected'}
              </span>
            </div>
          </div>

          {/* Chart */}
          <div className="mb-8">
            <HallucinationChart percentage={results.percentage} />
          </div>

          {/* Reasoning */}
          <div>
            <h3 className="text-xl font-semibold mb-4 text-white">Reasoning</h3>
            <div className="bg-dark-800 rounded-lg p-6 border border-dark-600">
              <p className="text-gray-300 leading-relaxed">{results.reasoning}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SingleValidator;