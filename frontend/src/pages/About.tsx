import React from 'react';
import { Shield, Zap, Target, Users } from 'lucide-react';

const About = () => {
  const features = [
    {
      icon: Shield,
      title: 'Advanced Detection',
      description: 'State-of-the-art algorithms to identify hallucinations in AI-generated content with high precision.',
    },
    {
      icon: Zap,
      title: 'Real-time Analysis',
      description: 'Get instant feedback on text authenticity with detailed reasoning and confidence scores.',
    },
    {
      icon: Target,
      title: 'Batch Processing',
      description: 'Analyze multiple texts simultaneously with comprehensive reporting and visualization.',
    },
    {
      icon: Users,
      title: 'Enterprise Ready',
      description: 'Scalable solution designed for teams and organizations requiring reliable AI validation.',
    },
  ];

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="text-center mb-16">
        <div className="flex items-center justify-center space-x-4 mb-6">
          <div className="p-4 bg-primary-600 rounded-2xl">
            <Shield className="h-12 w-12 text-white" />
          </div>
          <div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
              DAC-NET
            </h1>
            <p className="text-xl text-primary-400 font-medium">Hallucination Validator</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-lg leading-relaxed max-w-3xl mx-auto">
          DAC-NET is a cutting-edge AI validation platform designed to detect and analyze 
          hallucinations in large language model outputs. Our advanced algorithms provide 
          real-time assessment of AI-generated content, helping users identify potential 
          inaccuracies and maintain the integrity of AI-assisted workflows.
        </p>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
        {features.map((feature, index) => (
          <div
            key={index}
            className="bg-gradient-card rounded-xl p-8 border border-dark-600 hover:border-primary-500/50 transition-all duration-300 group"
          >
            <div className="flex items-start space-x-4">
              <div className="p-3 bg-primary-600/20 rounded-lg group-hover:bg-primary-600/30 transition-colors duration-200">
                <feature.icon className="w-6 h-6 text-primary-400" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white mb-3">{feature.title}</h3>
                <p className="text-gray-300 leading-relaxed">{feature.description}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Mission Statement */}
      <div className="bg-gradient-card rounded-2xl p-8 border border-dark-600 text-center">
        <h2 className="text-2xl font-bold text-white mb-4">Our Mission</h2>
        <p className="text-gray-300 text-lg leading-relaxed max-w-2xl mx-auto">
          To empower organizations and individuals with the tools they need to verify 
          AI-generated content, ensuring accuracy, reliability, and trustworthiness in 
          an era of increasing AI integration across industries.
        </p>
      </div>
    </div>
  );
};

export default About;