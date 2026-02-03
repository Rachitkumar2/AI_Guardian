import React from 'react';
import { motion } from 'framer-motion';

const features = [
  {
    icon: (
      <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      </svg>
    ),
    title: 'AI-Powered Detection',
    description: 'Advanced neural networks analyze audio patterns to identify AI-generated content.'
  },
  {
    icon: (
      <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <polyline points="12,6 12,12 16,14" />
      </svg>
    ),
    title: 'Real-Time Analysis',
    description: 'Get instant results with our optimized processing pipeline.'
  },
  {
    icon: (
      <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
        <path d="M7 11V7a5 5 0 0110 0v4" />
      </svg>
    ),
    title: 'Privacy First',
    description: 'Your audio files are processed securely and deleted immediately after analysis.'
  }
];

const FeatureCards = () => {
  return (
    <div className="mt-12">
      <h3 className="text-center text-lg font-semibold text-slate-900 mb-6">How It Works</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {features.map((feature, index) => (
          <motion.div
            key={index}
            className="bg-white border border-slate-200 rounded-xl p-6 text-center hover:border-slate-300 hover:shadow-md transition-all"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 + index * 0.1 }}
            whileHover={{ y: -3 }}
          >
            <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center mx-auto mb-4 text-blue-600">
              {feature.icon}
            </div>
            <h4 className="text-sm font-semibold text-slate-900 mb-2">{feature.title}</h4>
            <p className="text-xs text-slate-500 leading-relaxed">{feature.description}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default FeatureCards;
