import React from 'react';
import { motion } from 'framer-motion';

const ResultDisplay = ({ result, fileName, onReset }) => {
  const isReal = result === 'Real';

  return (
    <motion.div
      className="text-center py-8 px-4"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, type: 'spring' }}
    >
      <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-5 animate-icon-pop ${
        isReal ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
      }`}>
        {isReal ? (
          <svg className="w-10 h-10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
            <polyline points="22,4 12,14.01 9,11.01" />
          </svg>
        ) : (
          <svg className="w-10 h-10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        )}
      </div>

      <h3 className={`text-2xl font-semibold mb-2 ${isReal ? 'text-green-600' : 'text-red-600'}`}>
        {isReal ? 'Authentic Audio Detected' : 'Deepfake Audio Detected'}
      </h3>

      <p className="text-slate-500 text-sm max-w-sm mx-auto mb-6 leading-relaxed">
        {isReal
          ? 'Our AI analysis indicates this audio is likely genuine human speech.'
          : 'Our AI analysis indicates this audio may be artificially generated or manipulated.'}
      </p>

      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 mb-6">
        <div className="flex justify-between items-center py-2.5 border-b border-slate-200">
          <span className="text-slate-500 text-xs">File Analyzed</span>
          <span className="text-slate-900 text-xs font-semibold truncate max-w-[180px]">{fileName}</span>
        </div>
        <div className="flex justify-between items-center py-2.5 border-b border-slate-200">
          <span className="text-slate-500 text-xs">Detection Result</span>
          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
            isReal ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
          }`}>
            {result}
          </span>
        </div>
        <div className="flex justify-between items-center py-2.5">
          <span className="text-slate-500 text-xs">Analysis Method</span>
          <span className="text-slate-900 text-xs font-semibold">MFCC Neural Network</span>
        </div>
      </div>

      <div className="flex justify-center">
        <motion.button
          className="flex items-center gap-2 px-6 py-3 bg-white border border-slate-200 rounded-lg text-slate-700 text-sm font-semibold hover:bg-slate-50 hover:border-slate-300 transition-colors"
          onClick={onReset}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 12a9 9 0 11-6.219-8.56" />
            <polyline points="21,3 21,9 15,9" />
          </svg>
          Analyze Another File
        </motion.button>
      </div>

      {!isReal && (
        <div className="flex items-center justify-center gap-2 mt-5 px-4 py-3 bg-amber-50 border border-amber-200 rounded-lg text-amber-700 text-xs">
          <svg className="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <p>Exercise caution when sharing or relying on this audio content.</p>
        </div>
      )}
    </motion.div>
  );
};

export default ResultDisplay;
