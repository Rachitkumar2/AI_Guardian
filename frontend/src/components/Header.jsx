import React from 'react';
import { motion } from 'framer-motion';

const Header = () => {
  return (
    <motion.header 
      className="bg-white border-b border-slate-200 px-10 py-4 flex items-center justify-between shadow-sm sticky top-0 z-50"
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white">
          <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            <path d="M9 12l2 2 4-4" />
          </svg>
        </div>
        <div>
          <h1 className="text-xl font-semibold text-slate-900 tracking-tight">AI Guardian</h1>
          <p className="text-xs text-slate-500">Deepfake Audio Detection</p>
        </div>
      </div>
      
      <div className="flex items-center gap-5">
        <div className="bg-green-50 border border-green-200 px-3.5 py-1.5 rounded-full text-xs text-green-600 flex items-center gap-1.5 font-semibold">
          <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></span>
          Model Active
        </div>
      </div>
    </motion.header>
  );
};

export default Header;
