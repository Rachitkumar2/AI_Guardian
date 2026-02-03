import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';

const UploadZone = ({ file, onFileSelect, onRemove }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024
  });

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="relative">
      <AnimatePresence mode="wait">
        {!file ? (
          <motion.div
            key="dropzone"
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
              isDragActive 
                ? 'border-blue-500 bg-blue-50' 
                : 'border-slate-300 bg-slate-50 hover:border-blue-400 hover:bg-blue-50/50'
            }`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            whileHover={{ scale: 1.01 }}
          >
            <input {...getInputProps()} />
            <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4 text-blue-600">
              <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                <polyline points="17,8 12,3 7,8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <h3 className="text-base font-semibold text-slate-900 mb-1.5">
              {isDragActive ? 'Drop your audio file here' : 'Drag & Drop Audio File'}
            </h3>
            <p className="text-slate-500 text-sm mb-4">or click to browse from your computer</p>
            <div className="inline-block text-xs text-slate-400 bg-white px-3.5 py-1.5 rounded-full border border-slate-200">
              <span className="text-blue-600 font-medium mr-1">Supported:</span> 
              WAV, MP3, FLAC, OGG, M4A, AAC
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="file-preview"
            className="flex items-center gap-3.5 bg-slate-50 border border-slate-200 rounded-xl p-4"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
          >
            <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center text-blue-600 flex-shrink-0">
              <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M9 18V5l12-2v13" />
                <circle cx="6" cy="18" r="3" />
                <circle cx="18" cy="16" r="3" />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <h4 className="text-sm font-semibold text-slate-900 truncate">{file.name}</h4>
              <p className="text-xs text-slate-500">{formatFileSize(file.size)}</p>
            </div>
            <button 
              className="w-8 h-8 bg-white border border-slate-200 rounded-md flex items-center justify-center text-slate-400 hover:bg-red-50 hover:border-red-200 hover:text-red-500 transition-colors flex-shrink-0"
              onClick={(e) => { e.stopPropagation(); onRemove(); }}
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default UploadZone;
