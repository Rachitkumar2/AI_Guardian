import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Header from './components/Header';
import UploadZone from './components/UploadZone';
import ResultDisplay from './components/ResultDisplay';
import FeatureCards from './components/FeatureCards';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);

  const handleFileSelect = useCallback((selectedFile) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setAnalyzing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || data.details || 'Detection failed');
      }

      await new Promise(resolve => setTimeout(resolve, 500));
      
      setResult(data.result);
    } catch (err) {
      setError(err.message || 'An error occurred during analysis');
    } finally {
      setLoading(false);
      setAnalyzing(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setLoading(false);
    setAnalyzing(false);
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-slate-50 to-slate-200">
      <Header />
      
      <main className="max-w-2xl mx-auto my-12 px-6 flex-1 w-full">
        <motion.div 
          className="bg-white border border-slate-200 rounded-2xl p-12 shadow-md hover:shadow-lg transition-shadow"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-2xl font-semibold text-center text-slate-900 mb-2">
            Audio Analysis Center
          </h2>
          <p className="text-center text-slate-500 mb-8 text-sm">
            Upload an audio file to analyze its authenticity using advanced AI detection
          </p>

          <AnimatePresence mode="wait">
            {!result && !analyzing && (
              <motion.div
                key="upload"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <UploadZone 
                  file={file} 
                  onFileSelect={handleFileSelect}
                  onRemove={() => setFile(null)}
                />
                
                {file && (
                  <motion.button
                    className="flex items-center justify-center gap-2.5 w-full max-w-[280px] mx-auto mt-6 py-3.5 px-7 bg-blue-600 hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed rounded-lg text-white text-sm font-semibold cursor-pointer transition-all shadow-sm hover:shadow-md hover:-translate-y-0.5"
                    onClick={handleAnalyze}
                    disabled={loading}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {loading ? (
                      <>
                        <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M9 12l2 2 4-4" />
                          <circle cx="12" cy="12" r="10" />
                        </svg>
                        Analyze Audio
                      </>
                    )}
                  </motion.button>
                )}
              </motion.div>
            )}

            {analyzing && (
              <motion.div
                key="analyzing"
                className="text-center py-12"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="flex justify-center items-center gap-1 h-12">
                  {[...Array(12)].map((_, i) => (
                    <div 
                      key={i} 
                      className="w-1 bg-blue-600 rounded-sm animate-waveform" 
                      style={{ animationDelay: `${i * 0.1}s` }} 
                    />
                  ))}
                </div>
                <p className="text-slate-900 text-base font-semibold mt-6">Analyzing audio patterns...</p>
                <p className="text-slate-500 text-sm mt-1.5">Running deepfake detection algorithms</p>
              </motion.div>
            )}

            {result && !analyzing && (
              <ResultDisplay 
                result={result} 
                fileName={file?.name}
                onReset={handleReset}
              />
            )}
          </AnimatePresence>

          {error && (
            <motion.div 
              className="flex items-center gap-2.5 bg-red-50 border border-red-200 py-3.5 px-5 rounded-lg text-red-600 mt-5 text-sm"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <svg className="w-5 h-5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              {error}
            </motion.div>
          )}
        </motion.div>

        <FeatureCards />
      </main>

      <footer className="text-center py-6 text-slate-500 text-sm border-t border-slate-200 bg-white">
        <p>© 2026 AI Guardian. Protecting authenticity in the age of AI.</p>
      </footer>
    </div>
  );
}

export default App;
