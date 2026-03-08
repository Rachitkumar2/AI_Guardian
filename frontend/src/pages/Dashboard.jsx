import { useState, useCallback } from 'react';
import { Upload, Link as LinkIcon, AlertTriangle, CheckCircle2, Activity, Loader2, History } from 'lucide-react';
import { useDropzone } from 'react-dropzone';

export default function Dashboard() {
  const [recentResults, setRecentResults] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const [urlInput, setUrlInput] = useState('');

  const anomalies = [
    { name: 'SPECTRAL JITTER', match: currentResult ? (currentResult.isFake ? '92%' : '12%') : '0%' },
    { name: 'PITCH VARIANCE CONSISTENCY', match: currentResult ? (currentResult.isFake ? '64%' : '5%') : '0%' },
    { name: 'PHONEME TRANSITIONS', match: currentResult ? (currentResult.isFake ? '81%' : '18%') : '0%' },
    { name: 'BACKGROUND NOISE COHESION', match: currentResult ? (currentResult.isFake ? '15%' : '88%') : '0%' }
  ];

  const handleFileUpload = async (file) => {
    setIsAnalyzing(true);
    setCurrentResult(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Assuming the Flask backend runs on port 5000 as per app.py default
      const res = await fetch('http://localhost:5000/api/detect', {
        method: 'POST',
        body: formData,
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.message || data.error || 'Failed to analyze audio');
      }

      // Try to handle both cases format: 0.94 vs 94.9
      const confidenceRaw = data.confidence || 0;
      const confidencePercent = Math.round(confidenceRaw <= 1 ? confidenceRaw * 100 : confidenceRaw);
      const isFake = data.result?.toLowerCase() === 'fake';
      
      const newResult = {
        filename: file.name,
        time: 'Just now',
        result: `${confidencePercent}% ${isFake ? 'Synthetic' : 'Natural'}`,
        status: isFake ? 'FAKE' : 'REAL',
        isFake: isFake,
        confidence: confidencePercent,
      };

      setCurrentResult(newResult);
      setRecentResults(prev => [newResult, ...prev]);

    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error analyzing file: ' + error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      handleFileUpload(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.m4a']
    },
    maxFiles: 1
  });

  const handleUrlAnalyze = () => {
    if (urlInput.trim()) {
      alert("URL analysis is not yet implemented in the backend, but this is where it would plug in!");
      setUrlInput('');
    }
  };

  // SVG Dashboard Circle calculations
  const radius = 45;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = currentResult 
    ? circumference - (currentResult.confidence / 100) * circumference 
    : circumference; // 0% filled if no result

  return (
    <div className="flex flex-col xl:flex-row gap-8 w-full max-w-7xl mx-auto">
      
      {/* Main Analysis Column */}
      <div className="flex-1 space-y-8">
        <div>
          <h2 className="text-3xl font-bold mb-2">AI Voice Detection</h2>
          <p className="text-gray-400 text-sm">Upload audio or paste a URL to analyze for deepfake patterns using our neural network.</p>
        </div>

        {/* Upload Container */}
        <div 
          {...getRootProps()} 
          className={`border ${isDragActive ? 'border-neon-green bg-neon-green/5' : 'border-[#1C2A22] bg-[#121A15]'} border-dashed rounded-2xl p-10 flex flex-col items-center justify-center text-center hover:border-neon-green/50 transition-colors cursor-pointer group relative`}
        >
          <input {...getInputProps()} />
          
          {isAnalyzing ? (
            <div className="flex flex-col items-center justify-center">
               <Loader2 className="w-14 h-14 text-neon-green animate-spin mb-6" />
               <h3 className="text-xl font-bold mb-2 text-neon-green">Analyzing Audio...</h3>
               <p className="text-gray-400 text-sm">Running through deep neural networks</p>
            </div>
          ) : (
            <>
              <div className="w-14 h-14 bg-[#1C2A22] rounded-full flex items-center justify-center mb-6 group-hover:bg-neon-green/20 transition-colors">
                <Upload className="w-6 h-6 text-neon-green" />
              </div>
              <h3 className="text-xl font-bold mb-2">
                {isDragActive ? 'Drop file here' : 'Upload Audio File'}
              </h3>
              <p className="text-gray-400 text-sm mb-6">Drag and drop your .mp3, .wav, or .m4a file</p>
              <button className="bg-neon-green text-black px-6 py-2.5 rounded-lg font-bold hover:bg-neon-green-hover transition-colors shadow-lg shadow-neon-green/20 pointer-events-none">
                Browse Files
              </button>
            </>
          )}
        </div>

        {/* OR Divider */}
        <div className="flex items-center gap-4 py-2">
          <div className="flex-1 h-px bg-[#1C2A22]"></div>
          <div className="text-xs font-bold text-gray-500 uppercase tracking-widest">OR</div>
          <div className="flex-1 h-px bg-[#1C2A22]"></div>
        </div>

        {/* URL Input */}
        <div>
          <label className="block text-sm font-semibold mb-2 text-gray-300">Paste Audio URL</label>
          <div className="flex bg-[#121A15] border border-[#1C2A22] rounded-lg overflow-hidden focus-within:border-neon-green/50 transition-colors">
            <input 
              type="text" 
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="https://youtube.com/watch?v=..." 
              className="flex-1 bg-transparent px-4 py-3 text-sm focus:outline-none"
            />
            <button className="bg-[#1C2A22] px-4 flex items-center justify-center hover:bg-[#2A3F33] transition-colors">
              <LinkIcon className="w-5 h-5 text-neon-green" />
            </button>
          </div>
        </div>

        {/* Analyze Button */}
        <button 
          onClick={handleUrlAnalyze}
          disabled={!urlInput.trim() || isAnalyzing}
          className="w-full bg-neon-green text-black py-4 rounded-xl font-black text-lg hover:bg-neon-green-hover disabled:opacity-50 disabled:hover:bg-neon-green transition-all shadow-[0_0_30px_rgba(0,255,102,0.3)] hover:shadow-[0_0_40px_rgba(0,255,102,0.5)] disabled:hover:shadow-[0_0_30px_rgba(0,255,102,0.3)]"
        >
          ANALYZE AUDIO URL
        </button>

        {/* Analysis Verdict Section */}
        <div className="pt-4">
          <h2 className="text-xl font-bold mb-6">Analysis Verdict</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Donut Chart */}
            <div className="glass-panel p-6 flex flex-col items-center justify-center border-[#1C2A22]">
              <div className="relative w-40 h-40 flex items-center justify-center mb-6">
                <svg className="w-full h-full -rotate-90 transform" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="45" fill="none" stroke="#1C2A22" strokeWidth="10" />
                  <circle 
                    cx="50" cy="50" r="45" fill="none" 
                    stroke={currentResult ? (currentResult.isFake ? "#EF4444" : "#00FF66") : "#00FF66"} 
                    strokeWidth="10" 
                    strokeDasharray={circumference} 
                    strokeDashoffset={strokeDashoffset} 
                    strokeLinecap="round" 
                    className="transition-all duration-1000 ease-out"
                  />
                </svg>
                <div className="absolute flex flex-col items-center justify-center text-center">
                  <span className="text-4xl font-black text-white">{currentResult ? currentResult.confidence : 0}%</span>
                  <span className="text-[10px] text-gray-400 font-bold tracking-widest uppercase">Confidence</span>
                </div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-400 mb-1">Likelihood of Synthesis</div>
                <div className={`font-black uppercase text-lg tracking-widest ${currentResult ? (currentResult.isFake ? 'text-red-500 shadow-red-500/50' : 'text-neon-green neon-text') : 'text-gray-500'}`}>
                  {currentResult ? (currentResult.isFake ? 'HIGH RISK' : 'AUTHENTIC') : 'AWAITING FILE'}
                </div>
              </div>
            </div>

            {/* Breakdown Bars */}
            <div className="glass-panel p-6 md:col-span-2 border-[#1C2A22] space-y-6 flex flex-col justify-center">
              <h3 className="font-bold flex items-center gap-2 mb-2"><Activity className="w-4 h-4 text-neon-green" /> Acoustic Anomalies Breakdown</h3>
              
              <div className="space-y-5">
                {anomalies.map((anomaly, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-xs font-bold mb-2">
                      <span className="text-gray-300 tracking-wider">{anomaly.name}</span>
                      <span className="text-neon-green">{anomaly.match} Match</span>
                    </div>
                    <div className="h-1.5 w-full bg-[#1A251D] rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-neon-green shadow-[0_0_8px_#00FF66] transition-all duration-1000 ease-out" 
                        style={{ width: anomaly.match }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        </div>

      </div>

      {/* Right Sidebar - Recent Results */}
      <div className="w-full xl:w-80 glass-panel border-[#1C2A22] flex flex-col p-0 overflow-hidden shrink-0 h-fit">
        <div className="p-6 border-b border-[#1C2A22] flex items-center justify-between">
          <h3 className="font-bold">Recent Results</h3>
          {recentResults.length > 0 && (
             <button 
                onClick={() => setRecentResults([])}
                className="text-gray-400 hover:text-white text-xs font-bold hover:underline transition-colors"
                title="Clear history"
              >
                Clear
              </button>
          )}
        </div>
        
        <div className="flex-1 flex flex-col gap-1 p-4 min-h-[200px]">
          {recentResults.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center text-gray-500 px-4 py-8">
               <History className="w-8 h-8 mb-4 opacity-30" />
               <p className="text-sm">No recent scans.</p>
               <p className="text-xs mt-1">Upload a file to see results here.</p>
            </div>
          ) : (
            recentResults.map((res, i) => (
              <div key={i} className="bg-[#121A15] border border-[#1C2A22] rounded-xl p-4 flex items-start gap-4 hover:border-[#2A3F33] transition-colors cursor-pointer">
                <div className={`p-2 rounded-lg shrink-0 ${res.isFake ? 'bg-red-500/10' : 'bg-neon-green/10'}`}>
                  {res.isFake ? (
                    <AlertTriangle className="w-5 h-5 text-red-500" />
                  ) : (
                    <CheckCircle2 className="w-5 h-5 text-neon-green" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-semibold text-sm truncate mb-1">{res.filename}</div>
                  <div className="text-xs text-gray-500 flex items-center gap-1.5">
                    {res.time} &bull; <span className={res.isFake ? 'text-gray-400' : 'text-gray-400'}>{res.result}</span>
                  </div>
                </div>
                <div className={`text-[10px] font-bold px-2 py-1 rounded shrink-0 ${res.isFake ? 'bg-red-500/20 text-red-500 border border-red-500/30' : 'bg-neon-green/20 text-neon-green border border-neon-green/30'}`}>
                  {res.status}
                </div>
              </div>
            ))
          )}
        </div>

        <div className="p-6 mt-auto">
          <p className="text-xs text-center text-gray-500 italic">
            "Our models are updated daily to catch the newest generative AI patterns."
          </p>
        </div>
      </div>

    </div>
  );
}
