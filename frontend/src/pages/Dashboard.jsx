import { useState, useCallback, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Upload, Link as LinkIcon, AlertTriangle, CheckCircle2, Activity, Loader2, History, ShieldCheck, ShieldAlert } from 'lucide-react';
import { useDropzone } from 'react-dropzone';

const MAX_FREE_SCANS = 2;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL
  || (window.location.hostname.includes('vercel.app') ? 'https://ai-guardian-uzj8.onrender.com' : '');

const buildApiUrl = (path) => `${API_BASE_URL}${path}`;

async function parseApiResponse(res) {
  const rawText = await res.text();
  let data = null;

  if (rawText) {
    try {
      data = JSON.parse(rawText);
    } catch {
      data = null;
    }
  }

  return { data, rawText };
}

function getFriendlyApiError(res, data, fallback) {
  if (data?.message || data?.error) {
    return data.message || data.error;
  }

  if (res.status >= 500) {
    return 'Server is temporarily unavailable. Please try again in a few seconds.';
  }

  if (res.status === 0) {
    return 'Network error. Please check your connection and try again.';
  }

  return fallback;
}

export default function Dashboard() {
  const [recentResults, setRecentResults] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const [urlInput, setUrlInput] = useState('');
  const [uploadError, setUploadError] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(Boolean(localStorage.getItem('token')));
  const [guestScansUsed, setGuestScansUsed] = useState(0);

  useEffect(() => {
    const handleAuthChange = () => {
      const loggedIn = Boolean(localStorage.getItem('token'));
      setIsAuthenticated(loggedIn);

      if (loggedIn) {
        setUploadError('');
        setGuestScansUsed(0);
      }
    };

    handleAuthChange();
    window.addEventListener('authChange', handleAuthChange);

    return () => window.removeEventListener('authChange', handleAuthChange);
  }, []);

  useEffect(() => {
    const syncFreeScanStatus = async () => {
      if (isAuthenticated) return;

      try {
        const headers = {};
        const token = localStorage.getItem('token');
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        }

        const res = await fetch(buildApiUrl('/api/free-scan-status'), {
          method: 'GET',
          credentials: 'include',
          headers,
        });

        if (!res.ok) return;

        const { data } = await parseApiResponse(res);
        if (!data) return;
        setGuestScansUsed(data?.scans_used ?? 0);

        if (data?.is_locked) {
          setUploadError('You have reached your limit of 2 free scans. Please log in to continue.');
        }
      } catch {
        // Ignore status fetch failures and rely on detect endpoint enforcement.
      }
    };

    syncFreeScanStatus();
  }, [isAuthenticated]);

  // Common logic to process the result from the backend
  const handleAnalysisResult = (data, filename) => {
    const confidenceRaw = data.confidence || 0;
    const confidencePercent = Math.round(confidenceRaw <= 1 ? confidenceRaw * 100 : confidenceRaw);
    const isFake = data.result?.toLowerCase() === 'fake';
    const isUncertain = data.result?.toLowerCase() === 'uncertain';

    const newResult = {
      filename: filename,
      time: 'Just now',
      result: `${confidencePercent}% ${isFake ? 'Synthetic' : isUncertain ? 'Uncertain' : 'Natural'}`,
      status: isFake ? 'FAKE' : isUncertain ? 'UNCERTAIN' : 'REAL',
      isFake: isFake,
      isUncertain: isUncertain,
      confidence: confidencePercent,
      confidenceLevel: data.confidence_level || 'Unknown',
      signals: data.signals || [],
    };

    setCurrentResult(newResult);
    setRecentResults(prev => [newResult, ...prev]);
  };

  const handleFileUpload = async (file) => {
    setIsAnalyzing(true);
    setCurrentResult(null);
    setUploadError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const headers = {};
      const token = localStorage.getItem('token');
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const res = await fetch(buildApiUrl('/api/detect'), {
        method: 'POST',
        credentials: 'include',
        headers,
        body: formData,
      });

      const { data } = await parseApiResponse(res);

      if (!res.ok) {
        if (data?.error === 'limit_reached') {
          setGuestScansUsed(data?.scans_used ?? MAX_FREE_SCANS);
        }
        setUploadError(getFriendlyApiError(res, data, 'Failed to analyze audio'));
        return;
      }

      if (!data) {
        setUploadError('Unexpected server response. Please try again.');
        return;
      }

      if (!token) {
        setGuestScansUsed(data?.scans_used ?? 0);
      }

      handleAnalysisResult(data, file.name);

    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadError(error.message || 'An unexpected error occurred. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const onDrop = useCallback((acceptedFiles, fileRejections) => {
    if (fileRejections.length > 0) {
      const error = fileRejections[0].errors[0];
      if (error.code === 'file-invalid-type') {
        setUploadError('Invalid file type. Please upload an MP3, WAV, or M4A file.');
      } else if (error.code === 'file-too-large') {
        setUploadError('File is too large. Max size is 10MB.');
      } else {
        setUploadError(error.message);
      }
      return;
    }

    if (acceptedFiles.length > 0) {
      handleFileUpload(acceptedFiles[0]);
    }
  }, []);

  const isFreeTrialLocked = !isAuthenticated && guestScansUsed >= MAX_FREE_SCANS;
  const isUploadDisabled = !isAuthenticated && isFreeTrialLocked;

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/mpeg': ['.mp3'],
      'audio/wav': ['.wav'],
      'audio/x-wav': ['.wav'],
      'audio/mp4': ['.m4a'],
      'audio/x-m4a': ['.m4a'],
      'audio/*': []
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
    disabled: isUploadDisabled
  });

  const handleUrlAnalyze = async () => {
    if (!urlInput.trim()) return;

    setIsAnalyzing(true);
    setCurrentResult(null);
    setUploadError('');

    try {
      const headers = {
        'Content-Type': 'application/json',
      };
      const token = localStorage.getItem('token');
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const res = await fetch(buildApiUrl('/api/detect'), {
        method: 'POST',
        headers,
        credentials: 'include',
        body: JSON.stringify({ url: urlInput }),
      });

      const { data } = await parseApiResponse(res);

      if (!res.ok) {
        if (data?.error === 'limit_reached') {
          setGuestScansUsed(data?.scans_used ?? MAX_FREE_SCANS);
        }
        setUploadError(getFriendlyApiError(res, data, 'Failed to analyze URL'));
        return;
      }

      if (!data) {
        setUploadError('Unexpected server response. Please try again.');
        return;
      }

      if (!token) {
        setGuestScansUsed(data?.scans_used ?? 0);
      }

      // Extract filename from URL or use a default
      const urlFilename = urlInput.split('/').pop().split('?')[0] || 'Remote Audio';
      handleAnalysisResult(data, urlFilename);
      setUrlInput('');

    } catch (error) {
      console.error('Error analyzing URL:', error);
      setUploadError(error.message || 'An unexpected error occurred. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // SVG Dashboard Circle calculations
  const radius = 45;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = currentResult
    ? circumference - (currentResult.confidence / 100) * circumference
    : circumference;

  // Confidence level badge colors
  const confidenceLevelConfig = {
    High: { bg: 'bg-neon-green/20', text: 'text-neon-green', border: 'border-neon-green/30' },
    Moderate: { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' },
    Low: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' },
    Unknown: { bg: 'bg-gray-500/20', text: 'text-gray-400', border: 'border-gray-500/30' },
  };

  const levelStyle = confidenceLevelConfig[currentResult?.confidenceLevel] || confidenceLevelConfig.Unknown;

  // Fallback signals when no result yet
  const displaySignals = currentResult?.signals?.length > 0
    ? currentResult.signals
    : [
      { name: 'MFCC Spectral Consistency', score: 0 },
      { name: 'Spectrogram Artifact Score', score: 0 },
      { name: 'Prosody Pattern Score', score: 0 },
      { name: 'Signal Consistency', score: 0 },
    ];

  // Signal bar color based on score and context
  const getSignalColor = (score, signalName) => {
    if (!currentResult) return 'bg-gray-600';

    // For "Signal Consistency" and "MFCC Spectral Consistency":
    // high score = good (green), low score = bad (red)
    const isPositiveMetric = signalName.includes('Consistency');

    if (isPositiveMetric) {
      if (score >= 70) return 'bg-neon-green shadow-[0_0_8px_#00FF66]';
      if (score >= 40) return 'bg-amber-400 shadow-[0_0_8px_#f59e0b]';
      return 'bg-red-500 shadow-[0_0_8px_#ef4444]';
    } else {
      // For "Artifact Score" and "Prosody Pattern Score":
      // high score = suspicious (red), low score = clean (green)
      if (score >= 70) return 'bg-red-500 shadow-[0_0_8px_#ef4444]';
      if (score >= 40) return 'bg-amber-400 shadow-[0_0_8px_#f59e0b]';
      return 'bg-neon-green shadow-[0_0_8px_#00FF66]';
    }
  };

  return (
    <div className="flex flex-col xl:flex-row gap-8 w-full max-w-7xl mx-auto">

      {/* Main Analysis Column */}
      <div className="flex-1 space-y-8">
        <div>
          <h2 className="text-3xl font-bold mb-2">AI Voice Detection</h2>
          <p className="text-gray-400 text-sm">Upload audio or paste a URL to analyze for deepfake patterns using our neural network.</p>
        </div>

        {uploadError && (
          <div className="p-4 bg-red-500/10 border border-red-500/50 rounded-xl flex items-start gap-3 text-red-400 animate-slide-down">
            <div className="p-1 shrink-0"><AlertTriangle className="w-5 h-5 text-red-500" /></div>
            <div>
              <p className="font-bold text-sm text-red-500">Detection Failed</p>
              <p className="text-sm mt-0.5">{uploadError}</p>
              {uploadError.includes('limit of 2 free scans') && (
                <div className="mt-3 mb-1">
                  <Link to="/login" className="bg-red-500 text-white px-4 py-2 rounded-lg text-xs font-bold hover:bg-red-600 transition-colors">
                    Login
                  </Link>
                </div>
              )}
            </div>
          </div>
        )}

        {(!isAnalyzing && !currentResult) ? (
          <div className="space-y-8 animate-in fade-in duration-500">
            {/* Upload Container */}
            {isUploadDisabled ? (
              <div className="border border-red-500/40 bg-red-500/5 border-dashed rounded-2xl p-6 md:p-10 flex flex-col items-center justify-center text-center opacity-80">
                <div className="w-14 h-14 bg-[#1C2A22] rounded-full flex items-center justify-center mb-6">
                  <Upload className="w-6 h-6 text-neon-green" />
                </div>
                <h3 className="text-xl font-bold mb-2">Free Scans Exhausted</h3>
                <p className="text-gray-400 text-sm mb-6 max-w-md">
                  You have used all free scans. Log in to unlock file selection and continue analyzing audio.
                </p>
                <Link
                  to="/login"
                  className="bg-red-500 text-white px-6 py-2.5 rounded-lg font-bold hover:bg-red-600 transition-colors shadow-lg shadow-red-500/20"
                >
                  Login
                </Link>
              </div>
            ) : (
              <div
                {...getRootProps()}
                className={`border ${isDragActive ? 'border-neon-green bg-neon-green/5' : 'border-[#1C2A22] bg-[#121A15]'} border-dashed rounded-2xl p-6 md:p-10 flex flex-col items-center justify-center text-center hover:border-neon-green/50 cursor-pointer transition-colors group relative`}
              >
                <input {...getInputProps()} />

                <div className="w-14 h-14 bg-[#1C2A22] rounded-full flex items-center justify-center mb-6 group-hover:bg-neon-green/20 transition-colors">
                  <Upload className="w-6 h-6 text-neon-green" />
                </div>
                <h3 className="text-xl font-bold mb-2">
                  {isDragActive ? 'Drop file here' : 'Upload Audio File'}
                </h3>
                <p className="text-gray-400 text-sm mb-6">Drag and drop your .mp3, .wav, or .m4a file</p>
                <button
                  type="button"
                  className="bg-neon-green text-black px-6 py-2.5 rounded-lg font-bold hover:bg-neon-green-hover transition-colors shadow-lg shadow-neon-green/20 pointer-events-none"
                >
                  Browse Files
                </button>
              </div>
            )}

            {/* OR Divider & URL/Analyze Section */}
            {!isUploadDisabled && (
              <>
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
                      onKeyDown={(e) => e.key === 'Enter' && handleUrlAnalyze()}
                      placeholder="https://example.com/audio.mp3"
                      className="flex-1 bg-transparent px-4 py-3 text-sm focus:outline-none"
                    />
                    <button 
                      onClick={handleUrlAnalyze}
                      disabled={!urlInput.trim() || isAnalyzing}
                      className="bg-[#1C2A22] px-4 flex items-center justify-center hover:bg-[#2A3F33] transition-colors disabled:opacity-50"
                      title="Analyze URL"
                    >
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
                  {isAnalyzing ? "ANALYZING..." : "ANALYZE AUDIO URL"}
                </button>
              </>
            )}
          </div>
        ) : (
          <div className="pt-2 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex flex-wrap gap-4 justify-between items-center mb-6">
              <h2 className="text-xl font-bold">Analysis Verdict</h2>
              <button
                onClick={() => { setIsAnalyzing(false); setCurrentResult(null); }}
                className="bg-[#1C2A22] text-neon-green px-4 py-2 rounded-lg font-bold hover:bg-[#2A3F33] transition-colors border border-neon-green/20 text-sm flex items-center gap-2"
              >
                <Upload className="w-4 h-4" /> Scan Another File
              </button>
            </div>

            {isAnalyzing ? (
              <div className="flex flex-col items-center justify-center py-20 border border-[#1C2A22] bg-[#121A15] rounded-2xl glass-panel">
                <Loader2 className="w-14 h-14 text-neon-green animate-spin mb-6" />
                <h3 className="text-xl font-bold mb-2 text-neon-green">Analyzing Audio...</h3>
                <p className="text-gray-400 text-sm">Running through deep neural networks</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Donut Chart */}
                <div className="glass-panel p-6 flex flex-col items-center justify-center border-[#1C2A22]">
                  <div className="relative w-40 h-40 flex items-center justify-center mb-4">
                    <svg className="w-full h-full -rotate-90 transform" viewBox="0 0 100 100">
                      <circle cx="50" cy="50" r="45" fill="none" stroke="#1C2A22" strokeWidth="10" />
                      <circle
                        cx="50" cy="50" r="45" fill="none"
                        stroke={currentResult ? (currentResult.isFake ? "#EF4444" : currentResult.status === "UNCERTAIN" ? "#F59E0B" : "#00FF66") : "#00FF66"}
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

                  {/* Confidence Level Badge */}
                  {currentResult && (
                    <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border mb-3 ${levelStyle.bg} ${levelStyle.text} ${levelStyle.border}`}>
                      {currentResult.confidenceLevel === 'High' ? <ShieldCheck className="w-3.5 h-3.5" /> : <ShieldAlert className="w-3.5 h-3.5" />}
                      {currentResult.confidenceLevel} Confidence
                    </div>
                  )}

                  <div className="text-center">
                    <div className="text-sm text-gray-400 mb-1">Likelihood of Synthesis</div>
                    <div className={`font-black uppercase text-lg tracking-widest ${currentResult ? (currentResult.isFake ? 'text-red-500 shadow-red-500/50' : currentResult.status === "UNCERTAIN" ? 'text-amber-500 shadow-amber-500/50' : 'text-neon-green neon-text') : 'text-gray-500'}`}>
                      {currentResult ? (currentResult.isFake ? 'HIGH RISK' : currentResult.status === "UNCERTAIN" ? "UNCERTAIN" : 'AUTHENTIC') : 'AWAITING FILE'}
                    </div>
                  </div>
                </div>

                {/* Detection Signals Breakdown */}
                <div className="glass-panel p-6 md:col-span-2 border-[#1C2A22] space-y-6 flex flex-col justify-center">
                  <h3 className="font-bold flex items-center gap-2 mb-2"><Activity className="w-4 h-4 text-neon-green" /> Detection Signals Breakdown</h3>

                  <div className="space-y-5">
                    {displaySignals.map((signal, i) => (
                      <div key={i}>
                        <div className="flex justify-between text-xs font-bold mb-2">
                          <span className="text-gray-300 tracking-wider uppercase">{signal.name}</span>
                          <span className={currentResult ? (
                            signal.name.includes('Consistency')
                              ? (signal.score >= 70 ? 'text-neon-green' : signal.score >= 40 ? 'text-amber-400' : 'text-red-400')
                              : (signal.score >= 70 ? 'text-red-400' : signal.score >= 40 ? 'text-amber-400' : 'text-neon-green')
                          ) : 'text-gray-500'}>
                            {signal.score}% Feature Score
                          </span>
                        </div>
                        <div className="h-1.5 w-full bg-[#1A251D] rounded-full overflow-hidden">
                          <div
                            className={`h-full ${getSignalColor(signal.score, signal.name)} transition-all duration-1000 ease-out rounded-full`}
                            style={{ width: `${signal.score}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
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
                <div className={`p-2 rounded-lg shrink-0 ${res.isFake ? 'bg-red-500/10' : res.status === 'UNCERTAIN' ? 'bg-amber-500/10' : 'bg-neon-green/10'}`}>
                  {res.isFake ? (
                    <AlertTriangle className="w-5 h-5 text-red-500" />
                  ) : res.status === 'UNCERTAIN' ? (
                    <Activity className="w-5 h-5 text-amber-500" />
                  ) : (
                    <CheckCircle2 className="w-5 h-5 text-neon-green" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-semibold text-sm truncate mb-1">{res.filename}</div>
                  <div className="text-xs text-gray-500 flex items-center gap-1.5">
                    {res.time} &bull; <span className="text-gray-400">{res.result}</span>
                  </div>
                </div>
                <div className={`text-[10px] font-bold px-2 py-1 rounded shrink-0 ${res.isFake ? 'bg-red-500/20 text-red-500 border border-red-500/30' : res.status === 'UNCERTAIN' ? 'bg-amber-500/20 text-amber-500 border border-amber-500/30' : 'bg-neon-green/20 text-neon-green border border-neon-green/30'}`}>
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
