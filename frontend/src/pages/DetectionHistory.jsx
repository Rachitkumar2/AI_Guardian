import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { History, AlertTriangle, CheckCircle2, ShieldAlert, Loader2 } from 'lucide-react';

export default function DetectionHistory() {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const [user, setUser] = useState(null);

    useEffect(() => {
        const stored = localStorage.getItem('user');
        if (stored) {
            try { setUser(JSON.parse(stored)); } catch { setUser(null); }
        } else {
            setUser(null);
        }
    }, []);

    const fetchHistory = useCallback(async () => {
        try {
            const res = await fetch('/api/history', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                },
                credentials: 'include'
            });

            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.message || data.error || 'Failed to fetch history');
            }
            setHistory(data.history || []);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (user) {
            fetchHistory();
        } else {
            setLoading(false);
        }
    }, [user, fetchHistory]);

    if (!user) {
        return (
            <div className="w-full flex-1 flex flex-col items-center justify-center p-8 h-full min-h-[500px]">
                <div className="w-16 h-16 rounded-full bg-[#1C2A22] flex items-center justify-center mb-6 shadow-[0_0_15px_rgba(0,255,102,0.1)]">
                    <ShieldAlert className="w-8 h-8 text-neon-green" />
                </div>
                <h2 className="text-3xl font-bold mb-4">Authentication Required</h2>
                <p className="text-gray-400 mb-8 max-w-md text-center leading-relaxed">
                    You must be logged in to securely view and manage your AI detection history.
                </p>
                <Link
                    to="/login"
                    className="bg-neon-green text-black px-8 py-3 rounded-xl font-bold hover:bg-neon-green-hover transition-all neon-glow shadow-lg shadow-neon-green/20"
                >
                    Log In Now
                </Link>
            </div>
        );
    }

    return (
        <div className="max-w-7xl mx-auto w-full space-y-8">
            {loading ? (
                <div className="flex flex-col items-center justify-center py-20">
                    <Loader2 className="w-10 h-10 text-neon-green animate-spin mb-4" />
                    <p className="text-gray-400">Loading history...</p>
                </div>
            ) : error ? (
                <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 text-center text-red-400">
                    <p>Error: {error}</p>
                </div>
            ) : history.length === 0 ? (
                <div className="glass-panel text-center py-20 border-[#1C2A22] shadow-2xl shadow-black/50">
                    <History className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                    <h3 className="text-lg font-bold mb-2">No Scans Yet</h3>
                    <p className="text-gray-400 text-sm mb-6">You haven't analyzed any audio files yet.</p>
                    <Link to="/app" className="bg-[#1C2A22] text-neon-green px-6 py-2 rounded-lg font-bold hover:bg-[#2A3F33] transition-colors border border-neon-green/20">
                        Start Scanning
                    </Link>
                </div>
            ) : (
                <div className="space-y-4">
                    {history.map((item, idx) => {
                        const date = item.timestamp ? new Date(item.timestamp).toLocaleString() : 'Unknown Data';
                        const isFake = item.result?.toLowerCase() === 'fake';
                        return (
                            <div key={idx} className="glass-panel border-[#1C2A22] p-6 flex flex-col md:flex-row md:items-center gap-6 hover:border-neon-green/30 transition-colors shadow-xl shadow-black/40">
                                <div className={`p-4 rounded-xl shrink-0 flex items-center justify-center ${isFake ? 'bg-red-500/10' : 'bg-neon-green/10'}`}>
                                    {isFake ? <AlertTriangle className="w-6 h-6 text-red-500" /> : <CheckCircle2 className="w-6 h-6 text-neon-green" />}
                                </div>

                                <div className="flex-1 min-w-0">
                                    <h4 className="font-bold text-lg mb-1 truncate" title={item.filename}>{item.filename}</h4>
                                    <div className="text-sm text-gray-400 font-medium">
                                        {date}
                                    </div>
                                </div>

                                <div className="flex flex-row items-center gap-6 shrink-0">
                                    <div className="flex flex-col items-end">
                                        <span className="text-[10px] text-gray-500 uppercase tracking-widest font-bold">Confidence</span>
                                        <span className="text-xl font-black">{Math.round(item.confidence)}%</span>
                                    </div>
                                    <div className={`px-4 py-2 rounded-lg font-bold text-sm min-w-[120px] text-center ${isFake ? 'bg-red-500/20 text-red-500 border border-red-500/30' : 'bg-neon-green/20 text-neon-green border border-neon-green/30'}`}>
                                        {isFake ? 'SYNTHETIC' : 'AUTHENTIC'}
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
