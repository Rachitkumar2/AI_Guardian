import { useState, useEffect } from 'react';
import { Laptop, Smartphone, Monitor } from 'lucide-react';

export default function ActiveSessions() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      const res = await fetch('http://localhost:5000/api/security/sessions', { credentials: 'include' });
      if (res.ok) {
        setSessions(await res.json());
      }
    } catch (err) {
      console.error('Failed to fetch sessions', err);
    } finally {
      setLoading(false);
    }
  };

  const logoutOtherSessions = async () => {
    try {
      const res = await fetch('http://localhost:5000/api/security/logout-other', { 
        method: 'POST',
        credentials: 'include' 
      });
      if (res.ok) {
        fetchSessions();
      }
    } catch (err) {
      console.error('Failed to logout other sessions', err);
    }
  };

  const getDeviceIcon = (deviceStr) => {
    const l = deviceStr.toLowerCase();
    if (l.includes('mac') || l.includes('windows') || l.includes('linux')) return <Laptop className="w-5 h-5 text-neon-green" />;
    if (l.includes('ios') || l.includes('android') || l.includes('iphone') || l.includes('ipad')) return <Smartphone className="w-5 h-5 text-gray-400" />;
    return <Monitor className="w-5 h-5 text-gray-400" />;
  };

  if (loading) return null;

  return (
    <div className="bg-[#151E18] border border-[#1C2A22] rounded-2xl p-6 sm:p-8">
      <div className="flex items-center justify-between mb-8 pb-6 border-b border-[#1C2A22]">
        <div className="flex items-center gap-3">
          <Monitor className="w-6 h-6 text-neon-green" />
          <h2 className="text-xl font-bold">Active Sessions</h2>
        </div>
        
        {sessions.length > 1 && (
          <button 
            onClick={logoutOtherSessions}
            className="text-xs font-bold text-red-500 hover:text-red-400 uppercase tracking-widest transition-colors"
          >
            Log out all other sessions
          </button>
        )}
      </div>

      <div className="space-y-4">
        {sessions.map((s, i) => (
          <div key={i} className="bg-[#121A15] border border-[#1C2A22] rounded-xl p-4 flex items-center gap-4">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${s.current ? 'bg-neon-green/10' : 'bg-[#1C2A22]'}`}>
              {getDeviceIcon(s.device)}
            </div>
            
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-3 mb-1">
                <h4 className="font-bold text-sm truncate text-white">{s.device} ({s.browser})</h4>
                {s.current && (
                  <span className="bg-neon-green/20 text-neon-green text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-widest">
                    CURRENT
                  </span>
                )}
              </div>
              <div className="text-xs text-gray-500 font-mono flex gap-2">
                <span>{s.ip}</span> • <span>{new Date(s.login_time).toLocaleString()}</span>
              </div>
            </div>
          </div>
        ))}
        {sessions.length === 0 && <p className="text-gray-500 text-sm">No active sessions found.</p>}
      </div>
    </div>
  );
}
