import { useState, useEffect } from 'react';
import { History } from 'lucide-react';
import { apiUrl } from '../../utils/api';

export default function LoginHistoryTable() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch(apiUrl('/api/security/login-history'), { credentials: 'include' });
        if (res.ok) {
          setHistory(await res.json());
        }
      } catch (err) {
        console.error('Failed to fetch login history', err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  if (loading) return null;

  return (
    <div className="bg-[#151E18] border border-[#1C2A22] rounded-2xl p-6 sm:p-8 overflow-hidden">
      <div className="flex items-center gap-3 mb-6">
        <History className="w-6 h-6 text-neon-green" />
        <h2 className="text-xl font-bold">Login History</h2>
      </div>

      <div className="overflow-x-auto -mx-6 sm:-mx-8 px-6 sm:px-8">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-[#1C2A22] text-xs font-bold text-gray-500 uppercase tracking-widest">
              <th className="pb-4 font-semibold w-1/3">Date & Time</th>
              <th className="pb-4 font-semibold w-1/4">Status</th>
              <th className="pb-4 font-semibold w-1/3">Device / Browser</th>
              <th className="pb-4 font-semibold w-1/6 text-right">IP Address</th>
            </tr>
          </thead>
          <tbody className="text-sm">
            {history.map((log, i) => (
              <tr key={i} className="border-b border-[#1C2A22]/50 hover:bg-[#1A251D] transition-colors group">
                <td className="py-4 text-gray-300 whitespace-nowrap">
                  {new Date(log.time).toLocaleString(undefined, {
                    month: 'short', day: 'numeric', year: 'numeric',
                    hour: '2-digit', minute: '2-digit', second: '2-digit'
                  })}
                </td>
                <td className="py-4 flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${log.status === 'Success' ? 'bg-neon-green shadow-[0_0_8px_rgba(0,255,102,0.8)]' : 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]'}`}></span>
                  <span className={log.status === 'Success' ? 'text-neon-green font-medium' : 'text-red-400 font-medium'}>
                    {log.status}
                  </span>
                </td>
                <td className="py-4 text-gray-400">
                  <div className="truncate max-w-[200px]">{log.device} • {log.browser}</div>
                </td>
                <td className="py-4 text-gray-500 font-mono text-right whitespace-nowrap">
                  {log.ip}
                </td>
              </tr>
            ))}
            {history.length === 0 && (
              <tr>
                <td colSpan="4" className="py-8 text-center text-gray-500">No login history recorded yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
