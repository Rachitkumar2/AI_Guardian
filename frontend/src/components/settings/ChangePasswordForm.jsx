import { useState } from 'react';
import { Key, Eye, EyeOff } from 'lucide-react';

export default function ChangePasswordForm() {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showCurrent, setShowCurrent] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('');
    setError('');

    if (newPassword !== confirmPassword) {
      setError('New passwords do not match');
      return;
    }
    if (newPassword.length < 8) {
      setError('New password must be at least 8 characters');
      return;
    }

    setLoading(true);

    try {
      const token = localStorage.getItem('token');
      const res = await fetch('/api/security/change-password', {
        method: 'POST',
        credentials: 'include',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + token
        },
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword })
      });
      const data = await res.json();
      
      if (res.ok) {
        setMessage('Password updated successfully');
        setCurrentPassword('');
        setNewPassword('');
        setConfirmPassword('');
      } else {
        setError(data.message || 'Failed to update password');
      }
    } catch (err) {
      setError('Connection error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-[#151E18] border border-[#1C2A22] rounded-2xl p-6 sm:p-8">
      <div className="flex items-center justify-between mb-8 pb-6 border-b border-[#1C2A22]">
        <div className="flex items-center gap-3">
          <Key className="w-6 h-6 text-neon-green" />
          <h2 className="text-xl font-bold">Change Password</h2>
        </div>
      </div>

      {message && <p className="mb-4 text-sm text-neon-green font-medium bg-neon-green/10 p-3 rounded-lg border border-neon-green/30">{message}</p>}
      {error && <p className="mb-4 text-sm text-red-500 font-medium bg-red-500/10 p-3 rounded-lg border border-red-500/30">{error}</p>}

      <form onSubmit={handleSubmit} className="max-w-md space-y-6">
        <div>
          <label className="block text-sm font-semibold mb-2 text-gray-300">Current Password</label>
          <div className="relative group">
            <input
              type={showCurrent ? 'text' : 'password'}
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              className="w-full bg-[#121A15] border border-[#1C2A22] text-white rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:border-neon-green/50 transition-all"
              required
            />
            <button
              type="button"
              onClick={() => setShowCurrent(!showCurrent)}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 transition-colors focus:outline-none"
            >
              {showCurrent ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
            </button>
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-semibold mb-2 text-gray-300">New Password</label>
          <div className="relative group">
            <input
              type={showNew ? 'text' : 'password'}
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="w-full bg-[#121A15] border border-[#1C2A22] text-white rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:border-neon-green/50 transition-all"
              required
            />
            <button
              type="button"
              onClick={() => setShowNew(!showNew)}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 transition-colors focus:outline-none"
            >
              {showNew ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
            </button>
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-semibold mb-2 text-gray-300">Confirm New Password</label>
          <div className="relative group">
            <input
              type={showConfirm ? 'text' : 'password'}
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="w-full bg-[#121A15] border border-[#1C2A22] text-white rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:border-neon-green/50 transition-all"
              required
            />
            <button
              type="button"
              onClick={() => setShowConfirm(!showConfirm)}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 transition-colors focus:outline-none"
            >
              {showConfirm ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
            </button>
          </div>
        </div>

        <div>
          <button 
            type="submit"
            disabled={loading}
            className="bg-neon-green text-black px-6 py-3 rounded-xl font-bold hover:bg-neon-green-hover transition-all shadow-[0_0_20px_rgba(0,255,102,0.1)] hover:shadow-[0_0_30px_rgba(0,255,102,0.2)] disabled:opacity-70 flex items-center gap-2"
          >
            {loading ? 'Updating...' : 'Update Password'}
          </button>
        </div>
      </form>
    </div>
  );
}
