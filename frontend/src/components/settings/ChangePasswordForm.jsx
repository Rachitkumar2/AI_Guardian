import { useState } from 'react';
import { Key } from 'lucide-react';

export default function ChangePasswordForm() {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
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
          <input
            type="password"
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            className="w-full bg-[#121A15] border border-[#1C2A22] text-white rounded-xl px-4 py-3 focus:outline-none focus:border-neon-green/50 transition-colors"
            required
          />
        </div>
        
        <div>
          <label className="block text-sm font-semibold mb-2 text-gray-300">New Password</label>
          <input
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            className="w-full bg-[#121A15] border border-[#1C2A22] text-white rounded-xl px-4 py-3 focus:outline-none focus:border-neon-green/50 transition-colors"
            required
          />
        </div>
        
        <div>
          <label className="block text-sm font-semibold mb-2 text-gray-300">Confirm New Password</label>
          <input
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            className="w-full bg-[#121A15] border border-[#1C2A22] text-white rounded-xl px-4 py-3 focus:outline-none focus:border-neon-green/50 transition-colors"
            required
          />
        </div>

        <div>
          <button 
            type="submit"
            disabled={loading}
            className="bg-neon-green text-black px-6 py-3 rounded-xl font-bold hover:bg-neon-green-hover transition-all disabled:opacity-70 flex items-center gap-2"
          >
            {loading ? 'Updating...' : 'Update Password'}
          </button>
        </div>
      </form>
    </div>
  );
}
