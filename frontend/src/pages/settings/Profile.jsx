import { useState, useEffect } from 'react';
import { User, CheckCircle2 } from 'lucide-react';

export default function ProfileSettings() {
  const [profile, setProfile] = useState({ name: '', email: '' });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const token = localStorage.getItem('token');
        const headers = {};
        if (token) {
          headers['Authorization'] = 'Bearer ' + token;
        }
        const response = await fetch('/api/profile', {
          credentials: 'include',
          headers
        });
        if (response.ok) {
          const data = await response.json();
          setProfile({ name: data.name || '', email: data.email || '' });
        }
      } catch (err) {
        console.error('Failed to fetch profile', err);
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, []);

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    setMessage('');

    try {
      const token = localStorage.getItem('token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) {
        headers['Authorization'] = 'Bearer ' + token;
      }
      const response = await fetch('/api/profile', {
        method: 'PUT',
        credentials: 'include',
        headers,
        body: JSON.stringify({ name: profile.name })
      });
      if (response.ok) {
        const storedUser = JSON.parse(localStorage.getItem('user') || '{}');
        localStorage.setItem('user', JSON.stringify({ ...storedUser, name: profile.name }));
        window.dispatchEvent(new Event('authChange'));

        setMessage('Profile updated successfully');
        setTimeout(() => setMessage(''), 3000);
      }
    } catch (err) {
      console.error('Failed to save profile', err);
    } finally {
      setSaving(false);
    }
  };

  if (loading) return null;

  return (
    <div className="space-y-6 max-w-4xl">

      {/* Profile Information Card */}
      <div className="bg-[#151e18] border border-[#1c2a22] rounded-xl overflow-hidden shadow-sm">
        <div className="flex items-center gap-3 px-8 pt-8 pb-6">
          <div className="w-6 h-6 rounded-full border-[2.5px] border-neon-green flex items-center justify-center">
            <User strokeWidth={2.5} className="w-3.5 h-3.5 text-neon-green" />
          </div>
          <h2 className="text-[20px] font-bold text-white tracking-wide">Profile Information</h2>
        </div>

        {message && (
          <div className="mx-8 mb-4 p-4 bg-neon-green/10 border border-neon-green/30 rounded-lg flex items-center gap-3">
            <CheckCircle2 className="w-5 h-5 text-neon-green" />
            <p className="text-sm font-medium text-neon-green">{message}</p>
          </div>
        )}

        <form onSubmit={handleSave} className="px-8 pb-8 pt-0 flex flex-col gap-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6 mb-auto">
            <div>
              <label className="block text-[13px] font-semibold mb-2 text-white">Full Name</label>
              <input
                type="text"
                value={profile.name}
                onChange={(e) => setProfile({ ...profile, name: e.target.value })}
                className="w-full bg-[#121a15] border border-[#1c2a22] text-white rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-neon-green/50 transition-colors"
                required
              />
            </div>

            <div>
              <label className="block text-[13px] font-semibold mb-2 text-white">Email Address</label>
              <input
                type="email"
                value={profile.email}
                readOnly
                className="w-full bg-[#121a15] border border-[#1c2a22] text-gray-300 rounded-lg px-4 py-2.5 text-sm cursor-not-allowed opacity-80"
              />
            </div>
          </div>

          <div className="flex justify-end mt-4">
            <button
              type="submit"
              disabled={saving}
              className="bg-neon-green text-black px-5 py-2 rounded-lg font-bold text-[14px] hover:bg-neon-green-hover transition-all disabled:opacity-70 tracking-wide"
            >
              {saving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
