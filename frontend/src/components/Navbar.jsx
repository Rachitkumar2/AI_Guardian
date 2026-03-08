import { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Shield, LogOut } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const path = location.pathname;
  const [user, setUser] = useState(null);

  useEffect(() => {
    const stored = localStorage.getItem('user');
    if (stored) {
      try { setUser(JSON.parse(stored)); } catch { setUser(null); }
    }
  }, []);

  const getInitials = () => {
    if (!user?.name) return '?';
    const parts = user.name.trim().split(/\s+/);
    if (parts.length >= 2) return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    return parts[0][0].toUpperCase();
  };

  const handleLogout = async () => {
    try {
      await fetch('http://localhost:5000/api/logout', { credentials: 'include' });
    } catch (err) {
      console.error('Logout failed:', err);
    }
    localStorage.removeItem('user');
    setUser(null);
    navigate('/');
  };

  const links = [
    { name: 'How It Works', path: '/' },
    { name: 'Library', path: '/library' },
    { name: 'Tools', path: '/tools' },
  ];

  return (
    <nav className="flex items-center justify-between px-8 py-5 border-b border-dark-border/50 sticky top-0 z-50 bg-[#0E1511]/90 backdrop-blur-md">
      {/* Logo */}
      <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
        <Shield className="w-6 h-6 text-neon-green" fill="#00FF66" strokeWidth={1} />
        <span className="font-bold text-lg tracking-wide uppercase">AI Guardian</span>
      </Link>

      {/* Center Links */}
      <div className="hidden md:flex items-center gap-8">
        {links.map((link) => {
          const isActive = path === link.path || (path.startsWith(link.path) && link.path !== '/');
          return (
            <Link
              key={link.name}
              to={link.path}
              className={`text-sm font-medium transition-colors ${
                isActive ? 'text-neon-green' : 'text-gray-300 hover:text-white'
              }`}
            >
              {link.name}
            </Link>
          );
        })}
      </div>

      {/* Right Actions */}
      <div className="flex items-center gap-4">
        {user ? (
          <>
            <Link
              to="/app"
              className="bg-neon-green text-black px-5 py-2 rounded-md text-sm font-semibold hover:bg-neon-green-hover transition-colors neon-glow"
            >
              Dashboard
            </Link>
            <div className="flex items-center gap-3">
              <Link to="/app" className="w-8 h-8 rounded-full bg-neon-green/20 border-2 border-neon-green/40 flex items-center justify-center cursor-pointer hover:border-neon-green/60 transition-colors">
                <span className="text-neon-green text-xs font-bold">{getInitials()}</span>
              </Link>
              <button
                onClick={handleLogout}
                className="text-gray-400 hover:text-red-400 transition-colors"
                title="Logout"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          </>
        ) : (
          <>
            <Link to="/login" className="text-sm font-medium text-gray-300 hover:text-white transition-colors">
              Log In
            </Link>
            <Link
              to="/app"
              className="bg-neon-green text-black px-5 py-2 rounded-md text-sm font-semibold hover:bg-neon-green-hover transition-colors neon-glow"
            >
              Try for free
            </Link>
          </>
        )}
      </div>
    </nav>
  );
}
