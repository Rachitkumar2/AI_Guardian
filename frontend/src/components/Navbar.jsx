import { useState, useEffect, useRef } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Shield, LogOut, Settings, ChevronDown, Menu, X } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const path = location.pathname;
  const [user, setUser] = useState(null);
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);
  const profileRef = useRef(null);

  // Close profile menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (profileRef.current && !profileRef.current.contains(e.target)) {
        setShowProfileMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    const handleAuthChange = () => {
      const stored = localStorage.getItem('user');
      if (stored) {
        try { setUser(JSON.parse(stored)); } catch { setUser(null); }
      } else {
        setUser(null);
      }
    };

    handleAuthChange();
    window.addEventListener('authChange', handleAuthChange);
    return () => window.removeEventListener('authChange', handleAuthChange);
  }, []);

  const getInitials = () => {
    if (!user?.name) return '?';
    const parts = user.name.trim().split(/\s+/);
    if (parts.length >= 2) return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    return parts[0][0].toUpperCase();
  };

  const handleLogout = async () => {
    try {
      await fetch('/api/logout', { 
        method: 'POST',
        credentials: 'include' 
      });
    } catch (err) {
      console.error('Logout failed:', err);
    }
    localStorage.removeItem('user');
    localStorage.removeItem('token');
    window.dispatchEvent(new Event('authChange'));
    setUser(null);
    navigate('/');
  };

  const links = [
    { name: 'How It Works', path: '/' },
    { name: 'Library', path: '/library' },
    { name: 'Tools', path: '/tools' },
  ];

  return (
    <nav className="flex items-center justify-between px-4 md:px-8 py-4 md:py-5 border-b border-dark-border/50 sticky top-0 z-50 bg-[#0E1511]/90 backdrop-blur-md">
      {/* Logo */}
      <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity relative z-[51]">
        <Shield className="w-6 h-6 text-neon-green" fill="#00FF66" strokeWidth={1} />
        <span className="font-bold text-lg tracking-wide uppercase">AI Guardian</span>
      </Link>

      {/* Center Links (Desktop) */}
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

      {/* Right Actions (Desktop) */}
      <div className="hidden md:flex items-center gap-4">
        {user ? (
          <>
            <Link
              to="/app"
              className="bg-neon-green text-black px-5 py-2 rounded-md text-sm font-semibold hover:bg-neon-green-hover transition-colors neon-glow"
            >
              Dashboard
            </Link>
            
            {/* Profile Avatar with Dropdown */}
            <div className="relative" ref={profileRef}>
              <button
                onClick={() => setShowProfileMenu(!showProfileMenu)}
                className="flex items-center gap-2 hover:opacity-80 transition-opacity"
              >
                <div className="w-9 h-9 rounded-full bg-neon-green/20 border-2 border-neon-green/40 flex items-center justify-center cursor-pointer">
                  <span className="text-neon-green text-sm font-bold">{getInitials()}</span>
                </div>
                <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${showProfileMenu ? 'rotate-180' : ''}`} />
              </button>

              {/* Dropdown Menu */}
              {showProfileMenu && (
                <div className="absolute right-0 top-12 w-56 bg-[#121A15] border border-[#1C2A22] rounded-xl shadow-2xl shadow-black/50 overflow-hidden z-50">
                  <div className="px-4 py-3 border-b border-[#1C2A22]">
                    <p className="text-sm font-semibold text-white truncate">{user?.name || 'User'}</p>
                    <p className="text-xs text-gray-400 truncate">{user?.email || ''}</p>
                  </div>
                  <div className="p-1.5">
                    <Link
                      to="/settings/profile"
                      onClick={() => setShowProfileMenu(false)}
                      className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-300 hover:text-white hover:bg-[#1C2A22] transition-colors"
                    >
                      <Settings className="w-4 h-4" />
                      Account Settings
                    </Link>
                    <button
                      onClick={handleLogout}
                      className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-colors w-full"
                    >
                      <LogOut className="w-4 h-4" />
                      Sign Out
                    </button>
                  </div>
                </div>
              )}
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

      {/* Mobile Hamburger Toggle */}
      <button
        onClick={() => setShowMobileMenu(!showMobileMenu)}
        className="md:hidden p-2 text-gray-300 hover:text-white transition-colors relative z-[51]"
      >
        {showMobileMenu ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
      </button>

      {/* Mobile Menu Dropdown */}
      {showMobileMenu && (
        <div className="absolute top-0 left-0 w-full h-screen bg-[#0E1511]/95 backdrop-blur-xl z-50 flex flex-col pt-24 px-6 md:hidden">
          <div className="flex flex-col gap-6">
            {links.map((link) => {
              const isActive = path === link.path || (path.startsWith(link.path) && link.path !== '/');
              return (
                <Link
                  key={link.name}
                  to={link.path}
                  onClick={() => setShowMobileMenu(false)}
                  className={`text-xl font-bold transition-colors ${
                    isActive ? 'text-neon-green' : 'text-gray-300 hover:text-white'
                  }`}
                >
                  {link.name}
                </Link>
              );
            })}
          </div>

          <div className="mt-12 pt-8 border-t border-dark-border flex flex-col gap-4 w-full">
            {user ? (
              <>
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-12 h-12 rounded-full bg-neon-green/20 border-2 border-neon-green/40 flex items-center justify-center">
                    <span className="text-neon-green font-bold">{getInitials()}</span>
                  </div>
                  <div>
                    <p className="font-semibold text-white">{user?.name || 'User'}</p>
                    <p className="text-sm text-gray-400">{user?.email || ''}</p>
                  </div>
                </div>
                <Link
                  to="/app"
                  onClick={() => setShowMobileMenu(false)}
                  className="bg-neon-green text-black px-5 py-3 rounded-xl text-center font-bold"
                >
                  Dashboard
                </Link>
                <Link
                  to="/settings/profile"
                  onClick={() => setShowMobileMenu(false)}
                  className="flex items-center justify-center gap-3 px-5 py-3 rounded-xl bg-dark-border text-white font-semibold"
                >
                  Account Settings
                </Link>
                <button
                  onClick={() => {
                    handleLogout();
                    setShowMobileMenu(false);
                  }}
                  className="flex items-center justify-center gap-3 px-5 py-3 rounded-xl border border-red-500/30 text-red-400 hover:bg-red-500/10 font-semibold"
                >
                  Sign Out
                </button>
              </>
            ) : (
              <>
                <Link
                  to="/login"
                  onClick={() => setShowMobileMenu(false)}
                  className="py-3 text-center text-lg font-semibold text-gray-300 hover:text-white"
                >
                  Log In
                </Link>
                <Link
                  to="/app"
                  onClick={() => setShowMobileMenu(false)}
                  className="bg-neon-green text-black px-5 py-4 rounded-xl text-center font-bold text-lg"
                >
                  Try for free
                </Link>
              </>
            )}
          </div>
        </div>
      )}
    </nav>
  );
}
