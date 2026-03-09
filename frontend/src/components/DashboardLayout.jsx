import { useState, useEffect, useRef } from 'react';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { Shield, LayoutDashboard, History, Settings, HelpCircle, Bell, LogOut, ChevronDown } from 'lucide-react';

// Use environment variable for API base URL, fallback to relative path for production
const API_BASE = import.meta.env.VITE_API_URL || '';

export default function DashboardLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const path = location.pathname;
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const profileRef = useRef(null);

  // Get user info from localStorage
  const [user, setUser] = useState(null);

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

  // Get user initials
  const getInitials = () => {
    if (!user?.name) return '?';
    const parts = user.name.trim().split(/\s+/);
    if (parts.length >= 2) return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    return parts[0][0].toUpperCase();
  };

  const handleLogout = async () => {
    try {
      await fetch(`${API_BASE}/api/logout`, {
        method: 'POST',
        credentials: 'include',
      });
    } catch (err) {
      console.error('Logout request failed:', err);
    }
    localStorage.removeItem('user');
    window.dispatchEvent(new Event('authChange'));
    navigate('/');
  };

  const navItems = [
    { name: 'Dashboard', path: '/app', icon: LayoutDashboard },
    { name: 'Detection History', path: '/app/history', icon: History },
  ];

  const systemItems = [
    { name: 'Account Settings', path: '/app/settings', icon: Settings },
    { name: 'Help & Support', path: '/app/help', icon: HelpCircle },
  ];

  return (
    <div className="flex h-screen bg-[#0E1511]">
      {/* Sidebar */}
      <aside className="w-64 border-r border-[#1C2A22] flex flex-col pt-6 pb-6 bg-[#121A15]">
        <div className="px-6 mb-8">
          <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <Shield className="w-6 h-6 text-neon-green" fill="#00FF66" strokeWidth={1} />
            <div>
              <div className="font-bold text-lg tracking-wide uppercase leading-tight">AI Guardian</div>
              <div className="text-neon-green text-xs font-medium">Secure Detection</div>
            </div>
          </Link>
        </div>

        <nav className="flex-1 px-4 space-y-1">
          {navItems.map((item) => {
            const isActive = path === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.path}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive ? 'bg-[#1C2A22] text-neon-green' : 'text-gray-400 hover:text-white hover:bg-[#1C2A22]/50'
                }`}
              >
                <Icon className="w-5 h-5" />
                {item.name}
              </Link>
            );
          })}
          
          <div className="pt-8 pb-2 px-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            System
          </div>
          
          {systemItems.map((item) => {
            const isActive = path === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.path}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive ? 'bg-[#1C2A22] text-neon-green' : 'text-gray-400 hover:text-white hover:bg-[#1C2A22]/50'
                }`}
              >
                <Icon className="w-5 h-5" />
                {item.name}
              </Link>
            );
          })}

          {/* Logout Button in Sidebar */}
          {user && (
            <div className="pt-4">
              <button
                onClick={handleLogout}
                className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-colors w-full"
              >
                <LogOut className="w-5 h-5" />
                Logout
              </button>
            </div>
          )}
        </nav>

        {/* Pro Plan Box */}
        <div className="px-4 mt-auto">
          <div className="bg-[#1C2A22]/50 border border-[#2A3F33] rounded-xl p-4">
            <div className="text-neon-green text-[10px] font-bold tracking-wider mb-2 uppercase">Pro Plan</div>
            <div className="text-xs text-gray-400 mb-3">240/500 minutes used this month.</div>
            <div className="w-full bg-[#0E1511] h-1.5 rounded-full mb-4 overflow-hidden">
              <div className="bg-neon-green h-full w-[48%] shadow-[0_0_10px_#00FF66]"></div>
            </div>
            <button className="w-full bg-neon-green text-black text-xs font-bold py-2 rounded-md hover:bg-neon-green-hover transition-colors shadow-[0_0_15px_rgba(0,255,102,0.3)]">
              Upgrade Now
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Dashboard Header */}
        <header className="h-16 flex items-center justify-between px-8 border-b border-[#1C2A22] shrink-0 bg-[#0E1511]">
          <h1 className="font-semibold text-lg">New Analysis</h1>
          <div className="flex items-center gap-4">
            <button className="text-gray-400 hover:text-white transition-colors relative">
              <Bell className="w-5 h-5" />
              <span className="absolute top-0 right-0 w-2 h-2 bg-neon-green rounded-full shadow-[0_0_8px_#00FF66]"></span>
            </button>

            {/* Profile Avatar with Dropdown */}
            {user ? (
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
                        to="/app/settings"
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
            ) : (
              <Link
                to="/login"
                className="bg-neon-green text-black px-5 py-2 rounded-md text-sm font-semibold hover:bg-neon-green-hover transition-colors neon-glow"
              >
                Log In
              </Link>
            )}
          </div>
        </header>
        
        {/* Scrollable Content */}
        <div className="flex-1 overflow-auto p-8 custom-scrollbar">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
