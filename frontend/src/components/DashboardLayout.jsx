import { useState, useEffect, useRef } from 'react';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { Shield, LayoutDashboard, History, Settings, HelpCircle, Bell, LogOut, ChevronDown, Menu, X } from 'lucide-react';

// Use environment variable for API base URL, fallback to relative path for production
const API_BASE = import.meta.env.VITE_API_URL || '';

export default function DashboardLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const path = location.pathname;
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [showMobileSidebar, setShowMobileSidebar] = useState(false);
  const profileRef = useRef(null);

  // Get user info from localStorage
  const [user, setUser] = useState(null);

  useEffect(() => {
    const handleAuthChange = () => {
      const stored = localStorage.getItem('user');
      if (stored) {
        try { setUser(JSON.parse(stored)); } catch { setUser(null); }
      } else {
        localStorage.removeItem('token');
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
    localStorage.removeItem('token');
    window.dispatchEvent(new Event('authChange'));
    navigate('/');
  };

  const navItems = [
    { name: 'Dashboard', path: '/app', icon: LayoutDashboard },
    { name: 'Detection History', path: '/app/history', icon: History },
  ];

  const systemItems = [
    { name: 'Account Settings', path: '/settings/profile', icon: Settings },
  ];

  return (
    <div className="flex h-screen bg-dark-bg overflow-hidden relative">
      {/* Mobile Backdrop */}
      {showMobileSidebar && (
        <div 
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 md:hidden"
          onClick={() => setShowMobileSidebar(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`fixed inset-y-0 left-0 bg-[#121A15] border-r border-dark-border w-64 flex flex-col pt-6 pb-6 z-50 transform transition-transform duration-300 ease-in-out md:relative md:translate-x-0 ${showMobileSidebar ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="px-6 mb-8 flex items-center justify-between">
          <Link to="/" onClick={() => setShowMobileSidebar(false)} className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <Shield className="w-6 h-6 text-neon-green" fill="#00FF66" strokeWidth={1} />
            <div>
              <div className="font-bold text-lg tracking-wide uppercase leading-tight">AI Guardian</div>
              <div className="text-neon-green text-xs font-medium">Secure Detection</div>
            </div>
          </Link>
          <button 
            onClick={() => setShowMobileSidebar(false)}
            className="md:hidden text-gray-400 hover:text-white mt-1"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <nav className="flex-1 px-4 space-y-1">
          {navItems.map((item) => {
            const isActive = path === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.path}
                onClick={() => setShowMobileSidebar(false)}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive ? 'bg-dark-border text-neon-green' : 'text-gray-400 hover:text-white hover:bg-dark-border/50'
                }`}
              >
                <Icon className="w-5 h-5" />
                {item.name}
              </Link>
            );
          })}
          
          {user && (
            <>
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
                    onClick={() => setShowMobileSidebar(false)}
                    className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      isActive ? 'bg-dark-border text-neon-green' : 'text-gray-400 hover:text-white hover:bg-dark-border/50'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    {item.name}
                  </Link>
                );
              })}
            </>
          )}

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


      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Dashboard Header */}
        <header className="h-16 flex items-center justify-between px-4 md:px-8 border-b border-dark-border shrink-0 bg-dark-bg z-10">
          <div className="flex items-center gap-3">
            <button 
              className="md:hidden p-1.5 text-gray-400 hover:text-white transition-colors bg-dark-border/50 rounded-lg"
              onClick={() => setShowMobileSidebar(true)}
            >
              <Menu className="w-5 h-5" />
            </button>
            <h1 className="font-bold text-2xl max-sm:text-lg hidden sm:block">
              {
                navItems.find(i => i.path === path)?.name || 
                systemItems.find(i => i.path === path)?.name || 
                'Dashboard'
              }
            </h1>
          </div>
          <div className="flex items-center gap-4">

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
                  <div className="absolute right-0 top-12 w-56 bg-[#121A15] border border-dark-border rounded-xl shadow-2xl shadow-black/50 overflow-hidden z-50">
                    <div className="px-4 py-3 border-b border-dark-border">
                      <p className="text-sm font-semibold text-white truncate">{user?.name || 'User'}</p>
                      <p className="text-xs text-gray-400 truncate">{user?.email || ''}</p>
                    </div>
                    <div className="p-1.5">
                      <Link
                        to="/settings/profile"
                        onClick={() => setShowProfileMenu(false)}
                        className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-300 hover:text-white hover:bg-dark-border transition-colors"
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
        <div className="flex-1 overflow-auto p-4 md:p-8 custom-scrollbar relative z-0">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
