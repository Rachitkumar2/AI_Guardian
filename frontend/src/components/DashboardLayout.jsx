import { Outlet, Link, useLocation } from 'react-router-dom';
import { Shield, LayoutDashboard, History, Settings, HelpCircle, Bell } from 'lucide-react';

export default function DashboardLayout() {
  const location = useLocation();
  const path = location.pathname;

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
            <div className="w-8 h-8 rounded-full bg-[#FFB084] overflow-hidden flex items-center justify-center border-2 border-[#121A15] shadow-sm cursor-pointer">
              {/* Profile icon placeholder */}
              <div className="w-4 h-4 bg-white/50 rounded-full mb-3"></div>
              <div className="w-6 h-6 bg-white/50 rounded-full mt-4 absolute"></div>
            </div>
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
