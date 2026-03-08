import { Link, useLocation } from 'react-router-dom';
import { Shield } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();
  const path = location.pathname;

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
        <Link to="/login" className="text-sm font-medium text-gray-300 hover:text-white transition-colors">
          Log In
        </Link>
        <Link
          to="/app"
          className="bg-neon-green text-black px-5 py-2 rounded-md text-sm font-semibold hover:bg-neon-green-hover transition-colors neon-glow"
        >
          Try for free
        </Link>
      </div>
    </nav>
  );
}
