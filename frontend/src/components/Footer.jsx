import { Shield, Github, Linkedin, Mail } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="bg-[#121A15] border-t border-dark-border py-12 px-8 mt-20">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-8">
        <div className="md:col-span-1">
          <Link to="/" className="flex items-center gap-2 mb-4 hover:opacity-80 transition-opacity">
            <Shield className="w-6 h-6 text-neon-green" fill="#00FF66" strokeWidth={1} />
            <span className="font-bold text-lg tracking-wide uppercase">AI Guardian</span>
          </Link>
          <p className="text-gray-400 text-sm mb-4">
            Deepfake audio detection powered by machine learning.
          </p>
          <div className="flex gap-3">
            <a href="#" className="w-8 h-8 rounded-lg bg-[#1C2A22] flex items-center justify-center hover:bg-neon-green/20 transition-colors">
              <Github className="w-4 h-4 text-gray-400 hover:text-neon-green" />
            </a>
            <a href="#" className="w-8 h-8 rounded-lg bg-[#1C2A22] flex items-center justify-center hover:bg-neon-green/20 transition-colors">
              <Linkedin className="w-4 h-4 text-gray-400 hover:text-neon-green" />
            </a>
            <a href="#" className="w-8 h-8 rounded-lg bg-[#1C2A22] flex items-center justify-center hover:bg-neon-green/20 transition-colors">
              <Mail className="w-4 h-4 text-gray-400 hover:text-neon-green" />
            </a>
          </div>
        </div>
        
        <div>
          <h4 className="font-semibold mb-4 text-white">Navigation</h4>
          <ul className="space-y-3 text-sm text-gray-400">
            <li><Link to="/" className="hover:text-neon-green transition-colors">Home</Link></li>
            <li><Link to="/app" className="hover:text-neon-green transition-colors">Dashboard</Link></li>
            <li><Link to="/tools" className="hover:text-neon-green transition-colors">Tools</Link></li>
            <li><Link to="/library" className="hover:text-neon-green transition-colors">Library</Link></li>
          </ul>
        </div>
        
        <div>
          <h4 className="font-semibold mb-4 text-white">Features</h4>
          <ul className="space-y-3 text-sm text-gray-400">
            <li><span className="hover:text-neon-green transition-colors cursor-default">Audio Detection</span></li>
            <li><span className="hover:text-neon-green transition-colors cursor-default">Real-time Analysis</span></li>
            <li><span className="hover:text-neon-green transition-colors cursor-default">Waveform Visualization</span></li>
            <li><span className="hover:text-neon-green transition-colors cursor-default">Detection History</span></li>
          </ul>
        </div>

      </div>
      
      <div className="max-w-7xl mx-auto mt-12 pt-8 border-t border-dark-border flex flex-col md:flex-row items-center justify-between text-xs text-gray-500">
        <p>© 2026 AI Guardian. Built for Academic Purpose.</p>
        <p className="mt-2 md:mt-0">Deepfake Detection using Neural Networks</p>
      </div>
    </footer>
  );
}
