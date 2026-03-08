import { Shield, Share2, Globe } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="bg-[#121A15] border-t border-dark-border py-12 px-8 mt-20">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-5 gap-8">
        <div className="md:col-span-2">
          <Link to="/" className="flex items-center gap-2 mb-4 hover:opacity-80 transition-opacity">
            <Shield className="w-6 h-6 text-neon-green" fill="#00FF66" strokeWidth={1} />
            <span className="font-bold text-lg tracking-wide uppercase">AI Guardian</span>
          </Link>
          <p className="text-gray-400 text-sm max-w-xs mb-6">
            © 2024 AI Guardian Inc. All rights reserved.
          </p>
        </div>
        
        <div>
          <h4 className="font-semibold mb-4 text-white hover:text-neon-green transition-colors cursor-pointer">Product</h4>
          <ul className="space-y-3 text-sm text-gray-400">
            <li><a href="#" className="hover:text-neon-green transition-colors">Detection Engine</a></li>
            <li><a href="#" className="hover:text-neon-green transition-colors">API Docs</a></li>
            <li><a href="#" className="hover:text-neon-green transition-colors">Pricing</a></li>
            <li><a href="#" className="hover:text-neon-green transition-colors">Enterprise</a></li>
          </ul>
        </div>
        
        <div>
          <h4 className="font-semibold mb-4 text-white hover:text-neon-green transition-colors cursor-pointer">Company</h4>
          <ul className="space-y-3 text-sm text-gray-400">
            <li><a href="#" className="hover:text-neon-green transition-colors">About Us</a></li>
            <li><a href="#" className="hover:text-neon-green transition-colors">Careers</a></li>
            <li><a href="#" className="hover:text-neon-green transition-colors">Privacy Policy</a></li>
            <li><a href="#" className="hover:text-neon-green transition-colors">Security</a></li>
          </ul>
        </div>

        <div>
          <h4 className="font-semibold mb-4 text-white hover:text-neon-green transition-colors cursor-pointer">Resources</h4>
          <ul className="space-y-3 text-sm text-gray-400">
            <li><a href="#" className="hover:text-neon-green transition-colors">Blog</a></li>
            <li><a href="#" className="hover:text-neon-green transition-colors">Deepfake Report</a></li>
            <li><a href="#" className="hover:text-neon-green transition-colors">Case Studies</a></li>
            <li><a href="#" className="hover:text-neon-green transition-colors">Support</a></li>
          </ul>
        </div>
      </div>
      
      <div className="max-w-7xl mx-auto mt-12 pt-8 border-t border-dark-border flex flex-col md:flex-row items-center justify-between text-xs text-gray-500">
        <p>© 2024 AI Guardian Inc. All rights reserved.</p>
        <div className="flex gap-4 mt-4 md:mt-0">
          <a href="#" className="hover:text-neon-green">Terms of Service</a>
          <a href="#" className="hover:text-neon-green">Cookie Policy</a>
        </div>
      </div>
    </footer>
  );
}
