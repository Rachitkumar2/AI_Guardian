import { Shield, Github, Linkedin, Twitter, Mail, MapPin, ExternalLink } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Footer() {
  const currentYear = new Date().getFullYear();

  const quickLinks = [
    { name: 'Home', path: '/' },
    { name: 'Features', path: '/tools' },
    { name: 'How It Works', path: '/' },
    { name: 'Contact', path: '/library' },
  ];

  const supportLinks = [
    { name: 'Help Center', path: '#' },
    { name: 'Documentation', path: '#' },
    { name: 'API Reference', path: '#' },
    { name: 'Contact Us', path: '#' },
  ];

  return (
    <footer className="bg-[#0E1511] border-t border-dark-border/50 pt-16 pb-8 px-4 md:px-8 mt-20 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-neon-green/5 rounded-full blur-[100px] pointer-events-none"></div>
      
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 mb-16">
          {/* Column 1: Brand & Socials */}
          <div className="space-y-6">
            <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity group">
              <Shield className="w-7 h-7 text-neon-green group-hover:scale-110 transition-transform" fill="#00FF66" strokeWidth={1} />
              <span className="font-bold text-xl tracking-tight uppercase text-white">AI Guardian</span>
            </Link>
            <div className="flex gap-4">
              <a href="https://twitter.com" target="_blank" rel="noreferrer" className="w-10 h-10 rounded-full bg-[#151E18] border border-[#1C2A22] flex items-center justify-center hover:bg-neon-green/20 hover:border-neon-green/50 transition-all cursor-pointer group">
                <Twitter className="w-4 h-4 text-gray-400 group-hover:text-neon-green transition-colors" />
              </a>
              <a href="https://linkedin.com" target="_blank" rel="noreferrer" className="w-10 h-10 rounded-full bg-[#151E18] border border-[#1C2A22] flex items-center justify-center hover:bg-neon-green/20 hover:border-neon-green/50 transition-all cursor-pointer group">
                <Linkedin className="w-4 h-4 text-gray-400 group-hover:text-neon-green transition-colors" />
              </a>
              <a href="https://github.com" target="_blank" rel="noreferrer" className="w-10 h-10 rounded-full bg-[#151E18] border border-[#1C2A22] flex items-center justify-center hover:bg-neon-green/20 hover:border-neon-green/50 transition-all cursor-pointer group">
                <Github className="w-4 h-4 text-gray-400 group-hover:text-neon-green transition-colors" />
              </a>
            </div>
          </div>

          {/* Column 2: Quick Links */}
          <div className="lg:pl-8">
            <h4 className="text-white font-bold text-base mb-6 tracking-wide">Quick Links</h4>
            <ul className="space-y-4">
              {quickLinks.map((link) => (
                <li key={link.name}>
                  <Link 
                    to={link.path} 
                    className="text-gray-400 hover:text-neon-green text-sm transition-colors cursor-pointer flex items-center gap-1 group"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Column 3: Support */}
          <div>
            <h4 className="text-white font-bold text-base mb-6 tracking-wide">Support</h4>
            <ul className="space-y-4">
              {supportLinks.map((link) => (
                <li key={link.name}>
                  <Link 
                    to={link.path} 
                    className="text-gray-400 hover:text-neon-green text-sm transition-colors cursor-pointer"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Column 4: Contact */}
          <div className="space-y-6">
            <h4 className="text-white font-bold text-base mb-6 tracking-wide">Contact</h4>
            <div className="space-y-4">
              <div className="flex items-center gap-3 group">
                <div className="w-9 h-9 rounded-lg bg-[#151E18] border border-[#1C2A22] flex items-center justify-center group-hover:border-neon-green/30 transition-colors">
                  <Mail className="w-4 h-4 text-neon-green" />
                </div>
                <div className="text-sm">
                  <p className="text-gray-500 text-[10px] uppercase font-bold tracking-widest mb-0.5">Email Support</p>
                  <a href="mailto:support@aiguardian.com" className="text-gray-300 hover:text-neon-green transition-colors cursor-pointer">
                    support@aiguardian.com
                  </a>
                </div>
              </div>

              <div className="flex items-center gap-3 group">
                <div className="w-9 h-9 rounded-lg bg-[#151E18] border border-[#1C2A22] flex items-center justify-center group-hover:border-neon-green/30 transition-colors">
                  <MapPin className="w-4 h-4 text-neon-green" />
                </div>
                <div className="text-sm">
                  <p className="text-gray-500 text-[10px] uppercase font-bold tracking-widest mb-0.5">Location</p>
                  <p className="text-gray-300">Haryana, India</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-dark-border/40 flex flex-col md:flex-row items-center justify-between gap-6">
          <p className="text-gray-500 text-xs text-center md:text-left">
            © {currentYear} AI Guardian. All rights reserved 
            <span className="mx-2 hidden md:inline"></span> 
          </p>
          
          <div className="flex items-center gap-6 text-xs text-gray-500">
            <Link to="#" className="hover:text-neon-green transition-colors cursor-pointer">Privacy Policy</Link>
            <Link to="#" className="hover:text-neon-green transition-colors cursor-pointer">Terms of Service</Link>
            <Link to="#" className="hover:text-neon-green transition-colors cursor-pointer">Cookies</Link>
          </div>
        </div>
      </div>
    </footer>
  );
}

