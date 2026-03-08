import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Shield, ArrowRight, User, Lock, Mail } from 'lucide-react';

export default function Signup() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSignup = (e) => {
    e.preventDefault();
    // Simulate signup and redirect to the dashboard
    navigate('/app');
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#0E1511]">
      <div className="absolute top-0 left-0 w-full p-8 flex justify-between items-center z-10">
        <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
          <Shield className="w-6 h-6 text-neon-green" fill="#00FF66" strokeWidth={1} />
          <span className="font-bold text-lg tracking-wide uppercase">AI Guardian</span>
        </Link>
      </div>

      <div className="flex-1 flex items-center justify-center p-8 relative overflow-hidden">
        {/* Background ambient glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-neon-green/10 rounded-full blur-[120px] -z-10 pointer-events-none"></div>

        <div className="glass-panel w-full max-w-md p-10 border-[#1C2A22] relative z-10 my-16">
          <div className="text-center mb-10">
            <h1 className="text-3xl font-bold mb-3 text-white">Create an Account</h1>
            <p className="text-gray-400 text-sm">Join the enterprise standard for audio authenticity</p>
          </div>

          <form onSubmit={handleSignup} className="space-y-6">
            <div>
              <label className="block text-sm font-semibold mb-2 text-gray-300">Full Name</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <User className="h-5 w-5 text-gray-500" />
                </div>
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full bg-[#121A15] border border-[#1C2A22] text-white rounded-xl pl-12 pr-4 py-4 focus:outline-none focus:border-neon-green/50 transition-colors"
                  placeholder="Jane Doe"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-semibold mb-2 text-gray-300">Email Address</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Mail className="h-5 w-5 text-gray-500" />
                </div>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-[#121A15] border border-[#1C2A22] text-white rounded-xl pl-12 pr-4 py-4 focus:outline-none focus:border-neon-green/50 transition-colors"
                  placeholder="you@company.com"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-semibold mb-2 text-gray-300">Password</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-500" />
                </div>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-[#121A15] border border-[#1C2A22] text-white rounded-xl pl-12 pr-4 py-4 focus:outline-none focus:border-neon-green/50 transition-colors"
                  placeholder="••••••••"
                />
              </div>
            </div>

            <button
              type="submit"
              className="w-full bg-neon-green text-black py-4 rounded-xl font-bold text-lg hover:bg-neon-green-hover transition-all shadow-[0_0_20px_rgba(0,255,102,0.2)] hover:shadow-[0_0_30px_rgba(0,255,102,0.4)] flex justify-center items-center gap-2"
            >
              Start Free Trial <ArrowRight className="w-5 h-5" />
            </button>
          </form>

          <p className="mt-8 text-center text-sm text-gray-400">
            Already have an account? <Link to="/login" className="text-neon-green font-semibold hover:underline">Log In</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
