import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Shield, ArrowRight, User, Lock, Eye, EyeOff, CheckCircle2 } from 'lucide-react';
import { useGoogleLogin } from '@react-oauth/google';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const navigate = useNavigate();

  const validateForm = () => {
    const newErrors = {};
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!email) {
      newErrors.email = 'Email is required';
    } else if (!emailRegex.test(email)) {
      newErrors.email = 'Please enter a valid email address';
    }

    if (!password) {
      newErrors.password = 'Password is required';
    } else if (password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');

    if (!validateForm()) return;

    setLoading(true);

    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ email, password }),
      });
      
      const data = await response.json();
      
      if (response.ok) {
        localStorage.setItem('user', JSON.stringify(data.user));
        if (data.token) {
          localStorage.setItem('token', data.token);
        }
        window.dispatchEvent(new Event('authChange'));
        setSuccess(true);
        setTimeout(() => navigate('/'), 1500);
      } else {
        setError(data.message || 'Invalid email or password');
      }
    } catch (err) {
      setError('Failed to connect to the server');
    } finally {
      setLoading(false);
    }
  };

  const loginWithGoogle = useGoogleLogin({
    onSuccess: async (tokenResponse) => {
      setError('');
      setLoading(true);
      
      try {
        const response = await fetch('/api/google-login', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
          body: JSON.stringify({ access_token: tokenResponse.access_token }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
          localStorage.setItem('user', JSON.stringify(data.user));
          if (data.token) {
            localStorage.setItem('token', data.token);
          }
          window.dispatchEvent(new Event('authChange'));
          setSuccess(true);
          setTimeout(() => navigate('/'), 1500);
        } else {
          setError(data.message || 'Google Login failed');
        }
      } catch (err) {
        setError('Failed to connect to the server');
      } finally {
        setLoading(false);
      }
    },
    onError: () => setError('Google Sign-In failed'),
  });

  return (
    <div className="min-h-screen flex flex-col bg-[#0E1511]">
      <div className="absolute top-0 left-0 w-full p-8 flex justify-between items-center z-10">
        <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
          <Shield className="w-6 h-6 text-neon-green" fill="#00FF66" strokeWidth={1} />
          <span className="font-bold text-lg tracking-wide uppercase">AI Guardian</span>
        </Link>
      </div>

      {/* Success Toast */}
      {success && (
        <div className="fixed top-6 inset-x-0 z-50 flex justify-center animate-slide-down">
          <div className="bg-[#1C2A22] border border-[#2A3F33] rounded-xl px-6 py-4 flex items-center gap-3 shadow-lg">
            <CheckCircle2 className="w-5 h-5 text-neon-green" />
            <span className="text-white font-semibold text-sm">Login successful! Redirecting...</span>
          </div>
        </div>
      )}

      <div className="flex-1 flex items-center justify-center p-8 relative overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-neon-green/10 rounded-full blur-[120px] -z-10 pointer-events-none"></div>

        <div className="glass-panel w-full max-w-md p-10 border-[#1C2A22] relative z-10">
          <div className="text-center mb-10">
            <h1 className="text-3xl font-bold mb-3 text-white">Welcome Back</h1>
            <p className="text-gray-400 text-sm">Secure your digital identity with AI Guardian</p>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-xl flex items-start gap-3 text-red-400">
              <div className="p-1"><Shield className="w-4 h-4" /></div>
              <p className="text-sm font-medium">{error}</p>
            </div>
          )}

          <form onSubmit={handleLogin} className="space-y-6">
            <div>
              <label className="block text-sm font-semibold mb-2 text-gray-300">Email Address</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <User className="h-5 w-5 text-gray-500" />
                </div>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => { setEmail(e.target.value); setErrors(prev => ({ ...prev, email: '' })); }}
                  className={`w-full bg-[#121A15] border ${errors.email ? 'border-red-500/50' : 'border-[#1C2A22]'} text-white rounded-xl pl-12 pr-4 py-4 focus:outline-none focus:border-neon-green/50 transition-colors`}
                  placeholder="you@company.com"
                />
              </div>
              {errors.email && <p className="text-red-400 text-xs mt-1.5 pl-1">{errors.email}</p>}
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="block text-sm font-semibold text-gray-300">Password</label>
                <a href="#" className="text-neon-green text-xs font-semibold hover:underline">Forgot password?</a>
              </div>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-500" />
                </div>
                <input
                  type={showPassword ? 'text' : 'password'}
                  required
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); setErrors(prev => ({ ...prev, password: '' })); }}
                  className={`w-full bg-[#121A15] border ${errors.password ? 'border-red-500/50' : 'border-[#1C2A22]'} text-white rounded-xl pl-12 pr-12 py-4 focus:outline-none focus:border-neon-green/50 transition-colors`}
                  placeholder="••••••••"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-500 hover:text-gray-300 transition-colors"
                >
                  {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
              {errors.password && <p className="text-red-400 text-xs mt-1.5 pl-1">{errors.password}</p>}
            </div>

            <button
              type="submit"
              disabled={loading || success}
              className="w-full bg-neon-green text-black py-4 rounded-xl font-bold text-lg hover:bg-neon-green-hover transition-all shadow-[0_0_20px_rgba(0,255,102,0.2)] hover:shadow-[0_0_30px_rgba(0,255,102,0.4)] flex justify-center items-center gap-2 disabled:opacity-70"
            >
              {loading ? 'Signing In...' : 'Sign In'} <ArrowRight className="w-5 h-5" />
            </button>
          </form>

          <div className="mt-8 flex items-center gap-4">
            <div className="flex-1 h-px bg-[#1C2A22]"></div>
            <div className="text-xs font-bold text-gray-500 uppercase tracking-widest">OR</div>
            <div className="flex-1 h-px bg-[#1C2A22]"></div>
          </div>

          <div className="mt-8 flex justify-center w-full">
            <button 
              type="button"
              onClick={() => loginWithGoogle()}
              className="w-full max-w-[400px] bg-[#121A15] border border-[#1C2A22] text-white py-3 md:py-4 rounded-xl font-semibold hover:bg-[#1C2A22] transition-colors flex justify-center items-center gap-3"
            >
              <svg viewBox="0 0 24 24" className="w-5 h-5" xmlns="http://www.w3.org/2000/svg"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/></svg>
              Continue with Google
            </button>
          </div>

          <p className="mt-10 text-center text-sm text-gray-400">
            Don't have an account? <Link to="/signup" className="text-neon-green font-semibold hover:underline">Sign up</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
