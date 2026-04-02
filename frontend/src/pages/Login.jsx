import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Shield, ArrowRight, User, Lock, Eye, EyeOff, CheckCircle2, AudioLines, Activity, BarChart3, ShieldCheck } from 'lucide-react';
import { useGoogleLogin } from '@react-oauth/google';

const features = [
  { icon: AudioLines, title: 'Audio Detection', desc: 'Identify deepfake audio with high accuracy' },
  { icon: Activity, title: 'Real-time Analysis', desc: 'Instant results powered by neural networks' },
  { icon: BarChart3, title: 'Waveform Visualization', desc: 'Detailed visual breakdown of audio signals' },
  { icon: ShieldCheck, title: 'Secure & Private', desc: 'Your data stays protected at all times' },
];

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [emailFocus, setEmailFocus] = useState(false);
  const navigate = useNavigate();

  // Custom email validation for inline feedback
  const isValidEmail = (email) => {
    if (!email) return true; // Don't show error when empty string
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  };
  const isEmailInvalid = email.length > 0 && !isValidEmail(email) && !emailFocus;

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

    if (!isValidEmail(email)) {
      setErrors({ ...errors, email: 'Please enter a valid email address.' });
      setLoading(false);
      return;
    }

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
    <div className="min-h-screen flex bg-[#0E1511]">
      {/* Success Toast */}
      {success && (
        <div className="fixed top-6 inset-x-0 z-50 flex justify-center animate-slide-down">
          <div className="bg-[#1C2A22] border border-[#2A3F33] rounded-xl px-6 py-4 flex items-center gap-3 shadow-lg">
            <CheckCircle2 className="w-5 h-5 text-neon-green" />
            <span className="text-white font-semibold text-sm">Logged in successfully</span>
          </div>
        </div>
      )}

      {/* Left Panel - Feature Showcase */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden">
        {/* Animated gradient background */}
        <div className="absolute inset-0 bg-gradient-to-br from-[#0a1f0f] via-[#0E1511] to-[#071a0c] animate-gradient"></div>

        {/* Decorative elements */}
        <div className="absolute top-20 left-10 w-72 h-72 bg-neon-green/5 rounded-full blur-[100px]"></div>
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-neon-green/8 rounded-full blur-[120px]"></div>

        {/* Grid pattern overlay */}
        <div className="absolute inset-0 opacity-5"
          style={{
            backgroundImage: 'linear-gradient(rgba(0,255,102,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,102,0.3) 1px, transparent 1px)',
            backgroundSize: '60px 60px'
          }}
        ></div>

        <div className="relative z-10 flex flex-col justify-center px-12 py-8 w-full">
          {/* Logo */}
          <div>
            <Link to="/" className="flex items-center gap-2 mb-8 hover:opacity-80 transition-opacity">
              <Shield className="w-8 h-8 text-neon-green" fill="#00FF66" strokeWidth={1} />
              <span className="font-bold text-xl tracking-wide uppercase">AI Guardian</span>
            </Link>
          </div>

          {/* Headline */}
          <div className="mb-8">
            <h2 className="text-4xl font-bold text-white mb-4 leading-tight">
              Protect Against<br />
              <span className="text-neon-green">Deepfake Audio</span>
            </h2>
            <p className="text-gray-400 text-lg max-w-md">
              Advanced neural network detection to verify audio authenticity in real-time.
            </p>
          </div>

          {/* Feature Cards */}
          <div className="space-y-3">
            {features.map((feature, i) => (
              <div
                key={feature.title}
                className="flex items-start gap-4 p-4 rounded-xl bg-white/[0.03] border border-white/[0.05] backdrop-blur-sm hover:bg-white/[0.06] hover:border-neon-green/20 transition-all duration-300"
              >
                <div className="w-10 h-10 rounded-lg bg-neon-green/10 flex items-center justify-center shrink-0">
                  <feature.icon className="w-5 h-5 text-neon-green" />
                </div>
                <div>
                  <h3 className="font-semibold text-white text-sm">{feature.title}</h3>
                  <p className="text-gray-500 text-xs mt-0.5">{feature.desc}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Floating badge */}
          <div className="mt-8">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-neon-green/10 border border-neon-green/20">
              <div className="w-2 h-2 rounded-full bg-neon-green animate-pulse"></div>
              <span className="text-neon-green text-xs font-semibold">AI-Powered Detection Active</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel - Login Form */}
      <div className="w-full lg:w-1/2 flex flex-col min-h-screen lg:min-h-0">
        {/* Mobile-only logo */}
        <div className="lg:hidden p-5 sm:p-8">
          <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <Shield className="w-6 h-6 text-neon-green" fill="#00FF66" strokeWidth={1} />
            <span className="font-bold text-lg tracking-wide uppercase">AI Guardian</span>
          </Link>
        </div>

        <div className="flex-1 flex items-center justify-center p-4 sm:p-6 relative overflow-y-auto w-full max-w-[500px] mx-auto">
          {/* Subtle background glow */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-neon-green/5 rounded-full blur-[120px] pointer-events-none"></div>

          <div className="w-full max-w-md relative z-10">
            <div>
              <div className="text-center mb-6 sm:mb-8">
                <h1 className="text-2xl sm:text-3xl font-bold mb-3 text-white">Welcome Back</h1>
                <p className="text-gray-400 text-sm">Sign in to continue to AI Guardian</p>
              </div>
            </div>

            {error && (
              <div className="mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-xl flex items-start gap-3 text-red-400 animate-slide-down">
                <div className="p-1"><Shield className="w-4 h-4" /></div>
                <p className="text-sm font-medium">{error}</p>
              </div>
            )}

            <form onSubmit={handleLogin} className="space-y-4" noValidate>
              <div>
                <label className="block text-sm font-semibold mb-2 text-gray-300">Email Address</label>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <User className={`h-5 w-5 transition-colors ${isEmailInvalid ? 'text-red-500' : 'text-gray-500 group-focus-within:text-neon-green'}`} />
                  </div>
                  <input
                    type="email"
                    required
                    value={email}
                    onFocus={() => setEmailFocus(true)}
                    onBlur={() => setEmailFocus(false)}
                    onChange={(e) => { setEmail(e.target.value); setErrors(prev => ({ ...prev, email: '' })); }}
                    className={`w-full bg-[#121A15] border ${isEmailInvalid || errors.email ? 'border-red-500 focus:border-red-500 focus:shadow-[0_0_0_3px_rgba(239,68,68,0.15)]' : 'border-[#1C2A22] focus:border-neon-green/50 focus:shadow-[0_0_0_3px_rgba(0,255,102,0.08)]'} text-white rounded-xl pl-11 pr-4 py-3 sm:py-3.5 text-sm focus:outline-none transition-all`}
                    placeholder="you@company.com"
                  />
                </div>
                {(isEmailInvalid || errors.email) && (
                  <p className="text-red-500 text-xs mt-1.5 pl-1 font-medium select-none">
                    {errors.email || "Please enter a valid email address."}
                  </p>
                )}
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="block text-sm font-semibold text-gray-300">Password</label>
                  <a href="#" className="text-neon-green text-xs font-semibold hover:underline">Forgot password?</a>
                </div>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <Lock className="h-5 w-5 text-gray-500 group-focus-within:text-neon-green transition-colors" />
                  </div>
                  <input
                    type={showPassword ? 'text' : 'password'}
                    required
                    value={password}
                    onChange={(e) => { setPassword(e.target.value); setErrors(prev => ({ ...prev, password: '' })); }}
                    className={`w-full bg-[#121A15] border ${errors.password ? 'border-red-500/50' : 'border-[#1C2A22]'} text-white rounded-xl pl-11 pr-12 py-3 sm:py-3.5 text-sm focus:outline-none focus:border-neon-green/50 focus:shadow-[0_0_0_3px_rgba(0,255,102,0.08)] transition-all`}
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

              <div>
                <button
                  type="submit"
                  disabled={loading || success}
                  className="w-full bg-neon-green text-black py-3.5 sm:py-4 rounded-xl font-bold text-base sm:text-lg hover:bg-neon-green-hover transition-all shadow-[0_0_20px_rgba(0,255,102,0.2)] hover:shadow-[0_0_30px_rgba(0,255,102,0.4)] flex justify-center items-center gap-2 disabled:opacity-70 mt-2"
                >
                  {loading ? 'Signing In...' : 'Sign In'} <ArrowRight className="w-5 h-5" />
                </button>
              </div>
            </form>

            <div>
              <div className="mt-8 flex items-center gap-4">
                <div className="flex-1 h-px bg-[#1C2A22]"></div>
                <div className="text-xs font-bold text-gray-500 uppercase tracking-widest">OR</div>
                <div className="flex-1 h-px bg-[#1C2A22]"></div>
              </div>

              <div className="mt-6 flex justify-center w-full">
                <button
                  type="button"
                  onClick={() => loginWithGoogle()}
                  className="w-full bg-[#121A15] border border-[#1C2A22] text-white py-3 sm:py-3.5 rounded-xl font-semibold hover:bg-[#1C2A22] hover:border-[#2A3F33] transition-all flex justify-center items-center gap-3"
                >
                  <svg viewBox="0 0 24 24" className="w-5 h-5" xmlns="http://www.w3.org/2000/svg"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" /><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" /><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" /><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" /></svg>
                  Continue with Google
                </button>
              </div>

              <p className="mt-6 text-center text-sm text-gray-400">
                Don't have an account? <Link to="/signup" className="text-neon-green font-semibold hover:underline">Sign up</Link>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
