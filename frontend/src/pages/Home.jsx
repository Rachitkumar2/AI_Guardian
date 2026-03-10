import { ArrowRight, FileAudio, BrainCircuit, CheckCircle2, ShieldCheck, Database, FileCode2 } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="w-full">
      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-8 py-12 lg:py-20 grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#151E18] border border-[#1C2A22] text-neon-green text-xs font-bold tracking-wider mb-6">
            <span className="w-2 h-2 rounded-full bg-neon-green animate-pulse"></span>
            NEW: DEEPFAKE V4 DETECTION
          </div>
          <h1 className="text-5xl lg:text-7xl font-bold tracking-tight mb-6 leading-tight">
            Protect your identity from <span className="text-neon-green">AI voice clones</span>
          </h1>
          <p className="text-gray-400 text-lg mb-10 max-w-lg leading-relaxed">
            Our advanced detection engine identifies deepfake audio and synthetic voice clones in seconds. Maintain trust in your digital communication.
          </p>
          <div className="flex flex-col sm:flex-row items-center gap-4 mb-10">
            <Link to="/app" className="w-full sm:w-auto flex items-center justify-center gap-2 bg-neon-green text-black px-8 py-4 rounded-xl font-bold hover:bg-neon-green-hover transition-all neon-glow">
              Get Started <ArrowRight className="w-5 h-5" />
            </Link>
          </div>


        </div>

        {/* Hero Graphic */}
        <div className="relative">
          <div className="glass-panel p-8 h-[400px] relative overflow-hidden flex flex-col justify-center border-[#2A3F33]">
            {/* Warning tag */}
            <div className="absolute top-6 right-6 bg-red-500/10 border border-red-500/30 text-red-500 text-[10px] font-bold px-3 py-1 rounded uppercase tracking-wider">
              WARNING: CLONE DETECTED
            </div>
            
            {/* Audio Waveform visualization */}
            <div className="flex items-end justify-center gap-3 h-32 mb-12">
              {[40, 60, 45, 80, 100, 80, 45, 80, 40, 20].map((h, i) => (
                <div key={i} className="w-2 rounded-full bg-neon-green transition-all" style={{ height: `${h}%`, opacity: h / 100 + 0.2 }}></div>
              ))}
            </div>

            {/* Analysis card */}
            <div className="bg-[#0E1511] border border-[#1C2A22] rounded-xl p-4 absolute bottom-12 left-1/2 -translate-x-1/2 w-[80%]">
              <div className="flex items-start gap-4">
                <div className="p-2 bg-neon-green/10 rounded-lg">
                  <BrainCircuit className="w-5 h-5 text-neon-green" />
                </div>
                <div>
                  <div className="text-[10px] text-neon-green tracking-wider font-bold mb-1 uppercase">Scanning Audio</div>
                  <div className="font-semibold text-sm">Acoustic Score: 98.4% synthetic</div>
                </div>
              </div>
            </div>
            
            <p className="absolute bottom-4 left-0 right-0 text-center text-xs text-gray-500">Real-time Spectral Waveform Analysis</p>
          </div>
          
          {/* Blobs for background glow */}
          <div className="absolute -inset-4 bg-neon-green/20 blur-3xl -z-10 rounded-full opacity-30"></div>
        </div>
      </section>

      {/* 3 Step Process */}
      <section className="py-24 bg-[#121A15] border-y border-dark-border">
        <div className="max-w-7xl mx-auto px-8">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold mb-4">How AI Guardian Detects Fakes</h2>
            <p className="text-gray-400">Our proprietary neural networks look beyond the audible sound to find the microscopic digital signatures left by AI generative models.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="glass-panel p-8 hover:border-dark-border/80 transition-all group">
              <div className="w-12 h-12 rounded-xl bg-[#1C2A22] flex items-center justify-center mb-6 group-hover:bg-neon-green/10 transition-colors">
                <FileAudio className="w-6 h-6 text-neon-green" />
              </div>
              <h3 className="text-xl font-bold mb-3">1. Input Audio</h3>
              <p className="text-gray-400 text-sm leading-relaxed">Upload any audio file or stream live voice data through our secure API or dashboard.</p>
            </div>
            
            <div className="glass-panel p-8 hover:border-dark-border/80 transition-all group">
              <div className="w-12 h-12 rounded-xl bg-[#1C2A22] flex items-center justify-center mb-6 group-hover:bg-neon-green/10 transition-colors">
                <BrainCircuit className="w-6 h-6 text-neon-green" />
              </div>
              <h3 className="text-xl font-bold mb-3">2. AI Analysis</h3>
              <p className="text-gray-400 text-sm leading-relaxed">Our deep neural network analyzes over 184 acoustic features to differentiate human vocal patterns from AI-generated speech.</p>
            </div>

            <div className="glass-panel p-8 hover:border-dark-border/80 transition-all group">
              <div className="w-12 h-12 rounded-xl bg-[#1C2A22] flex items-center justify-center mb-6 group-hover:bg-neon-green/10 transition-colors">
                <CheckCircle2 className="w-6 h-6 text-neon-green" />
              </div>
              <h3 className="text-xl font-bold mb-3">3. Instant Verification</h3>
              <p className="text-gray-400 text-sm leading-relaxed">Receive a detailed authenticity report with probability scores and specific anomaly heatmaps within seconds.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Enterprise Features */}
      <section className="py-24 max-w-7xl mx-auto px-8 grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
        <div className="relative">
          <div className="glass-panel p-2 shadow-2xl relative z-10">
            <div className="bg-[#0E1511] rounded-lg border border-[#1C2A22] overflow-hidden">
               {/* Dashboard Mockup Header */}
               <div className="border-b border-[#1C2A22] p-4 flex gap-4">
                  <div className="w-32 h-6 bg-[#151E18] rounded"></div>
                  <div className="w-24 h-6 bg-[#151E18] rounded"></div>
               </div>
               {/* Dashboard Mockup Body */}
               <div className="p-6">
                 <div className="space-y-4 mb-8">
                   <div className="flex gap-4"><div className="w-full h-8 bg-[#151E18] rounded"></div></div>
                   <div className="flex gap-4"><div className="w-3/4 h-8 bg-[#151E18] rounded"></div></div>
                   <div className="flex gap-4"><div className="w-5/6 h-8 bg-[#151E18] rounded"></div></div>
                 </div>
                 <div className="border border-neon-green border-opacity-30 rounded p-4 relative">
                   <div className="text-[10px] text-neon-green font-bold tracking-widest mb-2 uppercase">Spectral Analysis</div>
                   <div className="h-1 w-full bg-[#151E18] rounded-full overflow-hidden">
                     <div className="h-full bg-neon-green w-[84%]"></div>
                   </div>
                   <div className="text-xs text-right mt-2 font-mono text-gray-400">Acoustic Score: 84% analyzed</div>
                 </div>
               </div>
            </div>
          </div>
          {/* Decorative element */}
          <div className="absolute top-10 -right-10 w-full h-full glass-panel border-[#1C2A22] -z-10 opacity-50 translate-x-4 translate-y-4"></div>
        </div>

        <div>
          <h2 className="text-3xl lg:text-4xl font-bold mb-8">Engineered for Enterprise Security</h2>
          
          <div className="space-y-8">
            <div className="flex gap-4">
              <div className="w-10 h-10 rounded-lg bg-neon-green/10 flex items-center justify-center shrink-0">
                <ShieldCheck className="w-5 h-5 text-neon-green" />
              </div>
              <div>
                <h4 className="text-lg font-bold mb-2">Real-time Detection</h4>
                <p className="text-gray-400 text-sm leading-relaxed">Integrate directly into call centers or meeting software to identify synthetic voices during live conversations.</p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="w-10 h-10 rounded-lg bg-neon-green/10 flex items-center justify-center shrink-0">
                <Database className="w-5 h-5 text-neon-green" />
              </div>
              <div>
                <h4 className="text-lg font-bold mb-2">Batch File Analysis</h4>
                <p className="text-gray-400 text-sm leading-relaxed">Securely process thousands of hours of historical recordings or evidence files with our high-throughput processing engine.</p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="w-10 h-10 rounded-lg bg-neon-green/10 flex items-center justify-center shrink-0">
                <FileCode2 className="w-5 h-5 text-neon-green" />
              </div>
              <div>
                <h4 className="text-lg font-bold mb-2">Robust API Access</h4>
                <p className="text-gray-400 text-sm leading-relaxed">Low-latency REST and WebSocket APIs designed for developers. Comprehensive documentation and SDKs for Python, Node, and Go.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="max-w-7xl mx-auto px-8 py-24">
        <div className="bg-neon-green rounded-3xl p-12 lg:p-20 text-center text-black shadow-[0_0_50px_rgba(0,255,102,0.2)]">
          <h2 className="text-4xl lg:text-5xl font-black tracking-tight mb-6">Ready to detect deepfakes?</h2>
          <p className="text-black/80 font-medium max-w-2xl mx-auto mb-10 text-lg">
            Start analyzing audio files now and protect yourself from AI-generated voice clones.
          </p>
          <div className="flex justify-center">
            <Link to="/app" className="bg-black text-white px-8 py-4 rounded-xl font-bold hover:bg-gray-900 transition-colors inline-block">
              Try It Now
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
