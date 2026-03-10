import { Radio, Database, Terminal, Fingerprint, Search, ArrowRight, History, Settings, Activity } from 'lucide-react';

export default function Tools() {
  const toolsList = [
    {
      title: 'Real-time Stream Monitor',
      icon: <Radio className="w-5 h-5 text-neon-green" />,
      description: 'Monitor live audio streams for synthetic voice patterns and deepfake injection attacks in real-time with sub-50ms latency.',
      tag: null,
      image: '/images/tools/stream_monitor.png'
    },
    {
      title: 'Batch File Analyzer',
      icon: <Database className="w-5 h-5 text-neon-green" />,
      description: 'Upload and analyze thousands of audio files simultaneously. Generates comprehensive forensic reports for legal and compliance needs.',
      tag: null,
      image: '/images/tools/batch_analyzer.png'
    },
    {
      title: 'API Playground',
      icon: <Terminal className="w-5 h-5 text-neon-green" />,
      description: 'Test our detection algorithms with custom parameters in a sandboxed environment. Integrate directly with your CI/CD pipelines.',
      tag: null,
      image: '/images/tools/api_playground.png'
    },
    {
      title: 'Voice Authenticator',
      icon: <Fingerprint className="w-5 h-5 text-neon-green" />,
      description: 'Verify identity using secure biometric voice authentication protocols. Zero-knowledge proof verification for sensitive transactions.',
      tag: null,
      image: '/images/tools/voice_authenticator.png'
    }
  ];

  return (
    <div className="w-full flex flex-col min-h-[calc(100vh-80px)] relative">
      <div className="flex-1 pb-24">
        {/* Header content */}
        <section className="max-w-7xl mx-auto px-8 py-16">
          <div className="flex flex-col md:flex-row md:items-start justify-between gap-8 mb-16">
            <div className="max-w-2xl">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#151E18] border border-[#1C2A22] text-neon-green text-xs font-bold tracking-wider mb-6">
                SECURITY SUITE
              </div>
              <h1 className="text-5xl lg:text-6xl font-bold tracking-tight mb-6">
                AI Voice <span className="text-neon-green">Detection Tools</span>
              </h1>
              <p className="text-gray-400 text-lg leading-relaxed">
                Secure your audio streams and digital assets with our suite of advanced AI guardian utilities. Built for enterprise-grade authentication.
              </p>
            </div>
          </div>

          {/* Tools Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {toolsList.map((tool, idx) => (
              <div key={idx} className="glass-panel overflow-hidden group hover:border-neon-green/30 transition-all flex flex-col p-0">
                
                {/* Tool Image */}
                <div className="h-40 w-full relative overflow-hidden flex items-center justify-center border-b border-[#1C2A22] bg-[#0E1511]">
                  <img src={tool.image} alt={tool.title} className="w-full h-full object-cover opacity-80 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500" />
                  {tool.tag && (
                    <div className="absolute top-4 right-4 bg-neon-green text-black text-[10px] font-bold px-2 py-1 rounded tracking-wider shadow-[0_0_10px_rgba(0,255,102,0.3)]">
                      {tool.tag}
                    </div>
                  )}
                </div>

                <div className="p-6 flex-1 flex flex-col">
                  <div className="flex items-center gap-3 mb-4">
                    {tool.icon}
                    <h3 className="text-lg font-bold text-white group-hover:text-neon-green transition-colors leading-tight">
                      {tool.title}
                    </h3>
                  </div>
                  <p className="text-gray-400 text-sm leading-relaxed mb-8 flex-1">
                    {tool.description}
                  </p>
                  
                  <button className="w-full bg-[#1C2A22] bg-opacity-50 text-neon-green border border-[#2A3F33] py-2 lg:py-3 rounded-lg flex items-center justify-center gap-2 text-sm font-bold group-hover:bg-neon-green/10 transition-colors">
                    Open Tool <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* SYSTEM STATUS BAR */}
      <div className="w-full bg-[#121A15] border-t border-[#1C2A22] py-4 px-8 mt-auto sticky bottom-0 z-10">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex flex-col sm:flex-row items-center gap-6">
            <h4 className="text-[10px] text-gray-400 font-bold uppercase tracking-widest">System Status</h4>
            <div className="flex items-center gap-4 text-xs font-medium">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-neon-green shadow-[0_0_8px_#00FF66]"></div>
                <span className="text-gray-300">Detection Engine: <span className="text-white">Online</span></span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-neon-green shadow-[0_0_8px_#00FF66]"></div>
                <span className="text-gray-300">API Latency: <span className="text-white">42ms</span></span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <button className="flex items-center gap-2 bg-[#1C2A22] hover:bg-[#2A3F33] text-gray-300 px-4 py-2 rounded-lg text-sm transition-colors border border-[#1C2A22]">
              <History className="w-4 h-4 text-neon-green" /> History
            </button>
            <button className="flex items-center gap-2 bg-[#1C2A22] hover:bg-[#2A3F33] text-gray-300 px-4 py-2 rounded-lg text-sm transition-colors border border-[#1C2A22]">
              <Settings className="w-4 h-4 text-neon-green" /> Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
