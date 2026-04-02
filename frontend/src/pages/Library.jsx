import { Search, ArrowRight } from 'lucide-react';

export default function Library() {
  const topics = ['All Topics', 'Basics', 'Scam Prevention', 'Detection Tech', 'Case Studies', 'Legal Framework'];
  
  const articles = [
    {
      category: 'BASICS',
      title: 'What is a Deepfake?',
      description: 'An introductory guide to how AI synthesizes human voices and the difference between basic TTS and advanced neural cloning.',
      image: '/images/library/deepfake_intro.png'
    },
    {
      category: 'SCAM PREVENTION',
      title: 'How to Spot AI Voice Scams',
      description: 'Learn the physiological cues of artificial speech, including unnatural breathing patterns and odd linguistic artifacts.',
      image: '/images/library/scam_spotting.png'
    },
    {
      category: 'TECHNOLOGY',
      title: 'Behind the Detection',
      description: 'Exploring the algorithms used to analyze frequency consistency and phase discrepancies that reveal voice manipulation.',
      image: '/images/library/detection_tech.png'
    },
    {
      category: 'CASE STUDIES',
      title: 'The CEO Voice Cloning Case',
      description: 'A breakdown of the first major documented voice cloning heist and how it could have been prevented using VoiceGuard.',
      image: '/images/library/ceo_case.png'
    },
    {
      category: 'PROTECTION',
      title: 'Family Safety Protocols',
      description: 'Simple steps your family can take today, including "safe words" and verification steps for urgent phone calls.',
      image: '/images/library/family_safety.png'
    },
    {
      category: 'LEGAL',
      title: 'Legal Rights in the AI Era',
      description: 'Understanding current laws regarding voice likeness, biometric data privacy, and the prosecution of AI fraud.',
      image: '/images/library/legal_rights.png'
    }
  ];

  return (
    <div className="w-full">
      {/* Header Section */}
      <section className="max-w-7xl mx-auto px-4 md:px-8 py-12 md:py-16">
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-6">
          Educational <span className="text-neon-green">Resources</span>
        </h1>
        <p className="text-gray-400 text-lg max-w-2xl mb-12 leading-relaxed">
          Stay ahead of emerging threats. Learn how to identify, analyze, and protect yourself from AI-generated voice scams and deepfake technology.
        </p>

        {/* Search Bar */}
        <div className="relative mb-8 max-w-4xl">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-gray-500" />
          </div>
          <input
            type="text"
            className="w-full bg-[#151E18] border border-[#1C2A22] text-white rounded-xl pl-12 pr-4 py-4 focus:outline-none focus:border-neon-green/50 transition-colors"
            placeholder="Search articles, guides, or technology terms..."
          />
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-3 mb-16">
          {topics.map((topic, index) => (
            <button
              key={topic}
              className={`px-5 py-2 rounded-full text-sm font-medium transition-colors ${
                index === 0
                  ? 'bg-neon-green text-black'
                  : 'bg-[#151E18] text-gray-300 hover:bg-[#1C2A22] border border-[#1C2A22]'
              }`}
            >
              {topic}
            </button>
          ))}
        </div>

        {/* Grid of Articles */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {articles.map((article, idx) => (
            <div key={idx} className="glass-panel overflow-hidden group hover:border-dark-border/80 transition-all flex flex-col">
              {/* Cover Image */}
              <div className="h-48 w-full relative overflow-hidden bg-[#0E1511]">
                <img 
                  src={article.image} 
                  alt={article.title} 
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#0E1511] via-[#0E1511]/40 to-transparent opacity-80 group-hover:opacity-60 transition-opacity"></div>
              </div>
              
              <div className="p-8 flex-1 flex flex-col">
                <div className="text-[10px] font-bold text-neon-green tracking-wider uppercase mb-3 text-shadow-sm">
                  {article.category}
                </div>
                <h3 className="text-xl font-bold mb-3 text-white group-hover:text-neon-green transition-colors">
                  {article.title}
                </h3>
                <p className="text-gray-400 text-sm leading-relaxed mb-6 flex-1">
                  {article.description}
                </p>
                <button className="flex items-center gap-2 text-neon-green text-sm font-bold group/btn self-start">
                  Read Article <ArrowRight className="w-4 h-4 group-hover/btn:translate-x-1 transition-transform" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}
