'use client';

import { motion } from 'framer-motion';
import { Sparkles, Zap, Shield, TrendingUp } from 'lucide-react';

interface HeroProps {
  onStartChat: () => void;
}

export default function Hero({ onStartChat }: HeroProps) {
  const stats = [
    { number: '3,800+', label: 'AI Tools', icon: Sparkles },
    { number: '50+', label: 'Categories', icon: Zap },
    { number: '100%', label: 'Free to Use', icon: Shield },
  ];

  const popularQueries = [
    'Generate marketing images',
    'Write blog content',
    'Build a chatbot',
    'Edit videos with AI',
    'Transcribe audio',
    'SEO optimization',
  ];

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden hero-bg">
      {/* Animated background orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      {/* Grid pattern overlay */}
      <div
        className="absolute inset-0 opacity-[0.02]"
        style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
          backgroundSize: '50px 50px',
        }}
      />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-32">
        <div className="text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-500/10 border border-primary-500/20 mb-8"
          >
            <Sparkles className="w-4 h-4 text-primary-400" />
            <span className="text-sm font-medium text-primary-300">
              Powered by AI • Updated Daily
            </span>
          </motion.div>

          {/* Main Heading */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-display font-bold tracking-tight mb-6"
          >
            <span className="text-white">Find Your Perfect</span>
            <br />
            <span className="text-gradient">AI Tool</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-lg sm:text-xl text-dark-300 max-w-2xl mx-auto mb-10"
          >
            Discover the best AI tools from{' '}
            <span className="text-white font-semibold">3,800+ curated options</span>.
            Get personalized recommendations powered by AI.
          </motion.p>

          {/* CTA Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mb-12"
          >
            <motion.button
              onClick={onStartChat}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="group relative px-8 py-4 bg-gradient-to-r from-primary-500 to-accent-purple text-white font-semibold text-lg rounded-2xl shadow-glow hover:shadow-glow-lg transition-all duration-300"
            >
              <span className="relative z-10 flex items-center gap-2">
                <Sparkles className="w-5 h-5" />
                Start Discovering
              </span>
              <div className="absolute inset-0 bg-gradient-to-r from-primary-600 to-purple-600 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity" />
            </motion.button>
          </motion.div>

          {/* Popular queries */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="mb-16"
          >
            <p className="text-sm text-dark-400 mb-3">Popular searches:</p>
            <div className="flex flex-wrap justify-center gap-2 max-w-2xl mx-auto">
              {popularQueries.map((query, index) => (
                <motion.button
                  key={query}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, delay: 0.5 + index * 0.05 }}
                  onClick={onStartChat}
                  className="px-4 py-2 text-sm text-dark-300 bg-dark-800/50 hover:bg-dark-700/50 border border-dark-700/50 hover:border-primary-500/30 rounded-full transition-all hover:text-white"
                >
                  {query}
                </motion.button>
              ))}
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            className="grid grid-cols-3 gap-4 sm:gap-8 max-w-xl mx-auto"
          >
            {stats.map((stat, index) => (
              <div key={stat.label} className="text-center">
                <div className="flex items-center justify-center mb-2">
                  <stat.icon className="w-5 h-5 text-primary-400 mr-2" />
                  <span className="text-2xl sm:text-3xl font-bold text-white">
                    {stat.number}
                  </span>
                </div>
                <span className="text-sm text-dark-400">{stat.label}</span>
              </div>
            ))}
          </motion.div>
        </div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 1 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-6 h-10 rounded-full border-2 border-dark-600 flex items-start justify-center p-2"
        >
          <div className="w-1.5 h-2.5 bg-dark-400 rounded-full" />
        </motion.div>
      </motion.div>
    </section>
  );
}
