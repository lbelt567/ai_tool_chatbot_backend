'use client';

import { motion } from 'framer-motion';
import { Sparkles, ArrowRight } from 'lucide-react';
import ToolCard from './ToolCard';

interface PopularToolsProps {
  onAskAbout: (tool: string) => void;
}

export default function PopularTools({ onAskAbout }: PopularToolsProps) {
  // Sample popular tools from the database
  const popularTools = [
    {
      name: 'ChatGPT',
      description: 'Revolutionize interaction, creativity, and innovation with the leader in AI.',
      categories: ['productivity', 'search engine'],
      pricing: 'Freemium',
      homepage: 'https://chat.openai.com/chat',
      verified: true,
      favoriteCount: 2,
    },
    {
      name: 'Leonardo.AI',
      description: 'Leverage generative AI with a unique suite of tools to convey your ideas to the world.',
      categories: ['image generator', 'image to image'],
      pricing: 'Freemium',
      homepage: 'https://leonardo.ai/',
      verified: true,
      favoriteCount: 1,
    },
    {
      name: 'Eleven Labs',
      description: 'Revolutionize audio content with AI-driven, lifelike voice synthesis and customization.',
      categories: ['text to speech'],
      pricing: 'Freemium',
      homepage: 'http://elevenlabs.io/',
      verified: true,
      favoriteCount: 18,
    },
    {
      name: 'Jasper',
      description: 'An AI-driven platform for efficient, high-quality content creation and marketing strategy enhancement.',
      categories: ['copywriting'],
      pricing: 'Free Trial',
      homepage: 'https://jasper.ai',
      verified: true,
      favoriteCount: 1,
    },
    {
      name: 'Google Gemini',
      description: 'Multimodal reasoning across text, images, audio, and video.',
      categories: ['code assistant', 'writing generators', 'ai chatbots'],
      pricing: 'Freemium',
      homepage: 'https://gemini.google.com/app',
      verified: true,
      favoriteCount: 1,
    },
    {
      name: 'Notion AI',
      description: 'Enhance productivity with AI-driven content generation and analysis.',
      categories: ['general writing', 'productivity'],
      pricing: 'Freemium',
      homepage: 'https://www.notion.so/product/ai',
      verified: true,
      favoriteCount: 3,
    },
  ];

  return (
    <section id="browse" className="py-24 bg-dark-950 relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute bottom-0 left-1/4 w-96 h-96 bg-accent-cyan/5 rounded-full blur-3xl" />
        <div className="absolute top-1/4 right-0 w-96 h-96 bg-primary-500/5 rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent-cyan/10 border border-accent-cyan/20 text-accent-cyan text-sm font-medium mb-4">
            <Sparkles className="w-4 h-4" />
            Popular Tools
          </span>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-bold text-white mb-4">
            Trending <span className="text-gradient-alt">AI Tools</span>
          </h2>
          <p className="text-lg text-dark-300 max-w-2xl mx-auto">
            Discover the most popular AI tools loved by creators, developers, and businesses worldwide.
          </p>
        </motion.div>

        {/* Tools grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {popularTools.map((tool, index) => (
            <ToolCard
              key={tool.name}
              {...tool}
              index={index}
            />
          ))}
        </div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="text-center"
        >
          <button
            onClick={() => onAskAbout('What are the best free AI tools?')}
            className="inline-flex items-center gap-2 px-6 py-3 bg-dark-800/50 hover:bg-dark-800 border border-dark-700/50 hover:border-primary-500/30 text-white font-medium rounded-xl transition-all group"
          >
            <span>Explore all 3,800+ tools</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </button>
        </motion.div>
      </div>
    </section>
  );
}
