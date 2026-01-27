'use client';

import { motion } from 'framer-motion';
import {
  Image,
  MessageSquare,
  Video,
  Code,
  PenTool,
  Search,
  Mic,
  Music,
  FileText,
  Briefcase,
  ShoppingCart,
  GraduationCap,
  Sparkles,
} from 'lucide-react';

interface CategorySectionProps {
  onCategoryClick: (category: string) => void;
}

export default function CategorySection({ onCategoryClick }: CategorySectionProps) {
  const categories = [
    { name: 'Image Generation', icon: Image, query: 'AI image generator', color: 'from-pink-500 to-rose-500' },
    { name: 'Chatbots', icon: MessageSquare, query: 'AI chatbot', color: 'from-blue-500 to-cyan-500' },
    { name: 'Video Editing', icon: Video, query: 'AI video editing', color: 'from-purple-500 to-pink-500' },
    { name: 'Code Assistant', icon: Code, query: 'AI code assistant', color: 'from-green-500 to-emerald-500' },
    { name: 'Writing', icon: PenTool, query: 'AI writing assistant', color: 'from-amber-500 to-orange-500' },
    { name: 'SEO', icon: Search, query: 'AI SEO tool', color: 'from-indigo-500 to-purple-500' },
    { name: 'Transcription', icon: Mic, query: 'AI transcription', color: 'from-red-500 to-pink-500' },
    { name: 'Audio/Music', icon: Music, query: 'AI music generator', color: 'from-violet-500 to-purple-500' },
    { name: 'Summarizer', icon: FileText, query: 'AI summarizer', color: 'from-teal-500 to-cyan-500' },
    { name: 'Productivity', icon: Briefcase, query: 'AI productivity tool', color: 'from-blue-500 to-indigo-500' },
    { name: 'E-Commerce', icon: ShoppingCart, query: 'AI ecommerce tool', color: 'from-emerald-500 to-green-500' },
    { name: 'Education', icon: GraduationCap, query: 'AI education tool', color: 'from-orange-500 to-amber-500' },
  ];

  return (
    <section id="categories" className="py-24 bg-dark-900/50 relative overflow-hidden">
      {/* Background grid */}
      <div
        className="absolute inset-0 opacity-[0.02]"
        style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
          backgroundSize: '60px 60px',
        }}
      />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent-purple/10 border border-accent-purple/20 text-accent-purple text-sm font-medium mb-4">
            <Sparkles className="w-4 h-4" />
            Browse by Category
          </span>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-bold text-white mb-4">
            Explore <span className="text-gradient">50+ Categories</span>
          </h2>
          <p className="text-lg text-dark-300 max-w-2xl mx-auto">
            Find the perfect AI tool for any task. Click a category to get instant recommendations.
          </p>
        </motion.div>

        {/* Categories grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {categories.map((category, index) => (
            <motion.button
              key={category.name}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onCategoryClick(category.query)}
              className="group relative flex flex-col items-center gap-3 p-6 bg-dark-800/50 hover:bg-dark-800 border border-dark-700/50 hover:border-primary-500/30 rounded-2xl transition-all duration-300"
            >
              {/* Glow effect */}
              <div className={`absolute inset-0 bg-gradient-to-br ${category.color} opacity-0 group-hover:opacity-10 rounded-2xl transition-opacity duration-300`} />

              {/* Icon */}
              <div className={`relative p-3 rounded-xl bg-gradient-to-br ${category.color}`}>
                <category.icon className="w-6 h-6 text-white" />
              </div>

              {/* Name */}
              <span className="relative text-sm font-medium text-dark-200 group-hover:text-white text-center transition-colors">
                {category.name}
              </span>
            </motion.button>
          ))}
        </div>

        {/* View all link */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="text-center mt-8"
        >
          <button
            onClick={() => onCategoryClick('popular AI tools')}
            className="inline-flex items-center gap-2 px-6 py-3 text-primary-400 hover:text-primary-300 font-medium transition-colors"
          >
            <span>View all categories</span>
            <span className="text-lg">→</span>
          </button>
        </motion.div>
      </div>
    </section>
  );
}
