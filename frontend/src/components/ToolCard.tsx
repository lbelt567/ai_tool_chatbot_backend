'use client';

import { motion } from 'framer-motion';
import { ExternalLink, Star, Verified, Sparkles } from 'lucide-react';

interface ToolCardProps {
  name: string;
  description: string;
  categories: string[];
  pricing: string;
  homepage: string;
  verified?: boolean;
  favoriteCount?: number;
  index?: number;
}

export default function ToolCard({
  name,
  description,
  categories,
  pricing,
  homepage,
  verified = false,
  favoriteCount = 0,
  index = 0,
}: ToolCardProps) {
  const getPricingColor = (pricing: string) => {
    const lower = pricing.toLowerCase();
    if (lower === 'free') return 'bg-accent-emerald/20 text-accent-emerald border-accent-emerald/30';
    if (lower === 'freemium') return 'bg-accent-cyan/20 text-accent-cyan border-accent-cyan/30';
    if (lower.includes('trial')) return 'bg-accent-amber/20 text-accent-amber border-accent-amber/30';
    if (lower === 'paid') return 'bg-accent-rose/20 text-accent-rose border-accent-rose/30';
    return 'bg-dark-700/50 text-dark-300 border-dark-600';
  };

  const getCategoryIcon = (category: string) => {
    const icons: Record<string, string> = {
      'image generator': '🎨',
      'ai chatbots': '💬',
      'video editing': '🎬',
      'code assistant': '💻',
      'writing generators': '✍️',
      'productivity': '⚡',
      'marketing': '📈',
      'seo': '🔍',
      'transcriber': '🎙️',
      'education': '📚',
    };
    const lower = category.toLowerCase();
    for (const [key, icon] of Object.entries(icons)) {
      if (lower.includes(key)) return icon;
    }
    return '🤖';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
      className="group relative"
    >
      {/* Glow effect on hover */}
      <div className="absolute -inset-0.5 bg-gradient-to-r from-primary-500 to-accent-purple rounded-2xl opacity-0 group-hover:opacity-30 blur transition-opacity duration-300" />

      <div className="relative bg-dark-800/80 border border-dark-700/50 group-hover:border-primary-500/30 rounded-2xl p-6 transition-all duration-300 h-full flex flex-col">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-500/20 to-accent-purple/20 border border-primary-500/20 flex items-center justify-center text-2xl">
              {getCategoryIcon(categories[0] || '')}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-white group-hover:text-primary-300 transition-colors">
                  {name}
                </h3>
                {verified && (
                  <Verified className="w-4 h-4 text-primary-400" />
                )}
              </div>
              {favoriteCount > 0 && (
                <div className="flex items-center gap-1 text-xs text-dark-400">
                  <Star className="w-3 h-3 fill-accent-amber text-accent-amber" />
                  <span>{favoriteCount}</span>
                </div>
              )}
            </div>
          </div>
          <span
            className={`px-2.5 py-1 text-xs font-medium rounded-lg border ${getPricingColor(
              pricing
            )}`}
          >
            {pricing}
          </span>
        </div>

        {/* Description */}
        <p className="text-sm text-dark-300 mb-4 flex-1 line-clamp-3">
          {description}
        </p>

        {/* Categories */}
        <div className="flex flex-wrap gap-1.5 mb-4">
          {categories.slice(0, 3).map((category) => (
            <span
              key={category}
              className="px-2 py-1 text-xs text-dark-400 bg-dark-700/50 rounded-md"
            >
              {category}
            </span>
          ))}
          {categories.length > 3 && (
            <span className="px-2 py-1 text-xs text-dark-500">
              +{categories.length - 3} more
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 pt-4 border-t border-dark-700/50">
          <a
            href={homepage}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-dark-700/50 hover:bg-primary-500/20 text-dark-200 hover:text-primary-300 rounded-xl transition-all text-sm font-medium group/btn"
          >
            <span>Visit Tool</span>
            <ExternalLink className="w-4 h-4 group-hover/btn:translate-x-0.5 group-hover/btn:-translate-y-0.5 transition-transform" />
          </a>
        </div>
      </div>
    </motion.div>
  );
}

// Skeleton loader for tool cards
export function ToolCardSkeleton() {
  return (
    <div className="bg-dark-800/80 border border-dark-700/50 rounded-2xl p-6 animate-pulse">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-dark-700" />
          <div className="space-y-2">
            <div className="h-4 w-24 bg-dark-700 rounded" />
            <div className="h-3 w-16 bg-dark-700 rounded" />
          </div>
        </div>
        <div className="h-6 w-16 bg-dark-700 rounded-lg" />
      </div>
      <div className="space-y-2 mb-4">
        <div className="h-3 w-full bg-dark-700 rounded" />
        <div className="h-3 w-3/4 bg-dark-700 rounded" />
      </div>
      <div className="flex gap-1.5 mb-4">
        <div className="h-6 w-16 bg-dark-700 rounded-md" />
        <div className="h-6 w-20 bg-dark-700 rounded-md" />
      </div>
      <div className="h-10 w-full bg-dark-700 rounded-xl" />
    </div>
  );
}
