'use client';

import { motion } from 'framer-motion';
import {
  Brain,
  Zap,
  DollarSign,
  MessageSquare,
  Search,
  Sparkles,
} from 'lucide-react';

export default function Features() {
  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Search',
      description:
        'Our semantic search understands what you need, not just what you type. Describe your problem and get relevant results.',
      gradient: 'from-primary-500 to-accent-purple',
    },
    {
      icon: DollarSign,
      title: 'Free Alternatives',
      description:
        'We highlight free and freemium options first. Save money while getting the same capabilities.',
      gradient: 'from-accent-emerald to-accent-cyan',
    },
    {
      icon: MessageSquare,
      title: 'Personalized Advice',
      description:
        'Get tailored recommendations with explanations of why each tool fits your specific use case.',
      gradient: 'from-accent-amber to-accent-rose',
    },
    {
      icon: Zap,
      title: '3,800+ Curated Tools',
      description:
        'Every tool is verified and categorized. No spam, no dead links, just quality recommendations.',
      gradient: 'from-accent-cyan to-primary-500',
    },
    {
      icon: Search,
      title: 'Smart Comparisons',
      description:
        'When multiple tools fit your needs, we compare them so you can make an informed decision.',
      gradient: 'from-accent-purple to-accent-rose',
    },
    {
      icon: Sparkles,
      title: 'Updated Daily',
      description:
        'New AI tools launch every day. Our database stays current so you never miss the latest innovations.',
      gradient: 'from-primary-500 to-accent-emerald',
    },
  ];

  return (
    <section id="about" className="py-24 bg-dark-950 relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-0 w-96 h-96 bg-primary-500/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-0 w-96 h-96 bg-accent-purple/5 rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-500/10 border border-primary-500/20 text-primary-300 text-sm font-medium mb-4">
            <Sparkles className="w-4 h-4" />
            Why Choose Us
          </span>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-bold text-white mb-4">
            Smarter Way to Find{' '}
            <span className="text-gradient">AI Tools</span>
          </h2>
          <p className="text-lg text-dark-300 max-w-2xl mx-auto">
            Stop wasting hours searching through hundreds of AI tools. Our
            intelligent recommendation engine does the work for you.
          </p>
        </motion.div>

        {/* Features grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="group relative"
            >
              <div className="absolute -inset-0.5 bg-gradient-to-r opacity-0 group-hover:opacity-30 blur rounded-2xl transition-opacity duration-300"
                style={{
                  backgroundImage: `linear-gradient(to right, var(--tw-gradient-stops))`,
                }}
              />
              <div className="relative h-full bg-dark-800/50 border border-dark-700/50 group-hover:border-dark-600 rounded-2xl p-6 transition-all duration-300">
                {/* Icon */}
                <div
                  className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${feature.gradient} mb-4`}
                >
                  <feature.icon className="w-6 h-6 text-white" />
                </div>

                {/* Content */}
                <h3 className="text-xl font-semibold text-white mb-2">
                  {feature.title}
                </h3>
                <p className="text-dark-300 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
