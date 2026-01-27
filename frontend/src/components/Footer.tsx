'use client';

import { motion } from 'framer-motion';
import { Sparkles, Github, Twitter, Linkedin, Mail, Heart } from 'lucide-react';

export default function Footer() {
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    product: [
      { name: 'Browse Tools', href: '#browse' },
      { name: 'Categories', href: '#categories' },
      { name: 'AI Chat', href: '#' },
      { name: 'Submit Tool', href: '#' },
    ],
    resources: [
      { name: 'Blog', href: '#' },
      { name: 'API Access', href: '#' },
      { name: 'Documentation', href: '#' },
      { name: 'Changelog', href: '#' },
    ],
    company: [
      { name: 'About Us', href: '#about' },
      { name: 'Contact', href: '#' },
      { name: 'Privacy Policy', href: '#' },
      { name: 'Terms of Service', href: '#' },
    ],
  };

  const socialLinks = [
    { name: 'Twitter', icon: Twitter, href: 'https://twitter.com' },
    { name: 'GitHub', icon: Github, href: 'https://github.com' },
    { name: 'LinkedIn', icon: Linkedin, href: 'https://linkedin.com' },
    { name: 'Email', icon: Mail, href: 'mailto:hello@aitoolrec.com' },
  ];

  return (
    <footer className="bg-dark-950 border-t border-dark-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main footer content */}
        <div className="py-12 md:py-16 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-8">
          {/* Brand column */}
          <div className="col-span-2 lg:col-span-2">
            <motion.a
              href="#"
              className="inline-flex items-center gap-2 group mb-4"
              whileHover={{ scale: 1.02 }}
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-purple flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <span className="font-display font-bold text-xl text-white">
                AI Tool<span className="text-primary-400">Rec</span>
              </span>
            </motion.a>
            <p className="text-dark-400 mb-6 max-w-sm">
              Discover the perfect AI tools for your needs. Our AI-powered
              recommendation engine searches through 3,800+ curated tools to find
              exactly what you need.
            </p>
            <div className="flex items-center gap-3">
              {socialLinks.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 text-dark-400 hover:text-white hover:bg-dark-800 rounded-lg transition-all"
                  aria-label={link.name}
                >
                  <link.icon className="w-5 h-5" />
                </a>
              ))}
            </div>
          </div>

          {/* Links columns */}
          <div>
            <h4 className="font-semibold text-white mb-4">Product</h4>
            <ul className="space-y-3">
              {footerLinks.product.map((link) => (
                <li key={link.name}>
                  <a
                    href={link.href}
                    className="text-dark-400 hover:text-white transition-colors"
                  >
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-white mb-4">Resources</h4>
            <ul className="space-y-3">
              {footerLinks.resources.map((link) => (
                <li key={link.name}>
                  <a
                    href={link.href}
                    className="text-dark-400 hover:text-white transition-colors"
                  >
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-white mb-4">Company</h4>
            <ul className="space-y-3">
              {footerLinks.company.map((link) => (
                <li key={link.name}>
                  <a
                    href={link.href}
                    className="text-dark-400 hover:text-white transition-colors"
                  >
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="py-6 border-t border-dark-800 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-sm text-dark-500">
            © {currentYear} AI Tool Recommender. All rights reserved.
          </p>
          <p className="text-sm text-dark-500 flex items-center gap-1">
            Made with <Heart className="w-4 h-4 text-accent-rose fill-accent-rose" /> for the AI community
          </p>
        </div>
      </div>
    </footer>
  );
}
