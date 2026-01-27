'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, X } from 'lucide-react';

interface FloatingChatButtonProps {
  isOpen: boolean;
  onClick: () => void;
}

export default function FloatingChatButton({
  isOpen,
  onClick,
}: FloatingChatButtonProps) {
  return (
    <motion.button
      onClick={onClick}
      className="fixed bottom-6 right-6 z-40 p-4 bg-gradient-to-r from-primary-500 to-accent-purple text-white rounded-full shadow-lg hover:shadow-glow transition-shadow"
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ delay: 1, type: 'spring', stiffness: 200 }}
    >
      <AnimatePresence mode="wait">
        {isOpen ? (
          <motion.div
            key="close"
            initial={{ rotate: -90, opacity: 0 }}
            animate={{ rotate: 0, opacity: 1 }}
            exit={{ rotate: 90, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <X className="w-6 h-6" />
          </motion.div>
        ) : (
          <motion.div
            key="chat"
            initial={{ rotate: 90, opacity: 0 }}
            animate={{ rotate: 0, opacity: 1 }}
            exit={{ rotate: -90, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <MessageSquare className="w-6 h-6" />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Ping animation when closed */}
      {!isOpen && (
        <span className="absolute top-0 right-0 flex h-3 w-3">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-emerald opacity-75"></span>
          <span className="relative inline-flex rounded-full h-3 w-3 bg-accent-emerald"></span>
        </span>
      )}
    </motion.button>
  );
}
