'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Bot,
  User,
  Sparkles,
  ExternalLink,
  Copy,
  Check,
  RefreshCw,
  X,
  Maximize2,
  Minimize2,
} from 'lucide-react';
import { Message } from '@/types';
import { sendChatMessage, formatResponseWithLinks } from '@/lib/api';

interface ChatInterfaceProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ChatInterface({ isOpen, onClose }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Add welcome message when chat opens
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([
        {
          id: 'welcome',
          role: 'assistant',
          content:
            "Hi! I'm your AI Tool advisor. Tell me what you're trying to accomplish, and I'll recommend the perfect tools for you. Whether you need help with image generation, writing, coding, or anything else - I've got you covered!",
          timestamp: new Date(),
        },
      ]);
    }
  }, [isOpen, messages.length]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await sendChatMessage(userMessage.content);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.error || response.response,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content:
          "I'm sorry, I encountered an error. Please try again in a moment.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const copyToClipboard = async (text: string, id: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const clearChat = () => {
    setMessages([
      {
        id: 'welcome-new',
        role: 'assistant',
        content:
          "Chat cleared! What AI tools are you looking for today?",
        timestamp: new Date(),
      },
    ]);
  };

  const suggestedQueries = [
    "I need a free AI image generator",
    "Best tool for writing blog posts",
    "AI for video editing",
    "Help me build a chatbot",
  ];

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-dark-950/80 backdrop-blur-sm"
        onClick={(e) => e.target === e.currentTarget && onClose()}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          transition={{ duration: 0.3, ease: 'easeOut' }}
          className={`relative w-full bg-dark-900 border border-dark-700/50 rounded-2xl shadow-2xl overflow-hidden flex flex-col transition-all duration-300 ${
            isExpanded
              ? 'max-w-6xl h-[90vh]'
              : 'max-w-2xl h-[80vh] md:h-[700px]'
          }`}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-dark-700/50 bg-dark-800/50">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-purple flex items-center justify-center">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-accent-emerald rounded-full border-2 border-dark-800" />
              </div>
              <div>
                <h2 className="font-semibold text-white">AI Tool Advisor</h2>
                <p className="text-xs text-dark-400">
                  Powered by 3,800+ curated tools
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={clearChat}
                className="p-2 text-dark-400 hover:text-white hover:bg-dark-700/50 rounded-lg transition-colors"
                title="Clear chat"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="p-2 text-dark-400 hover:text-white hover:bg-dark-700/50 rounded-lg transition-colors hidden md:block"
                title={isExpanded ? 'Minimize' : 'Expand'}
              >
                {isExpanded ? (
                  <Minimize2 className="w-4 h-4" />
                ) : (
                  <Maximize2 className="w-4 h-4" />
                )}
              </button>
              <button
                onClick={onClose}
                className="p-2 text-dark-400 hover:text-white hover:bg-dark-700/50 rounded-lg transition-colors"
                title="Close"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className={`flex gap-3 ${
                  message.role === 'user' ? 'flex-row-reverse' : ''
                }`}
              >
                {/* Avatar */}
                <div
                  className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
                    message.role === 'user'
                      ? 'bg-primary-500'
                      : 'bg-gradient-to-br from-primary-500/20 to-accent-purple/20 border border-primary-500/30'
                  }`}
                >
                  {message.role === 'user' ? (
                    <User className="w-4 h-4 text-white" />
                  ) : (
                    <Sparkles className="w-4 h-4 text-primary-400" />
                  )}
                </div>

                {/* Message content */}
                <div
                  className={`flex-1 max-w-[85%] ${
                    message.role === 'user' ? 'text-right' : ''
                  }`}
                >
                  <div
                    className={`inline-block px-4 py-3 rounded-2xl ${
                      message.role === 'user'
                        ? 'bg-primary-500 text-white rounded-tr-md'
                        : 'bg-dark-800/80 border border-dark-700/50 text-dark-100 rounded-tl-md'
                    }`}
                  >
                    {message.role === 'assistant' ? (
                      <div
                        className="prose-chat"
                        dangerouslySetInnerHTML={{
                          __html: formatResponseWithLinks(message.content),
                        }}
                      />
                    ) : (
                      <p>{message.content}</p>
                    )}
                  </div>

                  {/* Message actions */}
                  {message.role === 'assistant' && message.id !== 'welcome' && (
                    <div className="flex items-center gap-2 mt-2">
                      <button
                        onClick={() =>
                          copyToClipboard(message.content, message.id)
                        }
                        className="flex items-center gap-1 px-2 py-1 text-xs text-dark-400 hover:text-white hover:bg-dark-800/50 rounded transition-colors"
                      >
                        {copiedId === message.id ? (
                          <>
                            <Check className="w-3 h-3 text-accent-emerald" />
                            <span className="text-accent-emerald">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" />
                            <span>Copy</span>
                          </>
                        )}
                      </button>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}

            {/* Loading indicator */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex gap-3"
              >
                <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500/20 to-accent-purple/20 border border-primary-500/30 flex items-center justify-center">
                  <Sparkles className="w-4 h-4 text-primary-400" />
                </div>
                <div className="px-4 py-3 rounded-2xl rounded-tl-md bg-dark-800/80 border border-dark-700/50">
                  <div className="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Suggested queries for empty state */}
            {messages.length === 1 && !isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="mt-4"
              >
                <p className="text-sm text-dark-400 mb-3">Try asking:</p>
                <div className="flex flex-wrap gap-2">
                  {suggestedQueries.map((query) => (
                    <button
                      key={query}
                      onClick={() => {
                        setInput(query);
                        inputRef.current?.focus();
                      }}
                      className="px-3 py-2 text-sm text-dark-300 bg-dark-800/50 hover:bg-dark-700/50 border border-dark-700/50 hover:border-primary-500/30 rounded-xl transition-all hover:text-white"
                    >
                      {query}
                    </button>
                  ))}
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div className="p-4 border-t border-dark-700/50 bg-dark-800/30">
            <form onSubmit={handleSubmit} className="relative">
              <div className="relative flex items-end gap-2 p-2 bg-dark-800 border border-dark-700/50 rounded-xl focus-within:border-primary-500/50 focus-within:shadow-glow transition-all">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Describe what you need help with..."
                  rows={1}
                  className="flex-1 bg-transparent text-white placeholder-dark-400 resize-none focus:outline-none px-2 py-2 max-h-32"
                  style={{
                    minHeight: '44px',
                    height: 'auto',
                  }}
                  onInput={(e) => {
                    const target = e.target as HTMLTextAreaElement;
                    target.style.height = 'auto';
                    target.style.height = `${Math.min(target.scrollHeight, 128)}px`;
                  }}
                />
                <motion.button
                  type="submit"
                  disabled={!input.trim() || isLoading}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="flex-shrink-0 p-3 bg-gradient-to-r from-primary-500 to-accent-purple text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                  <Send className="w-4 h-4" />
                </motion.button>
              </div>
              <p className="mt-2 text-xs text-dark-500 text-center">
                Press Enter to send • Shift+Enter for new line
              </p>
            </form>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
