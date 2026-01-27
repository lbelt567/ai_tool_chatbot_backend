'use client';

import { useState, useCallback } from 'react';
import Navigation from '@/components/Navigation';
import Hero from '@/components/Hero';
import ChatInterface from '@/components/ChatInterface';
import Features from '@/components/Features';
import CategorySection from '@/components/CategorySection';
import PopularTools from '@/components/PopularTools';
import Footer from '@/components/Footer';
import FloatingChatButton from '@/components/FloatingChatButton';

export default function Home() {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [initialQuery, setInitialQuery] = useState<string | null>(null);

  const openChatWithQuery = useCallback((query: string) => {
    setInitialQuery(query);
    setIsChatOpen(true);
  }, []);

  const handleOpenChat = useCallback(() => {
    setIsChatOpen(true);
  }, []);

  const handleCloseChat = useCallback(() => {
    setIsChatOpen(false);
    setInitialQuery(null);
  }, []);

  return (
    <main className="relative">
      {/* Navigation */}
      <Navigation />

      {/* Hero Section */}
      <Hero onStartChat={handleOpenChat} />

      {/* Features Section */}
      <Features />

      {/* Category Section */}
      <CategorySection onCategoryClick={openChatWithQuery} />

      {/* Popular Tools Section */}
      <PopularTools onAskAbout={openChatWithQuery} />

      {/* Footer */}
      <Footer />

      {/* Chat Interface Modal */}
      <ChatInterface isOpen={isChatOpen} onClose={handleCloseChat} />

      {/* Floating Chat Button */}
      <FloatingChatButton isOpen={isChatOpen} onClick={() => setIsChatOpen(!isChatOpen)} />
    </main>
  );
}
