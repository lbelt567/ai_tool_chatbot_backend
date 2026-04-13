import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AI Tool Recommender | Find the Perfect AI Tools for Your Needs',
  description: 'Discover the best AI tools from 3,800+ curated options. Get personalized recommendations powered by AI to find free and paid tools for any task.',
  keywords: 'AI tools, AI recommendations, ChatGPT alternatives, AI directory, free AI tools, AI software',
  authors: [{ name: 'AI Tool Recommender' }],
  icons: {
    icon: [
      { url: '/innovative-mind-toolbox-logo.png', type: 'image/png' },
    ],
    apple: [
      { url: '/innovative-mind-toolbox-logo.png', type: 'image/png' },
    ],
    shortcut: ['/innovative-mind-toolbox-logo.png'],
  },
  openGraph: {
    title: 'AI Tool Recommender | Find the Perfect AI Tools',
    description: 'Discover 3,800+ AI tools with personalized recommendations. Find free alternatives and the best tools for your specific needs.',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'AI Tool Recommender',
    description: 'Discover 3,800+ AI tools with personalized recommendations.',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-dark-950 text-dark-50 overflow-x-hidden">
        {children}
      </body>
    </html>
  );
}
