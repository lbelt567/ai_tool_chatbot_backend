export interface Tool {
  name: string;
  slug: string;
  homepage: string;
  description: string;
  priceText?: string;
  pricing: 'Free' | 'Freemium' | 'Paid' | 'Free Trial' | 'Contact for Pricing';
  categories: string[];
  verified: boolean;
  favoriteCount: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  tools?: Tool[];
  timestamp: Date;
}

export interface ChatResponse {
  response: string;
  error?: string;
}

export type PricingFilter = 'all' | 'free' | 'freemium' | 'paid' | 'trial';

export interface Category {
  name: string;
  slug: string;
  count: number;
  icon?: string;
}
