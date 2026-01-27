import { ChatResponse } from '@/types';

// API base URL - update this to your deployed backend URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://your-backend-url.fly.dev';

export async function sendChatMessage(prompt: string): Promise<ChatResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('API Error:', error);
    return {
      response: '',
      error: error instanceof Error ? error.message : 'Failed to get response',
    };
  }
}

// Helper to parse markdown links from response
export function parseToolLinks(text: string): { name: string; url: string }[] {
  const linkRegex = /\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g;
  const links: { name: string; url: string }[] = [];
  let match;

  while ((match = linkRegex.exec(text)) !== null) {
    links.push({
      name: match[1],
      url: match[2],
    });
  }

  return links;
}

// Format response text with clickable links
export function formatResponseWithLinks(text: string): string {
  // Convert markdown links to HTML
  return text.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-primary-400 hover:text-primary-300 underline underline-offset-2 transition-colors">$1</a>'
  );
}
