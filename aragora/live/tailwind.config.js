/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'bg': '#0a0a0f',
        'surface': '#12121a',
        'border': '#2a2a3a',
        'text': '#e4e4eb',
        'text-muted': '#8888a0',
        'accent': '#6366f1',
        'accent-glow': 'rgba(99, 102, 241, 0.3)',
        'success': '#22c55e',
        'warning': '#f97316',
        'purple': '#a78bfa',
        'gold': '#fbbf24',
        'crimson': '#ef4444',  // Grok/xAI red
      },
    },
  },
  plugins: [],
}
