/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        space: {
          950: '#010307',
          900: '#030712',
          850: '#070f1e',
          800: '#0b162a',
          700: '#112240',
        },
        cyan: {
          400: '#38bdf8',
          500: '#0ea5e9',
        },
        teal: {
          400: '#2dd4bf',
          500: '#14b8a6',
        },
      },
      fontFamily: {
        serif: ['Cinzel', 'Georgia', 'serif'],
        sans: ['Outfit', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        'glow-teal': '0 0 20px rgba(45, 212, 191, 0.35)',
        'glow-cyan': '0 0 24px rgba(56, 189, 248, 0.4)',
        'glow-white': '0 0 25px rgba(255, 255, 255, 0.35)',
      }
    },
  },
  plugins: [],
}
