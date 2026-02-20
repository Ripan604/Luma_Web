/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}"
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['var(--font-display)', 'system-ui', 'sans-serif'],
        body: ['var(--font-body)', 'system-ui', 'sans-serif'],
      },
      colors: {
        surface: {
          950: '#0a0a0f',
          900: '#0f0f18',
          800: '#16161f',
          700: '#1e1e2a',
          600: '#2a2a3a',
        },
        accent: {
          cyan: '#22d3ee',
          violet: '#a78bfa',
          fuchsia: '#e879f9',
          emerald: '#34d399',
        },
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2.5s ease-in-out infinite alternate',
        'gradient-x': 'gradient-x 8s ease infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'scale-in': 'scale-in 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-up': 'slide-up 0.5s cubic-bezier(0.16, 1, 0.3, 1)',
        'progress': 'progress 0.4s ease-out forwards',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        glow: {
          '0%': { opacity: '0.6', filter: 'blur(20px)' },
          '100%': { opacity: '1', filter: 'blur(24px)' },
        },
        'gradient-x': {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'scale-in': {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        'slide-up': {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        progress: {
          '0%': { width: '0%' },
          '100%': { width: 'var(--progress-width, 100%)' },
        },
      },
      backgroundSize: {
        '300': '300% 300%',
      },
      boxShadow: {
        'glow-cyan': '0 0 40px -10px rgba(34, 211, 238, 0.5)',
        'glow-violet': '0 0 40px -10px rgba(167, 139, 250, 0.5)',
        'glow-emerald': '0 0 40px -10px rgba(52, 211, 153, 0.4)',
        'inner-glow': 'inset 0 0 60px -20px rgba(34, 211, 238, 0.15)',
      },
    },
  },
  plugins: [],
}
