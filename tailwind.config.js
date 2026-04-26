/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        base: '#000000',
        panel: '#07111f',
        line: '#16324d',
        neon: '#52f7ff',
        sky: '#4d8dff',
        mint: '#5cffc8',
        danger: '#ff5d7d',
        text: '#f4fbff',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      backgroundImage: {
        'mesh-gradient':
          'radial-gradient(circle at top, rgba(82,247,255,0.18), transparent 30%), radial-gradient(circle at 70% 20%, rgba(77,141,255,0.16), transparent 24%), radial-gradient(circle at bottom, rgba(12,29,46,0.9), transparent 40%)',
        'panel-gradient':
          'linear-gradient(180deg, rgba(10,21,36,0.86), rgba(2,8,16,0.82))',
      },
      boxShadow: {
        panel: '0 24px 80px rgba(0, 0, 0, 0.45)',
        neon: '0 0 32px rgba(82, 247, 255, 0.22)',
        blue: '0 0 32px rgba(77, 141, 255, 0.18)',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translate3d(0, 0, 0)' },
          '50%': { transform: 'translate3d(0, -12px, 0)' },
        },
        pulseGrid: {
          '0%, 100%': { opacity: '0.5' },
          '50%': { opacity: '0.9' },
        },
      },
      animation: {
        float: 'float 8s ease-in-out infinite',
        'pulse-grid': 'pulseGrid 4s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
