/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#0a0a0a',
          surface: '#141414',
          border: '#262626',
          text: {
            primary: '#e5e5e5',
            secondary: '#a3a3a3',
            muted: '#737373',
          }
        }
      }
    },
  },
  plugins: [],
}
