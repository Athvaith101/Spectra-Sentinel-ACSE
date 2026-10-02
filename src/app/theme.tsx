/**
 * Theme context — light/dark preference for the demo.
 * Preference persists in localStorage ('sih26055-theme') and is applied via
 * `data-theme` on <html> so every CSS variable resolves consistently.
 */
import React, { createContext, useContext, useEffect, useState } from 'react';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'sih26055-theme';

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    /* storage unavailable (private mode) — fall through to default */
  }
  return 'light';
}

const ThemeCtx = createContext<{ theme: Theme; setTheme: (t: Theme) => void }>({
  theme: 'light',
  setTheme: () => undefined,
});

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setThemeState] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const setTheme = (t: Theme) => {
    setThemeState(t);
    try {
      localStorage.setItem(STORAGE_KEY, t);
    } catch {
      /* ignore */
    }
  };

  return <ThemeCtx.Provider value={{ theme, setTheme }}>{children}</ThemeCtx.Provider>;
};

export function useTheme() {
  return useContext(ThemeCtx);
}
