/**
 * App shell — top navigation + client-side routing (hash-based, works on any static
 * host without server rewrites; spec §14).
 *
 * Pages: Home · Scenario Lab · Policy Arena · Dataset · Methodology.
 */
import React, { useEffect, useState } from 'react';
import { ThemeProvider } from './theme';
import { HomePage } from '../pages/HomePage';
import { ScenarioLabPage } from '../pages/ScenarioLabPage';
import { BenchmarkPage } from '../pages/BenchmarkPage';
import { MethodologyPage } from '../pages/MethodologyPage';
import { FinalModelPage } from '../pages/FinalModelPage';
import { BackendDatasetPage } from '../pages/BackendDatasetPage';
import { BackendPolicyArenaPage } from '../pages/BackendPolicyArenaPage';

export type RouteId = 'home' | 'final' | 'lab' | 'arena' | 'benchmark' | 'dataset' | 'method';

const ROUTES: { id: RouteId; path: string; label: string }[] = [
  { id: 'home', path: '#/', label: 'Home' },
  { id: 'final', path: '#/final', label: 'Final Model' },
  { id: 'lab', path: '#/lab', label: 'Scenario Lab' },
  { id: 'arena', path: '#/arena', label: 'Policy Arena' },
  { id: 'benchmark', path: '#/benchmark', label: 'Benchmark' },
  { id: 'dataset', path: '#/dataset', label: 'Dataset' },
  { id: 'method', path: '#/method', label: 'Methodology' },
];

function routeFromHash(): RouteId {
  const h = window.location.hash;
  if (h.startsWith('#/final')) return 'final';
  if (h.startsWith('#/lab')) return 'lab';
  if (h.startsWith('#/arena')) return 'arena';
  if (h.startsWith('#/benchmark')) return 'benchmark';
  if (h.startsWith('#/dataset')) return 'dataset';
  if (h.startsWith('#/method')) return 'method';
  return 'home';
}

export const App: React.FC = () => {
  const [route, setRoute] = useState<RouteId>(routeFromHash);

  useEffect(() => {
    const onHash = () => {
      setRoute(routeFromHash());
      window.scrollTo(0, 0);
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  return (
    <ThemeProvider>
      <div>
        <header className="topnav">
          <div className="topnav-inner">
            <a className="corner-mark" href="#/" aria-label="Spectra Sentinel home">
              <img src="/spectra-sentinel-mark.png" alt="Spectra Sentinel" />
            </a>
            <nav className="topnav-links" aria-label="Primary navigation">
              {ROUTES.map((r) => (
                <a key={r.id} className={`navlink${route === r.id ? ' active' : ''}`} href={r.path} onClick={() => { if (r.id === 'home') window.dispatchEvent(new Event('home:replay')); }}>
                  {r.label}
                </a>
              ))}
            </nav>
            <a className="sih-corner" href="#/" aria-label="SIH 2026 home">SIH 2026</a>
          </div>
        </header>

        {route === 'home' && <HomePage />}
        {route === 'final' && <FinalModelPage />}
        {route === 'lab' && <ScenarioLabPage />}
        {route === 'arena' && <BackendPolicyArenaPage />}
        {route === 'benchmark' && <BenchmarkPage />}
        {route === 'dataset' && <BackendDatasetPage />}
        {route === 'method' && <MethodologyPage />}

        <div className="page">
          <footer className="foot">
            SIH26055 · Smart Scan Strategy for Electronic Warfare — demonstration and verification layer for the SIH26055 v6.2 research engine. STARE-Cognitive is the canonical
            final-model path surfaced directly inside Policy Comparison and Dataset Evaluation; Scenario Lab is the continuous visual receiver demonstrator.
            Uploaded-file metrics are diagnostic replay measurements and are never presented as the locked benchmark.
          </footer>
        </div>
      </div>
    </ThemeProvider>
  );
};
