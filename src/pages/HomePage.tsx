import React, { useEffect, useState } from 'react';
import { LOCKED_EVALUATION, LOCKED_HEADLINE } from '../state/lockedBenchmark';
import { BenchmarkSnapshot } from '../components/Benchmark/BenchmarkSnapshot';

const ENTRIES = [
  { href: '#/final', group: 'FINAL MODEL', title: 'STARE-Cognitive', desc: 'Verify the frozen STARE encoder and cognitive adaptive scheduler on a real HDF5 capture.' },
  { href: '#/lab', group: 'SCENARIO LAB', title: 'Scenario Lab', desc: 'Build an RF environment and observe scan decisions.' },
  { href: '#/arena', group: 'POLICY ARENA', title: 'Policy Arena', desc: 'Compare supported policies on the same scenario.' },
  { href: '#/benchmark', group: 'BENCHMARK', title: 'Benchmark', desc: 'Explore the documented benchmark and research results.' },
];

export const HomePage: React.FC = () => {
  const [heroRun, setHeroRun] = useState(0);

  useEffect(() => {
    const replay = () => setHeroRun((v) => v + 1);
    window.addEventListener('home:replay', replay);
    return () => window.removeEventListener('home:replay', replay);
  }, []);

  return (
    <div className="page home-page">
      <section className="spectra-hero" aria-label="Spectra Sentinel introduction">
        <div key={heroRun} className="spectra-logo-reveal">
          <img
            src="/spectra-sentinel-full.png"
            alt="Spectra Sentinel — Cognitive EW Spectrum Scanning"
            className="spectra-full-logo"
          />
        </div>
        <div className="spectra-name">SPECTRA SENTINEL</div>
        <div className="spectra-acse">ACSE</div>
        <div className="spectra-acse-full">ADAPTIVE COGNITIVE SCAN ENGINE</div>
      </section>

      <section className="home-intro">
        <div className="hero-kicker">ADAPTIVE SENSING · SIH26055</div>
        <h1 className="hero-title">Adaptive Cognitive Scan Scheduling Engine</h1>
        <div className="hero-sub">
          An adaptive receiver that decides where and how long to scan under limited sensing bandwidth.
        </div>
      </section>

      <section style={{ padding: '20px 0 18px', borderBottom: '1px solid var(--border)' }}>
        <div className="panel-title" style={{ textAlign: 'center', marginBottom: 12 }}>Explore the system</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10, maxWidth: 860, margin: '0 auto' }}>
          {ENTRIES.map((e) => (
            <a key={e.href} className="entry" href={e.href}>
              <div className="eg">{e.group}</div>
              <div className="et">{e.title}</div>
              <div className="ed">{e.desc}</div>
            </a>
          ))}
        </div>
      </section>

      <BenchmarkSnapshot showCta />

      <section style={{ paddingTop: 16 }}>
        <p className="section-note" style={{ maxWidth: 760, margin: '0 auto', textAlign: 'center' }}>
          This site is the demonstration layer for the SIH26055 research prototype. The Final Model page is the canonical backend verification path. The Scenario Lab is the continuous
          browser receiver demonstrator, while the Policy Arena and Benchmark expose research evidence.
          Quantitative claims above come from the project's locked research evaluation
          ({LOCKED_EVALUATION.scale.seeds} seeds · {LOCKED_EVALUATION.scale.episodes} episodes ·{' '}
          {LOCKED_EVALUATION.scale.scenarioInstances} scenario instances · {LOCKED_EVALUATION.scale.policies} policies).
          The adaptive policy's measured Pd was {LOCKED_HEADLINE.pdRatio}× the sequential baseline's under
          that exact locked configuration.
        </p>
      </section>

      <section className="panel panel-pad" style={{ marginTop: 18, borderColor: 'var(--accent)' }}>
        <div className="spread" style={{ alignItems: 'flex-start', gap: 18 }}>
          <div>
            <div className="hero-kicker">RECOMMENDED START</div>
            <div className="panel-title" style={{ color: 'var(--text)', fontSize: 14 }}>Final Model · STARE-Cognitive</div>
            <p className="small" style={{ maxWidth: 760, marginBottom: 0 }}>
              Open this first to verify the frozen STARE encoder and cognitive adaptive scheduler on a real HDF5 receiver capture.
            </p>
          </div>
          <a className="btn btn-primary" href="#/arena">Open Final Model in Policy Comparison →</a>
        </div>
      </section>
    </div>
  );
};
