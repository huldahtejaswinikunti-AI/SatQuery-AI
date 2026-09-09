import React from 'react';
import type { ObservationDomain } from '../../types';
import { SolarSystemCanvas } from './SolarSystemCanvas';

interface PlanetaryHeroProps {
  activeDomain: ObservationDomain;
  onSelectDomain: (domain: ObservationDomain) => void;
  systemStatus?: string;
}

export const PlanetaryHero: React.FC<PlanetaryHeroProps> = ({
  activeDomain,
  onSelectDomain,
}) => {
  const isLunar = activeDomain === 'lunar';

  const scrollToDeck = () => {
    const target = document.getElementById('observation-workspace');
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="relative w-full h-[620px] bg-[#05070d] overflow-hidden select-none border-b border-[var(--border-hairline)] pt-16">
      {/* 3D WebGL Multi-Map Canvas Backdrop */}
      <SolarSystemCanvas activeDomain={activeDomain} onSelectDomain={onSelectDomain} />

      {/* Center Cinematic Typography & Call to Action */}
      <div className="absolute top-28 left-1/2 -translate-x-1/2 z-10 text-center pointer-events-none max-w-2xl w-[92%]">
        <div className="font-mono text-xs tracking-[8px] text-[var(--text-secondary)] uppercase mb-2">
          PLANET
        </div>

        <h1 className="text-6xl md:text-7xl font-serif font-black tracking-wider text-[var(--text-primary)] uppercase drop-shadow-[0_4px_30px_rgba(0,0,0,0.9)]">
          {isLunar ? 'MOON' : 'EARTH'}
        </h1>

        {/* Accent Glow Line */}
        <div className="w-16 h-[2px] bg-[var(--accent-verified)] mx-auto mt-3 mb-4 rounded-full shadow-[0_0_12px_var(--accent-verified)]" />

        <p className="font-sans text-sm md:text-base text-[var(--text-secondary)] leading-relaxed max-w-xl mx-auto mb-7 drop-shadow-[0_2px_12px_rgba(0,0,0,0.9)]">
          {isLunar
            ? 'High-resolution Chandrayaan-2 TMC-2 / OHRC lunar surface intelligence, crater morphology, and shadowed cold-trap analysis with deterministic verification.'
            : 'Multimodal remote sensing intelligence with GeoChat vision-language reasoning, CLIPSeg spatial grounding, and Sentinel-1/2 optical-SAR consensus.'}
        </p>

        {/* Buttons */}
        <div className="pointer-events-auto flex items-center justify-center gap-3">
          <button
            onClick={scrollToDeck}
            className="bg-[var(--text-primary)] hover:bg-[var(--accent-verified)] text-[var(--bg-void)] font-sans font-bold text-xs tracking-wider uppercase px-7 py-3 rounded-full shadow-[0_0_25px_rgba(255,255,255,0.35)] hover:shadow-[0_0_25px_rgba(77,184,255,0.5)] transition-all duration-200 transform hover:-translate-y-0.5 cursor-pointer focus:outline-none focus:ring-2 focus:ring-[var(--accent-verified)]"
          >
            EXPLORE OBSERVATIONS &darr;
          </button>
          <button
            onClick={() => onSelectDomain(isLunar ? 'earth' : 'lunar')}
            className="bg-[rgba(11,15,26,0.8)] hover:bg-[rgba(18,26,46,0.9)] text-[var(--text-primary)] border border-[var(--border-hairline)] hover:border-[var(--accent-verified)] font-sans font-bold text-xs tracking-wider uppercase px-5 py-3 rounded-full backdrop-blur-md transition-all duration-200 cursor-pointer focus:outline-none focus:ring-2 focus:ring-[var(--accent-verified)]"
          >
            SWITCH TO {isLunar ? 'EARTH' : 'MOON'} &rarr;
          </button>
        </div>
      </div>

      {/* Flank Celestial Switchers */}
      <button
        onClick={() => onSelectDomain('earth')}
        className={`absolute top-1/2 -translate-y-1/2 left-6 z-20 flex items-center gap-3 pointer-events-auto px-4 py-2 rounded-full backdrop-blur-md border transition-all duration-200 group cursor-pointer focus:outline-none focus:ring-1 focus:ring-[var(--accent-verified)] ${
          !isLunar
            ? 'bg-[rgba(11,15,26,0.85)] border-[var(--accent-verified)] shadow-[0_0_16px_rgba(77,184,255,0.3)]'
            : 'bg-[rgba(11,15,26,0.5)] border-[var(--border-hairline)] hover:border-[rgba(77,184,255,0.4)] hover:scale-105'
        }`}
      >
        <div className="w-6 h-6 rounded-full bg-[radial-gradient(circle_at_35%_35%,#4db8ff_0%,#0369a1_70%,#05070d_100%)] shadow-[0_0_10px_rgba(77,184,255,0.4)]" />
        <span className="font-sans text-xs font-bold tracking-widest text-[var(--text-primary)] uppercase">
          EARTH
        </span>
      </button>

      <button
        onClick={() => onSelectDomain('lunar')}
        className={`absolute top-1/2 -translate-y-1/2 right-6 z-20 flex items-center gap-3 pointer-events-auto px-4 py-2 rounded-full backdrop-blur-md border transition-all duration-200 group cursor-pointer focus:outline-none focus:ring-1 focus:ring-[var(--accent-verified)] ${
          isLunar
            ? 'bg-[rgba(11,15,26,0.85)] border-[var(--accent-verified)] shadow-[0_0_16px_rgba(77,184,255,0.3)]'
            : 'bg-[rgba(11,15,26,0.5)] border-[var(--border-hairline)] hover:border-[rgba(77,184,255,0.4)] hover:scale-105'
        }`}
      >
        <span className="font-sans text-xs font-bold tracking-widest text-[var(--text-primary)] uppercase">
          MOON
        </span>
        <div className="w-6 h-6 rounded-full bg-[radial-gradient(circle_at_35%_35%,#e2e8f0_0%,#94a3b8_60%,#334155_100%)] shadow-[0_0_10px_rgba(226,232,240,0.3)]" />
      </button>

      {/* Circular Scroll Down Cue */}
      <button
        onClick={scrollToDeck}
        className="absolute bottom-5 left-1/2 -translate-x-1/2 z-20 w-9 h-9 rounded-full bg-[rgba(11,15,26,0.8)] hover:bg-[var(--accent-verified)] text-[var(--text-primary)] hover:text-[var(--bg-void)] border border-[var(--border-hairline)] hover:border-[var(--accent-verified)] flex items-center justify-center text-sm transition-all duration-200 hover:translate-y-0.5 shadow-lg pointer-events-auto cursor-pointer focus:outline-none focus:ring-1 focus:ring-[var(--accent-verified)]"
        title="Scroll to Observation Workspace"
      >
        &darr;
      </button>
    </section>
  );
};
