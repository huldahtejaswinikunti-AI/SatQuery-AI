import React, { useEffect, useState } from 'react';
import type { ObservationDomain } from '../../types';
import { checkHealth } from '../../api/client';

interface NavProps {
  activeDomain: ObservationDomain;
  onSelectDomain: (domain: ObservationDomain) => void;
}

export const Nav: React.FC<NavProps> = ({ activeDomain, onSelectDomain }) => {
  const [calibration, setCalibration] = useState<'calibrated' | 'untrained_fallback' | 'unknown' | 'loading'>('loading');

  useEffect(() => {
    checkHealth()
      .then((data: any) => {
        if (data.classifier_calibration === 'untrained_fallback' || data.status === 'UNCALIBRATED') {
          setCalibration('untrained_fallback');
        } else if (data.classifier_calibration === 'calibrated' || data.status === 'READY') {
          setCalibration('calibrated');
        } else {
          setCalibration('unknown');
        }
      })
      .catch(() => setCalibration('unknown'));
  }, []);

  const isUncalibrated = calibration === 'untrained_fallback';
  const isUnknown = calibration === 'unknown' || calibration === 'loading';

  return (
    <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 backdrop-blur-md bg-[rgba(5,7,13,0.7)] border-b border-[var(--border-hairline)] transition-all">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-full border border-[var(--accent-verified)] flex items-center justify-center bg-[rgba(77,184,255,0.08)]">
          <span className="font-serif text-sm font-bold text-[var(--accent-verified)]">SQ</span>
        </div>
        <div className="flex flex-col">
          <span className="font-serif text-sm tracking-wider font-bold text-[var(--text-primary)]">
            SatQuery AI
          </span>
          <span className="font-mono text-[9px] tracking-widest text-[var(--text-secondary)] uppercase">
            MISSION CONTROL · PS 26167
          </span>
        </div>
      </div>

      {/* Center Domain Switcher */}
      <nav className="flex items-center gap-1 p-1 rounded-full bg-[rgba(18,26,46,0.8)] border border-[var(--border-hairline)]">
        <button
          onClick={() => onSelectDomain('earth')}
          className={`px-4 py-1.5 rounded-full text-xs font-medium tracking-wide transition-all focus:outline-none focus:ring-1 focus:ring-[var(--accent-verified)] ${
            activeDomain === 'earth'
              ? 'bg-[var(--accent-verified)] text-[var(--bg-void)] font-semibold shadow-sm'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
          }`}
        >
          🌍 Earth
        </button>
        <button
          onClick={() => onSelectDomain('lunar')}
          className={`px-4 py-1.5 rounded-full text-xs font-medium tracking-wide transition-all focus:outline-none focus:ring-1 focus:ring-[var(--accent-verified)] ${
            activeDomain === 'lunar'
              ? 'bg-[var(--accent-verified)] text-[var(--bg-void)] font-semibold shadow-sm'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
          }`}
        >
          🌕 Moon
        </button>
      </nav>

      {/* Right Telemetry & ISRO Indicator */}
      <div className="flex items-center gap-4">
        <a
          href="#observation-workspace"
          className="hidden sm:inline-block text-xs font-sans text-[var(--text-secondary)] hover:text-[var(--text-primary)] tracking-wide transition-colors"
        >
          Observation Deck ↓
        </a>
        <div
          className={`flex items-center gap-2 px-3 py-1 rounded-full border transition-colors ${
            isUncalibrated
              ? 'border-amber-500/50 bg-[rgba(245,158,11,0.12)] text-amber-300'
              : isUnknown
              ? 'border-zinc-700/50 bg-[rgba(24,24,27,0.6)] text-zinc-400'
              : 'border-[var(--border-hairline)] bg-[rgba(11,15,26,0.6)] text-[var(--text-primary)]'
          }`}
          title={
            isUncalibrated
              ? 'Warning: Land-cover classifier checkpoint missing or uncalibrated'
              : isUnknown
              ? 'Status: Connecting to telemetry server'
              : 'All systems calibrated'
          }
        >
          <span
            className={`w-2 h-2 rounded-full ${
              isUncalibrated
                ? 'bg-amber-400 animate-ping'
                : isUnknown
                ? 'bg-zinc-500 animate-pulse'
                : 'bg-emerald-400 animate-pulse'
            }`}
          />
          <span className="font-mono text-[10px] tracking-wider">
            {isUncalibrated
              ? 'SYSTEM READY · CLASSIFIER UNCALIBRATED'
              : isUnknown
              ? (calibration === 'loading' ? 'TELEMETRY INITIALIZING...' : 'SYSTEM STATUS · UNKNOWN')
              : 'SYSTEM READY · ISRO / SAC'}
          </span>
        </div>
      </div>
    </header>
  );
};
