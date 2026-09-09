import React from 'react';
import type { ObservationDomain } from '../../types';

interface ConfidenceBadgeProps {
  domain: ObservationDomain;
  confidenceScore?: number;
  isVerified?: boolean;
  confidenceTag?: string;
  calibrationStatus?: 'calibrated' | 'untrained_fallback' | string;
}

export const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({
  domain,
  confidenceScore,
  isVerified = true,
  confidenceTag,
  calibrationStatus,
}) => {
  const isLunar = domain === 'lunar';
  const isUncalibrated = calibrationStatus === 'untrained_fallback';
  const tag = (confidenceTag || '').toLowerCase();
  const isDisagreement = tag.includes('disagreement') || (tag === 'lower_confidence' && !isVerified);

  if (isUncalibrated || isLunar) {
    const label = isUncalibrated
      ? 'CLASSIFIER UNCALIBRATED'
      : 'EXPERIMENTAL / UNVERIFIED';

    return (
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-[rgba(245,166,35,0.3)] bg-[rgba(245,166,35,0.08)]">
        <span className="w-2 h-2 rounded-full bg-[var(--accent-experimental)] animate-pulse" />
        <span className="font-mono text-[10px] tracking-wider font-semibold text-[var(--accent-experimental)] uppercase">
          {label}
        </span>
        {confidenceScore !== undefined && (
          <span className="font-mono text-[10px] text-[var(--text-secondary)] border-l border-[rgba(245,166,35,0.3)] pl-2">
            {(confidenceScore * 100).toFixed(0)}%{isUncalibrated ? ' (UNCALIBRATED)' : ''}
          </span>
        )}
      </div>
    );
  }

  if (isDisagreement) {
    return (
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-rose-500/40 bg-rose-950/20 shadow-sm">
        <span className="w-2 h-2 rounded-full bg-rose-400 animate-pulse" />
        <span className="font-mono text-[10px] tracking-wider font-semibold text-rose-300 uppercase">
          CONFLICT DETECTED / DISAGREEMENT
        </span>
        {confidenceScore !== undefined && (
          <span className="font-mono text-[10px] text-rose-300/80 border-l border-rose-500/30 pl-2">
            {(confidenceScore * 100).toFixed(0)}% CONFIDENCE
          </span>
        )}
      </div>
    );
  }

  if (!isVerified) {
    return (
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-amber-500/30 bg-amber-500/10">
        <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
        <span className="font-mono text-[10px] tracking-wider font-semibold text-amber-300 uppercase">
          SINGLE SIGNAL — NOT CROSS-VERIFIED
        </span>
        {confidenceScore !== undefined && (
          <span className="font-mono text-[10px] text-[var(--text-secondary)] border-l border-amber-500/30 pl-2">
            {(confidenceScore * 100).toFixed(0)}%
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-[rgba(77,184,255,0.3)] bg-[rgba(77,184,255,0.08)]">
      <span className="w-2 h-2 rounded-full bg-[var(--accent-verified)] animate-pulse" />
      <span className="font-mono text-[10px] tracking-wider font-semibold text-[var(--accent-verified)] uppercase">
        VERIFIED OBSERVATION
      </span>
      {confidenceScore !== undefined && (
        <span className="font-mono text-[10px] text-[var(--text-secondary)] border-l border-[rgba(77,184,255,0.3)] pl-2">
          {(confidenceScore * 100).toFixed(0)}% CONFIDENCE
        </span>
      )}
    </div>
  );
};
