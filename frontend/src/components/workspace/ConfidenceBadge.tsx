import React from 'react';
import type { ObservationDomain } from '../../types';

interface ConfidenceBadgeProps {
  domain: ObservationDomain;
  confidenceScore?: number;
  isVerified?: boolean;
}

export const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({
  domain,
  confidenceScore,
  isVerified = true,
}) => {
  const isLunar = domain === 'lunar';

  if (isLunar) {
    return (
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-[rgba(245,166,35,0.3)] bg-[rgba(245,166,35,0.08)]">
        <span className="w-2 h-2 rounded-full bg-[var(--accent-experimental)] animate-pulse" />
        <span className="font-mono text-[10px] tracking-wider font-semibold text-[var(--accent-experimental)] uppercase">
          EXPERIMENTAL / UNVERIFIED
        </span>
        {confidenceScore !== undefined && (
          <span className="font-mono text-[10px] text-[var(--text-secondary)] border-l border-[rgba(245,166,35,0.3)] pl-2">
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
        {isVerified ? 'VERIFIED OBSERVATION' : 'CROSS-CHECK ACTIVE'}
      </span>
      {confidenceScore !== undefined && (
        <span className="font-mono text-[10px] text-[var(--text-secondary)] border-l border-[rgba(77,184,255,0.3)] pl-2">
          {(confidenceScore * 100).toFixed(0)}% CONFIDENCE
        </span>
      )}
    </div>
  );
};
