import React from 'react';

interface SectionDividerProps {
  className?: string;
  label?: string;
}

export const SectionDivider: React.FC<SectionDividerProps> = ({ className = '', label }) => {
  if (label) {
    return (
      <div className={`flex items-center my-6 gap-4 ${className}`}>
        <div className="h-px flex-1 bg-[var(--border-hairline)]" />
        <span className="font-mono text-[10px] tracking-widest text-[var(--text-secondary)] uppercase">
          {label}
        </span>
        <div className="h-px flex-1 bg-[var(--border-hairline)]" />
      </div>
    );
  }

  return <div className={`h-px w-full bg-[var(--border-hairline)] ${className}`} />;
};
