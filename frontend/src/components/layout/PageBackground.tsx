import React from 'react';

export const PageBackground: React.FC<{ children?: React.ReactNode }> = ({ children }) => {
  return (
    <div className="min-h-screen w-full relative bg-[var(--bg-void)] text-[var(--text-primary)]">
      {/* Deep celestial radial gradient & subtle grid pattern */}
      <div 
        className="fixed inset-0 pointer-events-none z-0 opacity-40"
        style={{
          backgroundImage: `
            radial-gradient(circle at 50% 15%, rgba(77, 184, 255, 0.08) 0%, transparent 60%),
            radial-gradient(circle at 80% 70%, rgba(20, 35, 60, 0.15) 0%, transparent 50%)
          `
        }}
      />
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};
