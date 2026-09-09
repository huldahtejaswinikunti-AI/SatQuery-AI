import { describe, it, expect } from 'vitest';
import { renderToString } from 'react-dom/server';
import { ConfidenceBadge } from './ConfidenceBadge';

describe('ConfidenceBadge component', () => {
  it('renders uncalibrated state when calibrationStatus is untrained_fallback and does not render VERIFIED OBSERVATION', () => {
    const html = renderToString(
      <ConfidenceBadge
        domain="earth"
        confidenceScore={0.85}
        isVerified={true}
        calibrationStatus="untrained_fallback"
      />
    );

    // Must visibly display uncalibrated state
    expect(html).toContain('CLASSIFIER UNCALIBRATED');

    // Must NOT display verified observation
    expect(html).not.toContain('VERIFIED OBSERVATION');

    // Must include uncalibrated tag
    expect(html).toContain('(UNCALIBRATED)');
  });

  it('renders verified state when calibrationStatus is calibrated', () => {
    const html = renderToString(
      <ConfidenceBadge
        domain="earth"
        confidenceScore={0.92}
        isVerified={true}
        calibrationStatus="calibrated"
      />
    );

    expect(html).toContain('VERIFIED OBSERVATION');
    expect(html).not.toContain('CLASSIFIER UNCALIBRATED');
  });

  it('renders lunar experimental state when domain is lunar', () => {
    const html = renderToString(
      <ConfidenceBadge
        domain="lunar"
        confidenceScore={0.88}
        isVerified={false}
      />
    );

    expect(html).toContain('EXPERIMENTAL / UNVERIFIED');
    expect(html).not.toContain('VERIFIED OBSERVATION');
  });
});
