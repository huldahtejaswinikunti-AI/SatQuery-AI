import { Stars as DreiStars } from '@react-three/drei';

export function Stars() {
  return (
    <DreiStars
      radius={300}
      depth={60}
      count={6000}
      factor={4}
      saturation={0}
      fade
      speed={0.4}
    />
  );
}
