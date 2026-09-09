import { useRef } from 'react';
import { useFrame, useLoader } from '@react-three/fiber';
import * as THREE from 'three';

interface MoonMeshProps {
  radius?: number;
  position?: [number, number, number];
  onClick?: () => void;
  orbit?: boolean;
}

export function MoonMesh({ radius = 0.5, position = [2.8, 0.6, -0.4], onClick, orbit = false }: MoonMeshProps) {
  const moonGroupRef = useRef<THREE.Group>(null);
  const moonRef = useRef<THREE.Mesh>(null);
  const [colorMap, bumpMap] = useLoader(THREE.TextureLoader, [
    '/textures/moon.jpg',
    '/textures/moon_bump.jpg',
  ]);

  const orbitAngle = useRef(0.6);

  useFrame((_, delta) => {
    if (moonRef.current) {
      moonRef.current.rotation.y += delta * 0.01;
    }
    if (orbit && moonGroupRef.current) {
      orbitAngle.current += delta * 0.08;
      const x = Math.cos(orbitAngle.current) * 3.4;
      const z = Math.sin(orbitAngle.current) * 1.8 - 0.5;
      const y = Math.sin(orbitAngle.current * 0.5) * 0.4 + 0.3;
      moonGroupRef.current.position.set(x, y, z);
    }
  });

  return (
    <group ref={moonGroupRef} position={position} onClick={onClick}>
      <mesh ref={moonRef}>
        <sphereGeometry args={[radius, 64, 64]} />
        <meshStandardMaterial
          map={colorMap}
          bumpMap={bumpMap}
          bumpScale={0.03}
          roughness={0.95}
          metalness={0.05}
        />
      </mesh>
    </group>
  );
}
