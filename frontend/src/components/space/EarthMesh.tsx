import { useRef } from 'react';
import { useFrame, useLoader } from '@react-three/fiber';
import * as THREE from 'three';
import { AtmosphereGlow } from './AtmosphereGlow';

const EARTH_RADIUS = 2;

interface EarthMeshProps {
  onClick?: () => void;
}

export function EarthMesh({ onClick }: EarthMeshProps) {
  const earthRef = useRef<THREE.Mesh>(null);
  const cloudsRef = useRef<THREE.Mesh>(null);

  const [colorMap, normalMap, specularMap, cloudsMap] = useLoader(THREE.TextureLoader, [
    '/textures/earth_daymap.jpg',
    '/textures/earth_normal.jpg',
    '/textures/earth_specular.jpg',
    '/textures/earth_clouds.jpg',
  ]);

  useFrame((_, delta) => {
    if (earthRef.current) earthRef.current.rotation.y += delta * 0.02;
    if (cloudsRef.current) cloudsRef.current.rotation.y += delta * 0.027;
  });

  return (
    <group onClick={onClick}>
      <mesh ref={earthRef}>
        <sphereGeometry args={[EARTH_RADIUS, 128, 128]} />
        <meshPhongMaterial
          map={colorMap}
          normalMap={normalMap}
          specularMap={specularMap}
          specular={new THREE.Color('#333333')}
          shininess={10}
        />
      </mesh>

      <mesh ref={cloudsRef} scale={1.008}>
        <sphereGeometry args={[EARTH_RADIUS, 128, 128]} />
        <meshStandardMaterial map={cloudsMap} transparent opacity={0.45} depthWrite={false} />
      </mesh>

      <AtmosphereGlow radius={EARTH_RADIUS} color="#4db8ff" />
    </group>
  );
}
