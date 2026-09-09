import { shaderMaterial } from '@react-three/drei';
import { extend } from '@react-three/fiber';
import * as THREE from 'three';

const AtmosphereMaterial = shaderMaterial(
  { glowColor: new THREE.Color('#4db8ff') },
  /* glsl */ `
    varying vec3 vNormal;
    void main() {
      vNormal = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  /* glsl */ `
    uniform vec3 glowColor;
    varying vec3 vNormal;
    void main() {
      // Fresnel rim glow: 0 at center, soft glow only at the planetary limb
      float dotProd = dot(vNormal, vec3(0.0, 0.0, 1.0));
      float intensity = pow(1.0 - max(dotProd, 0.0), 3.2);
      gl_FragColor = vec4(glowColor, clamp(intensity * 0.65, 0.0, 0.85));
    }
  `
);
extend({ AtmosphereMaterial });

declare global {
  namespace JSX {
    interface IntrinsicElements {
      atmosphereMaterial: React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
        glowColor?: THREE.Color;
        transparent?: boolean;
        side?: THREE.Side;
        blending?: THREE.Blending;
        depthWrite?: boolean;
      };
    }
  }
}

export function AtmosphereGlow({ radius, color }: { radius: number; color: string }) {
  return (
    <mesh scale={1.035}>
      <sphereGeometry args={[radius, 64, 64]} />
      {/* @ts-ignore */}
      <atmosphereMaterial
        glowColor={new THREE.Color(color)}
        transparent
        side={THREE.FrontSide}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </mesh>
  );
}
