import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import type { ObservationDomain } from '../../types';

interface SolarSystemCanvasProps {
  activeDomain: ObservationDomain;
  onSelectDomain: (domain: ObservationDomain) => void;
}

export const SolarSystemCanvas: React.FC<SolarSystemCanvasProps> = ({
  activeDomain,
  onSelectDomain,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const activeDomainRef = useRef<ObservationDomain>(activeDomain);
  activeDomainRef.current = activeDomain;

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight || 620;

    // 1. Scene & Renderer (Strict black clear color — NEVER white)
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070d);

    const camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x05070d, 1);
    container.appendChild(renderer.domElement);

    // 2. Starfield with 3,000 multi-magnitude celestial stars
    const starCount = 3000;
    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(starCount * 3);
    const starColors = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount * 3; i += 3) {
      const radius = 70 + Math.random() * 150;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      starPos[i] = radius * Math.sin(phi) * Math.cos(theta);
      starPos[i + 1] = radius * Math.sin(phi) * Math.sin(theta);
      starPos[i + 2] = radius * Math.cos(phi);

      const tint = Math.random();
      if (tint > 0.85) {
        starColors[i] = 0.3; starColors[i + 1] = 0.72; starColors[i + 2] = 1.0; // #4db8ff cyan
      } else if (tint < 0.1) {
        starColors[i] = 1.0; starColors[i + 1] = 0.88; starColors[i + 2] = 0.7; // warm sun star
      } else {
        starColors[i] = 0.95; starColors[i + 1] = 0.95; starColors[i + 2] = 1.0; // crisp white
      }
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    starGeo.setAttribute('color', new THREE.BufferAttribute(starColors, 3));
    const starMat = new THREE.PointsMaterial({
      size: 1.1,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
    });
    const starfield = new THREE.Points(starGeo, starMat);
    scene.add(starfield);

    // 3. Multi-Map Texture Loader
    const textureLoader = new THREE.TextureLoader();

    const earthDay = textureLoader.load('/textures/earth_daymap.jpg');
    const earthNormal = textureLoader.load('/textures/earth_normal.jpg');
    const earthSpec = textureLoader.load('/textures/earth_specular.jpg');
    const earthClouds = textureLoader.load('/textures/earth_clouds.jpg');
    const moonMap = textureLoader.load('/textures/moon.jpg');
    const moonBump = textureLoader.load('/textures/moon_bump.jpg');

    // 4. Planetary Setup: Earth Hero Geometry
    const planetRadius = 4.0;
    const planetY = -3.2;

    // Earth Sphere with Day/Night Phong Shading
    const earthGeo = new THREE.SphereGeometry(planetRadius, 64, 64);
    const earthMat = new THREE.MeshPhongMaterial({
      map: earthDay,
      normalMap: earthNormal,
      normalScale: new THREE.Vector2(0.85, 0.85),
      specularMap: earthSpec,
      specular: new THREE.Color(0x334455),
      shininess: 18,
    });
    const earth = new THREE.Mesh(earthGeo, earthMat);
    earth.position.set(0, planetY, 0);
    earth.rotation.z = (23.4 * Math.PI) / 180;
    scene.add(earth);

    // Separate Drifting Cloud Mesh
    const cloudGeo = new THREE.SphereGeometry(planetRadius + 0.035, 64, 64);
    const cloudMat = new THREE.MeshStandardMaterial({
      map: earthClouds,
      transparent: true,
      opacity: 0.38,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const clouds = new THREE.Mesh(cloudGeo, cloudMat);
    clouds.position.set(0, planetY, 0);
    scene.add(clouds);

    // Soft Rayleigh Atmosphere Limb Glow (Fresnel-like soft rim, no solid blowout)
    const atmoGeo = new THREE.SphereGeometry(planetRadius + 0.14, 64, 64);
    const atmoMat = new THREE.MeshBasicMaterial({
      color: 0x4db8ff,
      transparent: true,
      opacity: 0.22,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const atmosphere = new THREE.Mesh(atmoGeo, atmoMat);
    atmosphere.position.set(0, planetY, 0);
    scene.add(atmosphere);

    // 5. Orbiting Moon with Bump-Mapped Crater Relief
    const moonGeo = new THREE.SphereGeometry(0.82, 48, 48);
    const moonMat = new THREE.MeshStandardMaterial({
      map: moonMap,
      bumpMap: moonBump,
      bumpScale: 0.045,
      roughness: 0.95,
      metalness: 0.05,
    });
    const moon = new THREE.Mesh(moonGeo, moonMat);
    scene.add(moon);

    // Subtle Elliptical Orbit Line
    const orbitCurve = new THREE.EllipseCurve(0, planetY + 0.3, 7.8, 4.0, 0, 2 * Math.PI, false, 0);
    const orbitPoints = orbitCurve.getPoints(120);
    const orbitGeo = new THREE.BufferGeometry().setFromPoints(
      orbitPoints.map((p) => new THREE.Vector3(p.x, p.y + 0.6, p.y * 0.4))
    );
    const orbitMat = new THREE.LineBasicMaterial({
      color: 0x4db8ff,
      transparent: true,
      opacity: 0.12,
    });
    const orbitLine = new THREE.Line(orbitGeo, orbitMat);
    scene.add(orbitLine);

    // 6. Master Lighting Rig: Sharp Sunlight with Day/Night Terminator
    const sunLight = new THREE.DirectionalLight(0xfff6e5, 2.6);
    sunLight.position.set(15, 8, 10);
    scene.add(sunLight);

    const ambientLight = new THREE.AmbientLight(0x05070d, 0.35);
    scene.add(ambientLight);

    camera.position.set(0, 1.5, 5.2);
    camera.lookAt(0, -0.4, 0);

    // 7. Interactive Parallax & Raycasting
    let isDragging = false;
    let prevX = 0;
    let camTargetX = 0;
    let camTargetY = 1.5;

    const canvasEl = renderer.domElement;
    canvasEl.style.cursor = 'grab';

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      prevX = e.clientX;
      canvasEl.style.cursor = 'grabbing';
    };
    const onMouseUp = () => {
      isDragging = false;
      canvasEl.style.cursor = 'grab';
    };
    const onMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const dx = e.clientX - prevX;
        prevX = e.clientX;
        earth.rotation.y += dx * 0.005;
        clouds.rotation.y += dx * 0.005;
        moon.rotation.y += dx * 0.005;
      } else {
        const nx = e.clientX / width - 0.5;
        const ny = e.clientY / height - 0.5;
        camTargetX = nx * 0.35;
        camTargetY = 1.5 - ny * 0.25;
      }
    };

    canvasEl.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('mousemove', onMouseMove);

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onClick = (e: MouseEvent) => {
      const rect = canvasEl.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);

      const intersects = raycaster.intersectObjects([moon, earth]);
      if (intersects.length > 0) {
        const hit = intersects[0].object;
        if (hit === moon) {
          onSelectDomain('lunar');
        } else if (hit === earth) {
          onSelectDomain('earth');
        }
      }
    };
    canvasEl.addEventListener('click', onClick);

    // 8. Animation Loop
    let animId: number;
    let orbitAngle = 0.8;

    const animate = () => {
      animId = requestAnimationFrame(animate);

      // Rotations
      earth.rotation.y += 0.0012;
      clouds.rotation.y += 0.0018;

      // Orbit
      orbitAngle += 0.0022;
      const moonX = Math.cos(orbitAngle) * 5.4;
      const moonZ = Math.sin(orbitAngle) * 2.8 - 0.4;
      const moonY = planetY + 2.2 + Math.sin(orbitAngle * 0.5) * 0.5;
      moon.position.set(moonX, moonY, moonZ);
      moon.rotation.y += 0.0015;

      // Celestial Domain Focus Transition
      const isLunar = activeDomainRef.current === 'lunar';
      if (isLunar) {
        earth.position.x += (-3.4 - earth.position.x) * 0.05;
        earth.position.z += (-2.5 - earth.position.z) * 0.05;
        clouds.position.x = earth.position.x;
        clouds.position.z = earth.position.z;
        atmosphere.position.x = earth.position.x;
        atmosphere.position.z = earth.position.z;

        moon.scale.set(1.45, 1.45, 1.45);
        atmoMat.opacity = 0.08;
      } else {
        earth.position.x += (0 - earth.position.x) * 0.05;
        earth.position.z += (0 - earth.position.z) * 0.05;
        clouds.position.x = earth.position.x;
        clouds.position.z = earth.position.z;
        atmosphere.position.x = earth.position.x;
        atmosphere.position.z = earth.position.z;

        moon.scale.set(1.0, 1.0, 1.0);
        atmoMat.opacity = 0.22;
      }

      camera.position.x += (camTargetX - camera.position.x) * 0.05;
      camera.position.y += (camTargetY - camera.position.y) * 0.05;
      camera.lookAt(0, -0.4, 0);

      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight || 620;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(animId);
      canvasEl.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('mousemove', onMouseMove);
      canvasEl.removeEventListener('click', onClick);
      window.removeEventListener('resize', onResize);
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [onSelectDomain]);

  return <div ref={mountRef} className="w-full h-full absolute inset-0 bg-[#05070d]" />;
};
