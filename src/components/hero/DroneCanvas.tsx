import React, { useRef, useEffect, useMemo, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { MousePosition } from './types';

// Preload drone GLTF model immediately for instantaneous rendering
useGLTF.preload('/models/drone_design/scene.gltf');

interface DroneModelProps {
  mouseRef: React.MutableRefObject<MousePosition>;
  techProgress?: number; // 0.0 = Hero, 1.0 = Technology page settled
  modelPath: string;
}

function DroneModel({ mouseRef, techProgress = 0, modelPath }: DroneModelProps) {
  const groupRef = useRef<THREE.Group>(null);
  const droneRef = useRef<THREE.Group>(null);
  const gltfRotorsRef = useRef<THREE.Object3D[]>([]);

  // Load the GLTF model and clone to avoid shared-reference rendering issues
  const { scene: originalScene } = useGLTF(modelPath);
  const scene = useMemo(() => originalScene.clone(true), [originalScene]);

  // Target values for smooth lerping
  const targetPos = useRef(new THREE.Vector3(0, 0, 0));
  const targetLookAt = useRef(new THREE.Vector3(0, 0, 5));
  const currentLookAt = useRef(new THREE.Vector3(0, 0, 5));

  // 360 Trick Maneuver state (Randomized interval flourish)
  const lastTrickTimeRef = useRef(0);
  const nextTrickIntervalRef = useRef(7 + Math.random() * 6); // initial trick after 7-13s
  const isTrickActiveRef = useRef(false);
  const trickProgressRef = useRef(0);

  // Identify internal rotor / propeller / plane nodes and tune PBR materials for bright lighting
  useEffect(() => {
    if (scene) {
      const foundRotors: THREE.Object3D[] = [];
      scene.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          if (mesh.material) {
            const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
            mats.forEach((mat: any) => {
              if (mat.isMeshStandardMaterial || mat.isMeshPhysicalMaterial) {
                mat.roughness = 0.35;
                mat.metalness = 0.3;
                mat.envMapIntensity = 1.6;
                mat.needsUpdate = true;
              }
            });
          }
        }

        const name = child.name.toLowerCase();
        if (
          name.includes('plane') ||
          name.includes('rotor') ||
          name.includes('fan') ||
          name.includes('propeller') ||
          name.includes('blade')
        ) {
          foundRotors.push(child);
        }
      });

      if (foundRotors.length === 0) {
        scene.traverse((child) => {
          if ((child as THREE.Mesh).isMesh && child.name !== 'helicopter_box_0') {
            foundRotors.push(child);
          }
        });
      }

      gltfRotorsRef.current = foundRotors;
    }
  }, [scene]);

  useFrame((state, delta) => {
    if (!groupRef.current) return;

    const time = state.clock.getElapsedTime();
    const t = Math.max(0, Math.min(1, techProgress)); // 0.0 (Hero) to 1.0 (Tech)

    // Cursor tracking influence: 1.0 on Hero page, 0.0 on Technology page
    const mouseWeight = Math.max(0, 1.0 - t * 1.5);
    const mx = mouseRef.current.x * mouseWeight;
    const my = mouseRef.current.y * mouseWeight;

    // 1. FLIGHT-DOWN ANIMATION: Drone swoops down into the dark blue box on Tech page
    const p0x = 1.25 + mx * 0.45;
    const p0y = my * 0.70;
    const p0z = mx * 0.20;

    const p1x = -0.1;
    const p1y = 2.4;
    const p1z = 0.1;

    const p2x = -1.32;
    const p2y = 0.0;
    const p2z = 0.1;

    const oneMinusT = 1 - t;
    const pathX = oneMinusT * oneMinusT * p0x + 2 * oneMinusT * t * p1x + t * t * p2x;
    const pathY = oneMinusT * oneMinusT * p0y + 2 * oneMinusT * t * p1y + t * t * p2y;
    const pathZ = oneMinusT * oneMinusT * p0z + 2 * oneMinusT * t * p1z + t * t * p2z;

    // 2. Idle Rotation Flourish
    if (!isTrickActiveRef.current) {
      if (time - lastTrickTimeRef.current > nextTrickIntervalRef.current) {
        isTrickActiveRef.current = true;
        trickProgressRef.current = 0;
      }
    }

    let trickYawOffset = 0;
    let trickHoverOffset = 0;

    if (isTrickActiveRef.current) {
      trickProgressRef.current += delta * 0.75;
      if (trickProgressRef.current >= 1.0) {
        isTrickActiveRef.current = false;
        trickProgressRef.current = 0;
        lastTrickTimeRef.current = time;
        nextTrickIntervalRef.current = 7 + Math.random() * 7;
      } else {
        const p = trickProgressRef.current;
        const easeP = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
        trickYawOffset = easeP * Math.PI * 2;
        trickHoverOffset = Math.sin(p * Math.PI) * 0.18;
      }
    }

    // 3. Idle Hover Bobbing (Drone stays strictly inside the box)
    const hoverScale = 1.0 - t * 0.70;
    const hoverX = Math.sin(time * 1.5) * 0.05 * hoverScale;
    const hoverY = (Math.cos(time * 2.0) * 0.06 + Math.sin(time * 0.8) * 0.03 + trickHoverOffset) * hoverScale;
    const hoverZ = Math.sin(time * 1.2) * 0.04 * hoverScale;

    targetPos.current.set(pathX + hoverX, pathY + hoverY, pathZ + hoverZ);
    groupRef.current.position.lerp(targetPos.current, delta * 4.0);

    // 4. ORIENTATION & ROTATION
    const targetPointX = 0.35 + mx * 4.0;
    const targetPointY = my * 3.0;
    const targetPointZ = 4.0;

    targetLookAt.current.set(targetPointX, targetPointY, targetPointZ);
    currentLookAt.current.lerp(targetLookAt.current, delta * 4.0);

    if (t > 0.05) {
      groupRef.current.rotation.x = THREE.MathUtils.lerp(groupRef.current.rotation.x, 0, delta * 5.0);
      groupRef.current.rotation.y = THREE.MathUtils.lerp(groupRef.current.rotation.y, 0, delta * 5.0);
      groupRef.current.rotation.z = THREE.MathUtils.lerp(groupRef.current.rotation.z, 0, delta * 5.0);
    } else {
      groupRef.current.lookAt(currentLookAt.current);
    }

    const heroRoll = -mx * 0.32 + Math.sin(time * 2.5) * 0.04;
    const heroPitch = my * 0.22 + Math.cos(time * 1.8) * 0.03;

    const techYaw = 1.05 + trickYawOffset;
    const techPitch = 0.0;
    const techRoll = 0.0;

    const finalRoll = THREE.MathUtils.lerp(heroRoll, techRoll, t);
    const finalPitch = THREE.MathUtils.lerp(heroPitch, techPitch, t);
    const finalYaw = THREE.MathUtils.lerp(trickYawOffset, techYaw, t);

    if (droneRef.current) {
      droneRef.current.rotation.z = THREE.MathUtils.lerp(droneRef.current.rotation.z, finalRoll, delta * 5.0);
      droneRef.current.rotation.x = THREE.MathUtils.lerp(droneRef.current.rotation.x, finalPitch, delta * 5.0);
      droneRef.current.rotation.y = THREE.MathUtils.lerp(droneRef.current.rotation.y, finalYaw, delta * 5.0);
    }

    // 5. High-RPM Rotor Spin
    gltfRotorsRef.current.forEach((rotor, idx) => {
      const spinDir = idx % 2 === 0 ? 1 : -1;
      rotor.rotation.z += delta * 60.0 * spinDir;
      rotor.rotation.x += delta * 40.0 * spinDir;
      rotor.rotation.y += delta * 50.0 * spinDir;
    });

    // 6. Camera Position
    const heroCamY = 0.0;
    const heroCamZ = 4.0;
    const techCamY = 2.2;
    const techCamZ = 3.0;

    state.camera.position.y = THREE.MathUtils.lerp(heroCamY, techCamY, t);
    state.camera.position.z = THREE.MathUtils.lerp(heroCamZ, techCamZ, t);
    state.camera.lookAt(0, 0, 0);
  });

  const handlePointerDown = () => {
    if (!isTrickActiveRef.current) {
      isTrickActiveRef.current = true;
      trickProgressRef.current = 0;
    }
  };

  const currentScale = THREE.MathUtils.lerp(2.70, 2.15, Math.max(0, Math.min(1, techProgress)));

  return (
    <group ref={groupRef} position={[0, 0, 0]}>
      {/* 3D GLTF Drone Model with Dynamic Scale */}
      <group ref={droneRef} scale={[currentScale, currentScale, currentScale]} onPointerDown={handlePointerDown}>
        <primitive object={scene} />
      </group>
    </group>
  );
}

// Low-poly fallback wireframe drone while loading GLTF
function DroneFallback() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.8;
      meshRef.current.rotation.x = Math.sin(Date.now() * 0.002) * 0.1;
    }
  });

  return (
    <mesh ref={meshRef}>
      <octahedronGeometry args={[2.2, 2]} />
      <meshStandardMaterial color="#0f172a" wireframe transparent opacity={0.3} />
    </mesh>
  );
}

interface DroneCanvasProps {
  mouseRef: React.MutableRefObject<MousePosition>;
  techProgress?: number;
  modelPath?: string;
  className?: string;
}

export const DroneCanvas: React.FC<DroneCanvasProps> = ({
  mouseRef,
  techProgress = 0,
  modelPath = '/models/drone_design/scene.gltf',
  className = ''
}) => {
  return (
    <div className={`drone-canvas-container ${className}`} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        camera={{ position: [0, 0, 4.0], fov: 45 }}
        style={{ background: 'transparent' }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
      >
        <ambientLight intensity={2.2} />
        <hemisphereLight args={['#ffffff', '#0f172a', 2.0]} />
        <directionalLight position={[10, 15, 10]} intensity={3.0} />
        <directionalLight position={[-10, -5, -5]} intensity={1.5} color="#94a3b8" />
        <pointLight position={[0, 5, 4]} intensity={2.0} color="#38bdf8" />
        <pointLight position={[0, -5, 2]} intensity={1.2} color="#0d9488" />

        <Suspense fallback={<DroneFallback />}>
          <DroneModel mouseRef={mouseRef} techProgress={techProgress} modelPath={modelPath} />
        </Suspense>
      </Canvas>
    </div>
  );
};
