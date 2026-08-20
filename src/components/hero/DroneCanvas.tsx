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
                mat.roughness = 0.5;
                mat.metalness = 0.05;
                mat.envMapIntensity = 0.0;
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

// High-fidelity procedural 3D drone fallback while GLTF loads or if GLTF fails
function ProceduralDroneMesh({ techProgress = 0, mouseRef }: { techProgress: number; mouseRef: React.MutableRefObject<MousePosition> }) {
  const groupRef = useRef<THREE.Group>(null);
  const rotorsRef = useRef<THREE.Mesh[]>([]);

  useFrame((state, delta) => {
    if (!groupRef.current) return;
    const time = state.clock.getElapsedTime();
    const t = Math.max(0, Math.min(1, techProgress));

    const mouseWeight = Math.max(0, 1.0 - t * 1.5);
    const mx = mouseRef.current.x * mouseWeight;
    const my = mouseRef.current.y * mouseWeight;

    // Flight bezier positioning
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

    const hoverScale = 1.0 - t * 0.70;
    const hoverX = Math.sin(time * 1.5) * 0.05 * hoverScale;
    const hoverY = Math.cos(time * 2.0) * 0.06 * hoverScale;
    const hoverZ = Math.sin(time * 1.2) * 0.04 * hoverScale;

    groupRef.current.position.set(pathX + hoverX, pathY + hoverY, pathZ + hoverZ);

    const heroRoll = -mx * 0.32 + Math.sin(time * 2.5) * 0.04;
    const heroPitch = my * 0.22 + Math.cos(time * 1.8) * 0.03;
    const techYaw = 1.05;

    groupRef.current.rotation.z = THREE.MathUtils.lerp(heroRoll, 0, t);
    groupRef.current.rotation.x = THREE.MathUtils.lerp(heroPitch, 0, t);
    groupRef.current.rotation.y = THREE.MathUtils.lerp(0, techYaw, t);

    // High-speed rotor spin
    rotorsRef.current.forEach((rotor, idx) => {
      if (rotor) {
        const spinDir = idx % 2 === 0 ? 1 : -1;
        rotor.rotation.y += delta * 45.0 * spinDir;
      }
    });
  });

  const armAngles = [Math.PI / 4, 3 * Math.PI / 4, 5 * Math.PI / 4, 7 * Math.PI / 4];
  const armLength = 0.75;
  const currentScale = THREE.MathUtils.lerp(1.6, 1.25, Math.max(0, Math.min(1, techProgress)));

  return (
    <group ref={groupRef} scale={[currentScale, currentScale, currentScale]}>
      {/* Central Aerodynamic Fuselage */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[0.55, 0.16, 0.55]} />
        <meshStandardMaterial color="#0f172a" metalness={0.85} roughness={0.25} />
      </mesh>

      {/* Top Status LED Dome */}
      <mesh position={[0, 0.09, 0]}>
        <sphereGeometry args={[0.13, 16, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#38bdf8" emissive="#0284c7" emissiveIntensity={1.2} roughness={0.1} />
      </mesh>

      {/* Front Optical Gimbal Camera */}
      <mesh position={[0, -0.06, 0.28]}>
        <sphereGeometry args={[0.11, 16, 16]} />
        <meshStandardMaterial color="#020617" metalness={0.9} roughness={0.2} />
      </mesh>
      <mesh position={[0, -0.06, 0.35]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.06, 0.06, 0.05, 16]} />
        <meshBasicMaterial color="#22d3ee" />
      </mesh>

      {/* 4 Carbon Arms + Motors + Rotor Blades + Guards */}
      {armAngles.map((angle, idx) => {
        const mx = Math.cos(angle) * armLength;
        const mz = Math.sin(angle) * armLength;
        return (
          <group key={idx}>
            {/* Carbon Arm Tube */}
            <mesh
              position={[Math.cos(angle) * (armLength / 2), 0, Math.sin(angle) * (armLength / 2)]}
              rotation={[0, -angle, Math.PI / 2]}
            >
              <cylinderGeometry args={[0.035, 0.035, armLength, 8]} />
              <meshStandardMaterial color="#334155" metalness={0.8} roughness={0.3} />
            </mesh>

            {/* Brushless Motor Housing */}
            <mesh position={[mx, 0.04, mz]}>
              <cylinderGeometry args={[0.08, 0.08, 0.1, 16]} />
              <meshStandardMaterial color="#64748b" metalness={0.9} roughness={0.2} />
            </mesh>

            {/* Rotor Guard Torus */}
            <mesh position={[mx, 0.08, mz]} rotation={[Math.PI / 2, 0, 0]}>
              <torusGeometry args={[0.26, 0.014, 8, 24]} />
              <meshStandardMaterial color="#0ea5e9" transparent opacity={0.65} />
            </mesh>

            {/* Spinning Dual Propeller */}
            <mesh
              ref={(el) => { if (el) rotorsRef.current[idx] = el; }}
              position={[mx, 0.09, mz]}
            >
              <boxGeometry args={[0.48, 0.008, 0.045]} />
              <meshStandardMaterial color="#020617" roughness={0.4} />
            </mesh>
          </group>
        );
      })}

      {/* Landing Skids */}
      {[-0.24, 0.24].map((x, i) => (
        <mesh key={i} position={[x, -0.14, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.02, 0.02, 0.65, 8]} />
          <meshStandardMaterial color="#334155" metalness={0.7} />
        </mesh>
      ))}
    </group>
  );
}

// Low-poly fallback wireframe drone while loading GLTF
function DroneFallback({ mouseRef, techProgress = 0 }: { mouseRef: React.MutableRefObject<MousePosition>; techProgress?: number }) {
  return <ProceduralDroneMesh techProgress={techProgress} mouseRef={mouseRef} />;
}

class DroneErrorBoundary extends React.Component<
  { fallback: React.ReactNode; children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  componentDidCatch(error: any) {
    console.warn('GLTF Drone load failed, using procedural 3D model:', error);
  }
  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
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
        <ambientLight intensity={3.0} />
        <hemisphereLight args={['#ffffff', '#334155', 2.5]} />
        <directionalLight position={[10, 15, 10]} intensity={4.0} />
        <directionalLight position={[-10, 10, -5]} intensity={2.5} color="#e0f2fe" />
        <directionalLight position={[0, -5, 5]} intensity={1.5} color="#bae6fd" />
        <pointLight position={[0, 4, 4]} intensity={2.5} color="#38bdf8" />
        <pointLight position={[0, -4, 2]} intensity={1.5} color="#0d9488" />

        <DroneErrorBoundary fallback={<ProceduralDroneMesh techProgress={techProgress} mouseRef={mouseRef} />}>
          <Suspense fallback={<DroneFallback mouseRef={mouseRef} techProgress={techProgress} />}>
            <DroneModel mouseRef={mouseRef} techProgress={techProgress} modelPath={modelPath} />
          </Suspense>
        </DroneErrorBoundary>
      </Canvas>
    </div>
  );
};
