import React, { useRef, useEffect, useState } from 'react';
import { DroneCanvas } from './DroneCanvas';
import { TextArc } from './TextArc';
import { TechSection } from '../tech/TechSection';
import { HeroSectionProps, HeroTextItem, MousePosition } from './types';
import styles from './hero.module.css';

const DEFAULT_HERO_ITEMS: HeroTextItem[] = [
  {
    id: '1',
    number: '01',
    tagline: 'THERMAL & OPTICAL INTELLIGENCE',
    heading: 'Autonomous Aerial Search',
    description: 'Real-time multi-agent spatial scanning with thermal sensors for rapid survivor localization.'
  },
  {
    id: '2',
    number: '02',
    tagline: '3D ELEVATION NAVIGATION',
    heading: 'Collision-Free Flight Dynamics',
    description: 'Continuous magnetic repulsion fields prevent inter-drone collisions and clear obstacles safely.'
  },
  {
    id: '3',
    number: '03',
    tagline: 'BLOCK-BY-BLOCK EXECUTION',
    heading: '100% Area Search Guarantee',
    description: 'Dynamic backfilling automatically reassigns active drones to cover unsearched sectors immediately.'
  },
  {
    id: '4',
    number: '04',
    tagline: 'DEBRIS & BUILDING DETECTION',
    heading: 'Thermal Occlusion Scanning',
    description: 'Penetrates debris and structures to detect hidden survivors with zero ground truth leakage.'
  },
  {
    id: '5',
    number: '05',
    tagline: 'INSTANT SQUAD RECOVERY',
    heading: 'Tactical Resilient Fleet',
    description: 'When a drone unit goes offline, adjacent squad drones instantly recalculate search boundaries.'
  }
];

export const HeroSection: React.FC<HeroSectionProps> = ({
  modelPath = '/models/drone_design/scene.gltf',
  items = DEFAULT_HERO_ITEMS,
  className = ''
}) => {
  const mouseRef = useRef<MousePosition>({ x: 0, y: 0 });
  const trackRef = useRef<HTMLDivElement>(null);

  const [itemProgress, setItemProgress] = useState(0);
  const [techTransition, setTechTransition] = useState(0);
  const [techSeqProgress, setTechSeqProgress] = useState(0);

  const targetHeroProgressRef = useRef(0);
  const currentHeroProgressRef = useRef(0);

  const targetTechTransRef = useRef(0);
  const currentTechTransRef = useRef(0);

  const targetTechSeqRef = useRef(0);
  const currentTechSeqRef = useRef(0);

  const rafIdRef = useRef<number | null>(null);
  const isLoopRunningRef = useRef(false);

  useEffect(() => {
    const updateProgress = () => {
      const heroDiff = targetHeroProgressRef.current - currentHeroProgressRef.current;
      const transDiff = targetTechTransRef.current - currentTechTransRef.current;
      const seqDiff = targetTechSeqRef.current - currentTechSeqRef.current;

      let isMoving = false;

      if (Math.abs(heroDiff) > 0.0002) {
        currentHeroProgressRef.current += heroDiff * 0.15;
        setItemProgress(currentHeroProgressRef.current);
        isMoving = true;
      } else {
        currentHeroProgressRef.current = targetHeroProgressRef.current;
        setItemProgress(targetHeroProgressRef.current);
      }

      if (Math.abs(transDiff) > 0.0002) {
        currentTechTransRef.current += transDiff * 0.15;
        setTechTransition(currentTechTransRef.current);
        isMoving = true;
      } else {
        currentTechTransRef.current = targetTechTransRef.current;
        setTechTransition(targetTechTransRef.current);
      }

      if (Math.abs(seqDiff) > 0.0002) {
        currentTechSeqRef.current += seqDiff * 0.15;
        setTechSeqProgress(currentTechSeqRef.current);
        isMoving = true;
      } else {
        currentTechSeqRef.current = targetTechSeqRef.current;
        setTechSeqProgress(targetTechSeqRef.current);
      }

      if (isMoving) {
        rafIdRef.current = requestAnimationFrame(updateProgress);
      } else {
        isLoopRunningRef.current = false;
        rafIdRef.current = null;
      }
    };

    const handleScroll = () => {
      if (!trackRef.current) return;
      const rect = trackRef.current.getBoundingClientRect();
      const totalScroll = rect.height - window.innerHeight;
      if (totalScroll <= 0) return;

      const currentScroll = -rect.top;
      const rawProgress = Math.max(0, Math.min(1, currentScroll / totalScroll));

      // 3-Phase Scroll Mapping:
      // Phase 1: Hero Arc (rawProgress 0.00 to 0.38)
      const heroNorm = Math.max(0, Math.min(1, rawProgress / 0.38));
      targetHeroProgressRef.current = heroNorm * (items.length - 1);

      // Phase 2: Technology Page Slide-Up Overlay (rawProgress 0.38 to 0.48)
      const transNorm = Math.max(0, Math.min(1, (rawProgress - 0.38) / 0.10));
      targetTechTransRef.current = transNorm;

      // Phase 3: Technology Headings Smooth Scroll Sequence (rawProgress 0.48 to 1.00)
      const seqNorm = Math.max(0, Math.min(1, (rawProgress - 0.48) / 0.52));
      targetTechSeqRef.current = seqNorm;

      if (!isLoopRunningRef.current) {
        isLoopRunningRef.current = true;
        rafIdRef.current = requestAnimationFrame(updateProgress);
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();

    return () => {
      window.removeEventListener('scroll', handleScroll);
      if (rafIdRef.current !== null) cancelAnimationFrame(rafIdRef.current);
      isLoopRunningRef.current = false;
    };
  }, [items.length]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const x = (e.clientX / window.innerWidth) * 2 - 1;
      const y = -(e.clientY / window.innerHeight) * 2 + 1;
      mouseRef.current = { x, y };
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <div ref={trackRef} className={styles.heroTrack} style={{ height: '800vh' }}>
      <section className={`${styles.heroContainer} ${className}`}>


        {/* Left ~50% Hero Text Arc (Fixed on Hero page until Technology covers it) */}
        <div className={styles.leftSection} style={{ opacity: Math.max(0, 1 - techTransition * 2.0) }}>
          <TextArc items={items} progress={itemProgress} />
        </div>

        {/* 3D Drone Canvas (Positioned on right side of Hero page, Z-Index 30 above backdrop) */}
        <div className={styles.rightCanvasSection} style={{ pointerEvents: techTransition > 0.8 ? 'none' : 'auto' }}>
          <DroneCanvas mouseRef={mouseRef} techProgress={techTransition} modelPath={modelPath} />
        </div>

        {/* Technology Page (Slides up from below, transparent left side so drone is crisp & hovering) */}
        <div style={{ position: 'absolute', inset: 0, zIndex: 15, pointerEvents: 'none' }}>
          <TechSection techTransition={techTransition} techSequenceProgress={techSeqProgress} />
        </div>

        {/* Bottom Status Ticker */}
        <footer className={styles.bottomTicker}>
          <div className={styles.tickerText}>
            <span className={styles.tickerDot}>●</span> FLEET STATUS: ONLINE
            <span style={{ opacity: 0.4 }}>|</span> 5 AUTONOMOUS DRONES
            <span style={{ opacity: 0.4 }}>|</span> 100% AREA COVERAGE GUARANTEE
            <span style={{ opacity: 0.4 }}>|</span> THERMAL OCCLUSION ACTIVE
          </div>
        </footer>
      </section>
    </div>
  );
};

export default HeroSection;
