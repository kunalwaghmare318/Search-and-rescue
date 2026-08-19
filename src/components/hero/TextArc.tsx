import React, { useEffect, useState, useRef } from 'react';
import { HeroTextItem } from './types';
import styles from './hero.module.css';

interface TextArcProps {
  items: HeroTextItem[];
  progress?: number;
  className?: string;
}

export const TextArc: React.FC<TextArcProps> = ({ items, progress, className = '' }) => {
  const [internalProgress, setInternalProgress] = useState(0);
  const targetProgressRef = useRef(0);
  const currentProgressRef = useRef(0);
  const rafIdRef = useRef<number | null>(null);

  const isLoopRunningRef = useRef(false);

  useEffect(() => {
    if (progress !== undefined) return;

    const updateProgress = () => {
      const diff = targetProgressRef.current - currentProgressRef.current;
      if (Math.abs(diff) > 0.0002) {
        currentProgressRef.current += diff * 0.12;
        setInternalProgress(currentProgressRef.current);
        rafIdRef.current = requestAnimationFrame(updateProgress);
      } else {
        currentProgressRef.current = targetProgressRef.current;
        setInternalProgress(targetProgressRef.current);
        isLoopRunningRef.current = false;
        rafIdRef.current = null;
      }
    };

    const handleScroll = () => {
      const scrollY = window.scrollY || window.pageYOffset || 0;
      targetProgressRef.current = scrollY / 260;

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
  }, [progress]);

  const activeProgress = progress !== undefined ? progress : internalProgress;
  const totalItems = items.length;
  // Radius of the quarter clock arc in pixels
  const radius = 280;

  return (
    <div className={`${styles.textArcWrapper} ${className}`}>
      {/* Quarter-circle SVG Guide Line (12 o'clock down through 9 o'clock and below) */}
      <svg className={styles.arcSvgGuide} viewBox="0 0 400 400" aria-hidden="true">
        <path
          d="M 280 20 A 280 280 0 0 0 0 300 A 280 280 0 0 0 280 580"
          fill="none"
          stroke="rgba(226, 232, 240, 0.5)"
          strokeWidth="1.5"
          strokeDasharray="4 6"
        />
      </svg>

      <div className={styles.arcItemsContainer}>
        {items.map((item, index) => {
          // Relative index from current scroll progress:
          // relativeIndex === 0  => EXACTLY at 9 o'clock position
          // relativeIndex > 0    => coming down from 12 o'clock (top position)
          // relativeIndex < 0    => going down below 9 o'clock (bottom position)
          let relativeIndex = index - activeProgress;

          // Angle along the quarter clock arc (9 o'clock is 0 rad anchor)
          // Positive angle = UP toward 12 o'clock (~75 deg max)
          // Negative angle = DOWN below 9 o'clock (~ -75 deg max)
          const angleRad = relativeIndex * (Math.PI / 2.4);

          // Polar to Cartesian relative to 9 o'clock anchor (0, 0)
          // 9 o'clock (relativeIndex = 0) => x = 0, y = 0
          const x = radius * (1 - Math.cos(angleRad));
          const y = -radius * Math.sin(angleRad); // negative y is UP (12 o'clock), positive y is DOWN
          const rotation = 0; // Exactly 0deg horizontal alignment (no tilt)

          // Opacity & Scale: ONLY text at 9 o'clock position is visible
          // Fades in smoothly as text slides down from 12 o'clock into 9 o'clock
          // Fades out smoothly as text moves down below 9 o'clock
          const distFrom9 = Math.abs(relativeIndex);
          const opacity = Math.max(0, Math.min(1, 1 - Math.pow(distFrom9 / 0.65, 2)));
          const scale = Math.max(0.85, 1.05 - distFrom9 * 0.15);

          const isActive = distFrom9 < 0.35;

          return (
            <div
              key={item.id}
              className={`${styles.arcItem} ${isActive ? styles.arcItemActive : ''}`}
              style={{
                transform: `translate3d(${x.toFixed(2)}px, ${y.toFixed(2)}px, 0px) scale(${scale.toFixed(3)})`,
                opacity: opacity.toFixed(3),
                visibility: opacity > 0.005 ? 'visible' : 'hidden',
                pointerEvents: opacity > 0.5 ? 'auto' : 'none',
                zIndex: isActive ? 10 : 1
              }}
            >
              {item.tagline && (
                <div className={styles.arcItemHeader}>
                  <span className={styles.itemTagline}>{item.tagline}</span>
                </div>
              )}
              <h2 className={styles.itemHeading}>{item.heading}</h2>
              <p className={styles.itemDescription}>{item.description}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
};

