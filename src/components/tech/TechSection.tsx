import React from 'react';
import styles from './tech.module.css';

export interface TechSectionItem {
  title?: string;
  desc?: string;
  lines?: string[];
}

export const DEFAULT_SECTIONS: TechSectionItem[] = [
  {
    title: '20km Transmission Range',
    desc: 'Long-range SAR ops, mountain/forest search zones, no signal drop'
  },
  {
    title: '55min Flight Time',
    desc: 'Long mission window, less battery-swap downtime during rescue'
  },
  {
    title: 'RTK 1cm Positioning Accuracy',
    desc: 'Pin-point survivor location, critical for rescue team dispatch'
  },
  {
    title: 'IP55 + -20°C to 50°C Rating',
    desc: 'Deploy in flood/storm/extreme terrain, not lab-only toy'
  },
  {
    title: 'Multi-Payload (Thermal + Zoom + LiDAR)',
    desc: 'One drone = heat detect + visual confirm + 3D map, matches your LiDAR area-mapping mode directly'
  }
];

interface TechSectionProps {
  techTransition: number; // 0.0 to 1.0 (overlay slide up)
  techSequenceProgress: number; // 0.0 to 1.0 (scroll sequence progress)
  sections?: TechSectionItem[];
}

export const TechSection: React.FC<TechSectionProps> = ({
  techTransition,
  techSequenceProgress,
  sections = DEFAULT_SECTIONS
}) => {
  // Slide-up overlay translateY percentage (100% when transition=0, 0% when transition=1)
  const slideUpPercent = Math.max(0, (1 - Math.min(1, techTransition)) * 100);

  const numSections = sections.length;
  const windowSize = 1.0 / numSections;

  // Render sentence with word-wrapping protection + noticeable Lightning Blue loading trail
  const renderSentenceWithBlueTrail = (text: string, revealProgress: number, exitProgress: number) => {
    const words = text.split(' ');
    const chars = text.split('');
    const totalChars = chars.length;

    // Number of revealed characters (0 to totalChars)
    const numRevealed = Math.floor(revealProgress * (totalChars + 1));

    // Number of exited characters (0 to totalChars)
    const numExited = Math.floor(exitProgress * (totalChars + 1));

    // Noticeable Lightning Blue loading trail length (3 characters remain glowing blue while loading)
    const BLUE_TRAIL_LENGTH = 3;

    let globalCharIndex = 0;

    return words.map((word, wordIdx) => {
      const wordCharSpans = word.split('').map((char) => {
        const idx = globalCharIndex;
        globalCharIndex++;

        const isExited = idx < numExited;
        const isRevealed = idx < numRevealed && !isExited;

        // Character is in the active electric lightning blue loading trail!
        const isBlueLoading =
          isRevealed &&
          idx >= numRevealed - BLUE_TRAIL_LENGTH &&
          revealProgress < 1.0;

        if (!isRevealed) {
          // Invisible before loading or after exit
          return (
            <span key={idx} style={{ opacity: 0, display: 'inline-block' }}>
              {char}
            </span>
          );
        }

        if (isBlueLoading) {
          // Vibrant Electric Lightning Blue Glow for Noticeable Loading Trail!
          return (
            <span
              key={idx}
              className={styles.lightningChar}
              style={{
                color: '#00f0ff',
                textShadow: '0 0 22px #00f0ff, 0 0 45px #00f0ff, 0 0 65px #0284c7',
                display: 'inline-block',
                fontWeight: 800
              }}
            >
              {char}
            </span>
          );
        }

        // Settled Pure Bright White Text
        return (
          <span
            key={idx}
            className={styles.whiteChar}
            style={{
              color: '#ffffff',
              textShadow: '0 0 25px rgba(255, 255, 255, 0.95), 0 0 50px rgba(255, 255, 255, 0.5)',
              display: 'inline-block'
            }}
          >
            {char}
          </span>
        );
      });

      // Track space character index after word
      const spaceIndex = globalCharIndex;
      globalCharIndex++;

      const isSpaceExited = spaceIndex < numExited;
      const isSpaceRevealed = spaceIndex < numRevealed && !isSpaceExited;

      return (
        <span
          key={wordIdx}
          style={{ whiteSpace: 'nowrap', display: 'inline-block' }}
        >
          {wordCharSpans}
          {wordIdx < words.length - 1 && (
            <span style={{ opacity: isSpaceRevealed ? 1 : 0, display: 'inline-block' }}>
              {'\u00A0'}
            </span>
          )}
        </span>
      );
    });
  };

  return (
    <div
      className={styles.techSticky}
      style={{
        transform: `translate3d(0, ${slideUpPercent.toFixed(2)}%, 0)`,
        opacity: techTransition > 0.02 ? 1 : 0,
        pointerEvents: techTransition > 0.1 ? 'auto' : 'none'
      }}
    >
      {/* Vivid Background Image Backdrop */}
      <div className={styles.techBackdrop} />

      {/* Left Side (42% width): Deep Blue-Teal Glass Box framing the 3D Drone */}
      <div className={styles.leftViewportSpace}>
        <div className={styles.droneCardBox} />
      </div>

      {/* Right Side (60% width): Scroll-driven Text Container */}
      <div className={styles.rightTextContainer}>
        <div className={styles.slideStage}>
          {sections.map((sec, idx) => {
            const start = idx * windowSize;
            const end = (idx + 1) * windowSize;

            const isCurrentWindow = techSequenceProgress >= start && techSequenceProgress <= end;
            const isBefore = techSequenceProgress < start;

            const localProgress = isBefore
              ? 0
              : Math.max(0, Math.min(1, (techSequenceProgress - start) / windowSize));

            let opacity = 0;
            let titleRevealProgress = 0;
            let descRevealProgress = 0;
            let exitProgress = 0;

            if (isCurrentWindow) {
              if (localProgress >= 0.06 && localProgress < 0.26) {
                opacity = 1;
                titleRevealProgress = (localProgress - 0.06) / 0.20;
                descRevealProgress = 0;
                exitProgress = 0;
              } else if (localProgress >= 0.26 && localProgress < 0.50) {
                opacity = 1;
                titleRevealProgress = 1;
                descRevealProgress = (localProgress - 0.26) / 0.24;
                exitProgress = 0;
              } else if (localProgress >= 0.50 && localProgress <= 0.72) {
                opacity = 1;
                titleRevealProgress = 1;
                descRevealProgress = 1;
                exitProgress = 0;
              } else if (localProgress > 0.72 && localProgress <= 0.94) {
                opacity = 1;
                titleRevealProgress = 1;
                descRevealProgress = 1;
                exitProgress = (localProgress - 0.72) / 0.22;
              }
            }

            const hasTitleDesc = sec.title && sec.desc;

            return (
              <div
                key={idx}
                className={styles.slideItem}
                style={{
                  opacity: opacity,
                  pointerEvents: opacity > 0.1 ? 'auto' : 'none',
                  visibility: opacity > 0.001 ? 'visible' : 'hidden'
                }}
              >
                {hasTitleDesc ? (
                  <div>
                    <h2 className={styles.lineTitle}>
                      {renderSentenceWithBlueTrail(sec.title!, titleRevealProgress, exitProgress)}
                    </h2>
                    <p className={styles.lineDesc}>
                      {renderSentenceWithBlueTrail(sec.desc!, descRevealProgress, exitProgress)}
                    </p>
                  </div>
                ) : (
                  (sec.lines || []).map((lineText, lineIdx) => (
                    <div key={lineIdx} className={styles.singleLineWrapper}>
                      <h2 className={styles.lineHeadline}>
                        {renderSentenceWithBlueTrail(lineText, titleRevealProgress, exitProgress)}
                      </h2>
                    </div>
                  ))
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default TechSection;
