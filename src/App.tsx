import React, { useState, useEffect } from 'react';
import HeroSection from './components/hero';
import RobotChatWidget from './components/RobotChatWidget';
import { Maximize2, Minimize2, ShieldAlert, Cpu, Radio, Award, Navigation, Zap, Compass } from 'lucide-react';

export function App() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeSection, setActiveSection] = useState<'overview' | 'tech' | 'demo'>('overview');
  const [isChatOpen, setIsChatOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const demoEl = document.getElementById('demo');
      const overviewEl = document.getElementById('overview');
      if (demoEl && demoEl.getBoundingClientRect().top < window.innerHeight * 0.5) {
        setActiveSection('demo');
      } else if (overviewEl) {
        const totalScroll = overviewEl.offsetHeight - window.innerHeight;
        const currentScroll = window.scrollY - overviewEl.offsetTop;
        const rawProgress = totalScroll > 0 ? currentScroll / totalScroll : 0;
        if (rawProgress > 0.35) {
          setActiveSection('tech');
        } else {
          setActiveSection('overview');
        }
      } else {
        setActiveSection('overview');
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <main style={{ background: '#0f172a', minHeight: '100vh', margin: 0, padding: 0, color: '#f8fafc', overflowX: 'clip' }}>
      {/* Fixed Global Navigation */}
      <nav style={{
        position: 'fixed',
        top: 28,
        left: 0,
        width: '100%',
        padding: '0 calc(4vw + 20px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        zIndex: 9999,
        boxSizing: 'border-box',
        pointerEvents: 'none'
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', pointerEvents: 'auto' }}>
          <span style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: '1.15rem',
            fontWeight: 700,
            letterSpacing: '-0.02em',
            color: activeSection === 'overview' ? '#0f172a' : '#ffffff',
            transition: 'color 0.3s ease'
          }}>VIHANG</span>
          <span style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: '0.72rem',
            fontWeight: 500,
            letterSpacing: '0.12em',
            textTransform: 'uppercase' as const,
            color: activeSection === 'overview' ? '#64748b' : 'rgba(255,255,255,0.9)',
            marginTop: 2,
            transition: 'color 0.3s ease'
          }}>SEARCH & RESCUE · AUTONOMOUS DRONE SQUAD</span>
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          background: activeSection === 'overview' ? 'rgba(241,245,249,0.85)' : 'rgba(15,23,42,0.75)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          padding: '6px 8px',
          borderRadius: 999,
          border: activeSection === 'overview' ? '1px solid rgba(226,232,240,0.8)' : '1px solid rgba(255,255,255,0.15)',
          pointerEvents: isChatOpen ? 'none' : 'auto',
          opacity: isChatOpen ? 0 : 1,
          boxShadow: '0 4px 20px -2px rgba(15,23,42,0.15)',
          transition: 'all 0.3s ease'
        }}>
          {([
            { id: 'overview' as const, label: 'OVERVIEW', onClick: () => window.scrollTo({ top: 0, behavior: 'smooth' }) },
            { id: 'tech' as const, label: 'TECHNOLOGY', onClick: () => {
              const overviewEl = document.getElementById('overview');
              if (overviewEl) {
                const totalScroll = overviewEl.offsetHeight - window.innerHeight;
                window.scrollTo({ top: overviewEl.offsetTop + totalScroll * 0.55, behavior: 'smooth' });
              }
            }},
            { id: 'demo' as const, label: 'SIMULATION', onClick: () => document.getElementById('demo')?.scrollIntoView({ behavior: 'smooth' }), isSim: true },
          ]).map(btn => (
            <a
              key={btn.id}
              href={`#${btn.id}`}
              onClick={(e) => { e.preventDefault(); btn.onClick(); }}
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: '0.75rem',
                fontWeight: btn.isSim ? 700 : 600,
                letterSpacing: '0.08em',
                color: btn.isSim ? '#ffffff' : (activeSection === btn.id ? (activeSection === 'overview' ? '#0f172a' : '#ffffff') : (activeSection === 'overview' ? '#475569' : '#94a3b8')),
                padding: '8px 16px',
                borderRadius: 999,
                textDecoration: 'none',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                background: btn.isSim
                  ? 'linear-gradient(135deg, #0d9488, #0284c7)'
                  : (activeSection === btn.id ? (activeSection === 'overview' ? '#ffffff' : 'rgba(255,255,255,0.15)') : 'transparent'),
                boxShadow: btn.isSim
                  ? '0 4px 14px rgba(13,148,136,0.35)'
                  : (activeSection === btn.id && activeSection === 'overview' ? '0 2px 8px rgba(15,23,42,0.08)' : 'none'),
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6
              }}
            >
              {btn.isSim && <span style={{
                display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
                background: '#38bdf8', boxShadow: '0 0 8px #38bdf8',
                animation: 'pulseGlow 1.5s infinite ease-in-out'
              }} />}
              {btn.label}
            </a>
          ))}
        </div>
      </nav>

      <style>{`@keyframes pulseGlow { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.5);opacity:0.6} }`}</style>

      {/* 1. Hero Section (Pinned Scroll Canvas + Text Arc) */}
      <div id="overview">
        <HeroSection />
      </div>

      {/* 2. Live Interactive Simulation Showcase Section (#demo) */}
      <section id="demo" style={{
        position: 'relative',
        padding: '100px 4vw',
        background: 'radial-gradient(ellipse at top, #1e293b 0%, #0f172a 100%)',
        borderTop: '1px solid rgba(255, 255, 255, 0.08)',
        overflow: 'hidden'
      }}>
        <div style={{ maxWidth: isExpanded ? '100%' : '1280px', margin: '0 auto', transition: 'max-width 0.4s ease' }}>
          <div style={{ textAlign: 'center', marginBottom: '40px' }}>
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 16px',
              borderRadius: '999px',
              background: 'rgba(13, 148, 136, 0.15)',
              border: '1px solid rgba(13, 148, 136, 0.3)',
              color: '#2dd4bf',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '0.8rem',
              fontWeight: 600,
              letterSpacing: '0.1em',
              marginBottom: '16px'
            }}>
              <Radio size={14} /> LIVE MARL INFERENCE SUITE
            </div>

            <h2 style={{
              fontFamily: 'Space Grotesk, sans-serif',
              fontSize: '2.75rem',
              fontWeight: 700,
              letterSpacing: '-0.02em',
              color: '#ffffff',
              margin: '0 0 14px 0'
            }}>
              Autonomous Squad Simulation Engine
            </h2>
            <p style={{
              maxWidth: '700px',
              margin: '0 auto',
              color: '#94a3b8',
              fontSize: '1.05rem',
              lineHeight: 1.6
            }}>
              Directly control the live trained PPO search and rescue drones inside this interactive 3D disaster workspace.
            </p>
          </div>

          {/* Interactive Viewport Frame */}
          <div style={{
            position: 'relative',
            borderRadius: '20px',
            border: '1px solid rgba(13, 148, 136, 0.3)',
            background: '#0f172a',
            boxShadow: '0 25px 60px -15px rgba(0, 0, 0, 0.7), 0 0 40px rgba(13, 148, 136, 0.15)',
            overflow: 'hidden',
            transition: 'all 0.4s ease'
          }}>
            {/* Viewport Top Control Bar */}
            <div style={{
              padding: '14px 20px',
              background: 'rgba(30, 41, 59, 0.85)',
              backdropFilter: 'blur(12px)',
              borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '12px',
              flexWrap: 'wrap'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ display: 'flex', gap: '6px' }}>
                  <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#ef4444' }} />
                  <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#f59e0b' }} />
                  <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#10b981' }} />
                </div>
                <span style={{
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: '0.82rem',
                  fontWeight: 600,
                  color: '#2dd4bf',
                  letterSpacing: '0.05em'
                }}>
                  VIHANG // 3D_SIMULATION_CANVAS
                </span>
                <span style={{
                  fontSize: '0.7rem',
                  fontFamily: 'JetBrains Mono, monospace',
                  padding: '2px 8px',
                  borderRadius: '4px',
                  background: 'rgba(16, 185, 129, 0.15)',
                  color: '#34d399',
                  border: '1px solid rgba(16, 185, 129, 0.3)'
                }}>
                  ● LIVE INFERENCE ONLINE
                </span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '6px 14px',
                    borderRadius: '8px',
                    background: 'rgba(15, 23, 42, 0.7)',
                    border: '1px solid rgba(255, 255, 255, 0.15)',
                    color: '#f8fafc',
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: '0.78rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                >
                  {isExpanded ? (
                    <>
                      <Minimize2 size={14} /> Compact Frame
                    </>
                  ) : (
                    <>
                      <Maximize2 size={14} /> Expand Viewport
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Embedded Live Simulation Canvas Container */}
            <div style={{
              position: 'relative',
              width: '100%',
              height: isExpanded ? '85vh' : '680px',
              background: '#0f172a',
              transition: 'height 0.4s ease'
            }}>
              <iframe
                src="/simulation.html"
                title="VIHANG 3D Live Simulation Engine"
                style={{
                  width: '100%',
                  height: '100%',
                  border: 'none',
                  display: 'block',
                  background: '#0f172a'
                }}
              />
            </div>
          </div>
        </div>
      </section>

      {/* 3. Technology Architecture Section (#tech) */}
      <section id="tech" style={{
        padding: '120px 5vw',
        background: '#0f172a',
        borderTop: '1px solid rgba(255, 255, 255, 0.08)'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: '60px' }}>
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 16px',
              borderRadius: '999px',
              background: 'rgba(2, 132, 199, 0.15)',
              border: '1px solid rgba(2, 132, 199, 0.3)',
              color: '#38bdf8',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '0.8rem',
              fontWeight: 600,
              letterSpacing: '0.1em',
              marginBottom: '16px'
            }}>
              <Cpu size={14} /> SYSTEM ARCHITECTURE
            </div>
            <h2 style={{
              fontFamily: 'Space Grotesk, sans-serif',
              fontSize: '2.5rem',
              fontWeight: 700,
              color: '#ffffff',
              margin: '0 0 16px 0'
            }}>
              Engineered for Zero-Gaps Disaster Response
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '1rem', maxWidth: '600px', margin: '0 auto' }}>
              Built with decentralized Multi-Agent Reinforcement Learning (MARL), advanced thermal sensor simulation, and continuous physical collision dynamics.
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '24px'
          }}>
            {/* Tech Card 1 */}
            <div style={{
              padding: '32px',
              borderRadius: '16px',
              background: 'rgba(30, 41, 59, 0.4)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              backdropFilter: 'blur(10px)'
            }}>
              <div style={{
                width: '48px',
                height: '48px',
                borderRadius: '12px',
                background: 'rgba(13, 148, 136, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#2dd4bf',
                marginBottom: '20px'
              }}>
                <Zap size={24} />
              </div>
              <h3 style={{ fontFamily: 'Space Grotesk, sans-serif', fontSize: '1.25rem', fontWeight: 600, color: '#ffffff', marginBottom: '10px' }}>
                Thermal & Optical Fusion
              </h3>
              <p style={{ color: '#94a3b8', fontSize: '0.92rem', lineHeight: 1.6, margin: 0 }}>
                Open thermal sensor arrays locate visible victims immediately, while deep-penetration sensors track heat signatures occluded inside collapsed structures.
              </p>
            </div>

            {/* Tech Card 2 */}
            <div style={{
              padding: '32px',
              borderRadius: '16px',
              background: 'rgba(30, 41, 59, 0.4)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              backdropFilter: 'blur(10px)'
            }}>
              <div style={{
                width: '48px',
                height: '48px',
                borderRadius: '12px',
                background: 'rgba(2, 132, 199, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#38bdf8',
                marginBottom: '20px'
              }}>
                <Navigation size={24} />
              </div>
              <h3 style={{ fontFamily: 'Space Grotesk, sans-serif', fontSize: '1.25rem', fontWeight: 600, color: '#ffffff', marginBottom: '10px' }}>
                Magnetic Repulsion Navigation
              </h3>
              <p style={{ color: '#94a3b8', fontSize: '0.92rem', lineHeight: 1.6, margin: 0 }}>
                Continuous potential fields exert outward repulsion between squad units and obstacle walls, eliminating mid-air collisions even in dense urban corridors.
              </p>
            </div>

            {/* Tech Card 3 */}
            <div style={{
              padding: '32px',
              borderRadius: '16px',
              background: 'rgba(30, 41, 59, 0.4)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              backdropFilter: 'blur(10px)'
            }}>
              <div style={{
                width: '48px',
                height: '48px',
                borderRadius: '12px',
                background: 'rgba(245, 158, 11, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fbbf24',
                marginBottom: '20px'
              }}>
                <ShieldAlert size={24} />
              </div>
              <h3 style={{ fontFamily: 'Space Grotesk, sans-serif', fontSize: '1.25rem', fontWeight: 600, color: '#ffffff', marginBottom: '10px' }}>
                Tactical Squad Resilience
              </h3>
              <p style={{ color: '#94a3b8', fontSize: '0.92rem', lineHeight: 1.6, margin: 0 }}>
                If a drone suffers catastrophic failure or battery loss, active squad members instantly re-partition the search grid without human intervention.
              </p>
            </div>

            {/* Tech Card 4 */}
            <div style={{
              padding: '32px',
              borderRadius: '16px',
              background: 'rgba(30, 41, 59, 0.4)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              backdropFilter: 'blur(10px)'
            }}>
              <div style={{
                width: '48px',
                height: '48px',
                borderRadius: '12px',
                background: 'rgba(16, 185, 129, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#34d399',
                marginBottom: '20px'
              }}>
                <Compass size={24} />
              </div>
              <h3 style={{ fontFamily: 'Space Grotesk, sans-serif', fontSize: '1.25rem', fontWeight: 600, color: '#ffffff', marginBottom: '10px' }}>
                3D Volumetric Terrain Mapping
              </h3>
              <p style={{ color: '#94a3b8', fontSize: '0.92rem', lineHeight: 1.6, margin: 0 }}>
                Simultaneously constructs point cloud maps during flight to evaluate building damage, road blockages, and optimal ground crew entry routes.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Performance Metrics & Benchmark (#metrics) */}
      <section id="metrics" style={{
        padding: '120px 5vw',
        background: 'radial-gradient(ellipse at bottom, #1e293b 0%, #0f172a 100%)',
        borderTop: '1px solid rgba(255, 255, 255, 0.08)'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: '60px' }}>
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 16px',
              borderRadius: '999px',
              background: 'rgba(16, 185, 129, 0.15)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              color: '#34d399',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '0.8rem',
              fontWeight: 600,
              letterSpacing: '0.1em',
              marginBottom: '16px'
            }}>
              <Award size={14} /> BENCHMARK EVALUATION RESULTS
            </div>
            <h2 style={{
              fontFamily: 'Space Grotesk, sans-serif',
              fontSize: '2.5rem',
              fontWeight: 700,
              color: '#ffffff',
              margin: '0 0 16px 0'
            }}>
              Trained Model Performance Metrics (PPO V17)
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '1rem', maxWidth: '600px', margin: '0 auto' }}>
              Empirically evaluated across 10 deterministic disaster scenarios with randomized survivor placements.
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '24px'
          }}>
            {/* Stat Box 1 */}
            <div style={{
              padding: '36px 24px',
              borderRadius: '16px',
              background: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid rgba(13, 148, 136, 0.3)',
              textAlign: 'center'
            }}>
              <div style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '3.25rem',
                fontWeight: 700,
                color: '#2dd4bf',
                lineHeight: 1,
                marginBottom: '12px'
              }}>
                33.4%
              </div>
              <div style={{
                fontFamily: 'Space Grotesk, sans-serif',
                fontSize: '1rem',
                fontWeight: 600,
                color: '#ffffff',
                marginBottom: '6px'
              }}>
                Grid Coverage Rate
              </div>
              <p style={{ color: '#64748b', fontSize: '0.85rem', margin: 0 }}>
                High-priority sector exploration efficiency
              </p>
            </div>

            {/* Stat Box 2 */}
            <div style={{
              padding: '36px 24px',
              borderRadius: '16px',
              background: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid rgba(2, 132, 199, 0.3)',
              textAlign: 'center'
            }}>
              <div style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '3.25rem',
                fontWeight: 700,
                color: '#38bdf8',
                lineHeight: 1,
                marginBottom: '12px'
              }}>
                45.0s
              </div>
              <div style={{
                fontFamily: 'Space Grotesk, sans-serif',
                fontSize: '1rem',
                fontWeight: 600,
                color: '#ffffff',
                marginBottom: '6px'
              }}>
                Average Rescue Speed
              </div>
              <p style={{ color: '#64748b', fontSize: '0.85rem', margin: 0 }}>
                First survivor localization threshold
              </p>
            </div>

            {/* Stat Box 3 */}
            <div style={{
              padding: '36px 24px',
              borderRadius: '16px',
              background: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              textAlign: 'center'
            }}>
              <div style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '3.25rem',
                fontWeight: 700,
                color: '#34d399',
                lineHeight: 1,
                marginBottom: '12px'
              }}>
                29.8
              </div>
              <div style={{
                fontFamily: 'Space Grotesk, sans-serif',
                fontSize: '1rem',
                fontWeight: 600,
                color: '#ffffff',
                marginBottom: '6px'
              }}>
                Lowest Collision Rate
              </div>
              <p style={{ color: '#64748b', fontSize: '0.85rem', margin: 0 }}>
                Inter-agent & obstacle avoidance
              </p>
            </div>

            {/* Stat Box 4 */}
            <div style={{
              padding: '36px 24px',
              borderRadius: '16px',
              background: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              textAlign: 'center'
            }}>
              <div style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '3.25rem',
                fontWeight: 700,
                color: '#fbbf24',
                lineHeight: 1,
                marginBottom: '12px'
              }}>
                100%
              </div>
              <div style={{
                fontFamily: 'Space Grotesk, sans-serif',
                fontSize: '1rem',
                fontWeight: 600,
                color: '#ffffff',
                marginBottom: '6px'
              }}>
                Squad Dynamic Backfilling
              </div>
              <p style={{ color: '#64748b', fontSize: '0.85rem', margin: 0 }}>
                Automatic recovery on unit failure
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 5. Footer */}
      <footer style={{
        padding: '40px 5vw',
        background: '#020617',
        borderTop: '1px solid rgba(255, 255, 255, 0.08)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '20px'
      }}>
        <div>
          <div style={{ fontFamily: 'Space Grotesk, sans-serif', fontWeight: 700, fontSize: '1.1rem', color: '#ffffff' }}>
            VIHANG
          </div>
          <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem', color: '#64748b', margin: '4px 0 0 0' }}>
            Multi-Agent Search & Rescue Autonomous Flight Platform
          </p>
        </div>

        <div style={{ display: 'flex', gap: '20px' }}>
          <a
            href="#overview"
            onClick={(e) => { e.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
            style={{ color: '#94a3b8', textDecoration: 'none', fontSize: '0.85rem' }}
          >
            Overview
          </a>
          <a
            href="#demo"
            onClick={(e) => { e.preventDefault(); document.getElementById('demo')?.scrollIntoView({ behavior: 'smooth' }); }}
            style={{ color: '#2dd4bf', textDecoration: 'none', fontSize: '0.85rem', fontWeight: 600 }}
          >
            Live Simulation
          </a>
          <a
            href="#tech"
            onClick={(e) => { e.preventDefault(); document.getElementById('tech')?.scrollIntoView({ behavior: 'smooth' }); }}
            style={{ color: '#94a3b8', textDecoration: 'none', fontSize: '0.85rem' }}
          >
            Technology
          </a>
          <a
            href="#metrics"
            onClick={(e) => { e.preventDefault(); document.getElementById('metrics')?.scrollIntoView({ behavior: 'smooth' }); }}
            style={{ color: '#94a3b8', textDecoration: 'none', fontSize: '0.85rem' }}
          >
            Metrics
          </a>
        </div>
      </footer>
      <RobotChatWidget
        isVisible={activeSection === 'tech'}
        isOpen={isChatOpen}
        onToggleChat={(open) => setIsChatOpen(open)}
      />
    </main>
  );
}

export default App;
