import React, { useState, useRef, useEffect, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF, useAnimations, Float, OrbitControls } from '@react-three/drei';
import { MessageSquare, X, Send, Bot, Sparkles, Shield, Cpu, RefreshCw, ChevronRight } from 'lucide-react';
import * as THREE from 'three';

// Preload robot GLTF model
useGLTF.preload('/models/robot/scene.gltf');

function FlyingRobotModel({ onClick }: { onClick: () => void }) {
  const groupRef = useRef<THREE.Group>(null);
  const { scene, animations } = useGLTF('/models/robot/scene.gltf');
  const { actions } = useAnimations(animations, groupRef);

  useEffect(() => {
    // Play first available animation if present
    const firstAnim = Object.keys(actions)[0];
    if (firstAnim && actions[firstAnim]) {
      actions[firstAnim]?.reset().fadeIn(0.5).play();
    }
  }, [actions]);

  useFrame((state) => {
    if (groupRef.current) {
      // Gentle floating bobbing centered at midpoint (-0.65)
      const t = state.clock.getElapsedTime();
      groupRef.current.position.y = -0.65 + Math.sin(t * 2) * 0.05;
      groupRef.current.rotation.y = Math.sin(t * 0.8) * 0.3 + Math.PI / 8;
    }
  });

  return (
    <group
      ref={groupRef}
      onClick={onClick}
      onPointerOver={() => { document.body.style.cursor = 'pointer'; }}
      onPointerOut={() => { document.body.style.cursor = 'auto'; }}
      scale={[1.5, 1.5, 1.5]}
      position={[0, -0.65, 0]}
    >
      <primitive object={scene} />
    </group>
  );
}

function FallbackRobot() {
  const meshRef = useRef<THREE.Mesh>(null);
  useFrame((state) => {
    if (meshRef.current) {
      const t = state.clock.getElapsedTime();
      meshRef.current.position.y = Math.sin(t * 2) * 0.1;
      meshRef.current.rotation.y += 0.02;
    }
  });

  return (
    <mesh ref={meshRef}>
      <octahedronGeometry args={[1.2, 0]} />
      <meshStandardMaterial color="#06b6d4" wireframe emissive="#0284c7" emissiveIntensity={0.5} />
    </mesh>
  );
}

interface Message {
  id: string;
  sender: 'bot' | 'user';
  text: string;
  timestamp: string;
}

const INITIAL_MESSAGES: Message[] = [
  {
    id: '1',
    sender: 'bot',
    text: "Greetings! I am VIHANG's Autonomous Tactical AI Assistant. How can I assist you with multi-agent search & rescue operations or simulation analytics today?",
    timestamp: 'Just now'
  }
];

const PRESET_PROMPTS = [
  "What is VIHANG?",
  "How does PPO Area Mapping work?",
  "What sensors do drones use?",
  "How to launch live simulation?"
];

const KNOWLEDGE_BASE: Record<string, string> = {
  "what is vihang?": "VIHANG is an AI-powered autonomous multi-agent search & rescue swarm framework. It utilizes PPO Reinforcement Learning to navigate complex disaster zones, detect survivors, and generate 3D point cloud maps.",
  "how does ppo area mapping work?": "Area Mapping (V2 & V3) uses Proximal Policy Optimization (PPO) with 360-degree LiDAR and thermal sensors to rapidly map unexplored grid sectors while avoiding obstacles and maintaining swarm separation.",
  "what sensors do drones use?": "Each drone is equipped with RGB optical cameras, 360° LiDAR rangefinders, thermal occlusion detectors, and GPS/IMU navigation modules.",
  "how to launch live simulation?": "Scroll down to the 'Live Interactive Simulation' section on this page! Select your environment parameters (Grid Size, Swarm Count, Model Checkpoint) and click 'RUN LIVE SIMULATION'."
};

interface RobotChatWidgetProps {
  isVisible?: boolean;
  isOpen?: boolean;
  onToggleChat?: (open: boolean) => void;
}

export default function RobotChatWidget({
  isVisible = true,
  isOpen: externalIsOpen,
  onToggleChat
}: RobotChatWidgetProps) {
  const [internalIsOpen, setInternalIsOpen] = useState(false);
  const isOpen = externalIsOpen !== undefined ? externalIsOpen : internalIsOpen;

  const setIsOpen = (val: boolean | ((prev: boolean) => boolean)) => {
    const nextVal = typeof val === 'function' ? val(isOpen) : val;
    if (externalIsOpen === undefined) {
      setInternalIsOpen(nextVal);
    }
    if (onToggleChat) {
      onToggleChat(nextVal);
    }
  };

  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [hasHovered, setHasHovered] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatDrawerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen]);

  // Lock scroll wheel inside chat window to prevent background page scrolling
  useEffect(() => {
    const drawerEl = chatDrawerRef.current;
    if (!isOpen || !drawerEl) return;

    const handleWheel = (e: WheelEvent) => {
      e.stopPropagation();
      
      // Find scrollable child if any
      const target = e.target as HTMLElement | null;
      const scrollable = target?.closest('.custom-chat-scroll') as HTMLElement | null;
      
      if (scrollable) {
        const isScrollableY = scrollable.scrollHeight > scrollable.clientHeight;
        const isScrollableX = scrollable.scrollWidth > scrollable.clientWidth;
        
        if (isScrollableY) {
          const atTop = scrollable.scrollTop === 0 && e.deltaY < 0;
          const atBottom = Math.abs(scrollable.scrollHeight - scrollable.clientHeight - scrollable.scrollTop) < 2 && e.deltaY > 0;
          if (atTop || atBottom) {
            e.preventDefault();
          }
        } else if (isScrollableX) {
          const atLeft = scrollable.scrollLeft === 0 && e.deltaX < 0;
          const atRight = Math.abs(scrollable.scrollWidth - scrollable.clientWidth - scrollable.scrollLeft) < 2 && e.deltaX > 0;
          if (atLeft || atRight) {
            e.preventDefault();
          }
        } else {
          e.preventDefault();
        }
      } else {
        e.preventDefault();
      }
    };

    drawerEl.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      drawerEl.removeEventListener('wheel', handleWheel);
    };
  }, [isOpen]);

  if (!isVisible) return null;

  const handleSend = (textToSend?: string) => {
    const query = (textToSend || input).trim();
    if (!query) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      sender: 'user',
      text: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!textToSend) setInput('');
    setIsTyping(true);

    setTimeout(() => {
      const lower = query.toLowerCase();
      let replyText = "I have logged your request in VIHANG tactical telemetry. You can run custom multi-agent benchmarks directly in the simulation panel below!";
      
      for (const [key, val] of Object.entries(KNOWLEDGE_BASE)) {
        if (lower.includes(key.replace('?', '')) || key.includes(lower)) {
          replyText = val;
          break;
        }
      }

      if (lower.includes('hello') || lower.includes('hi')) {
        replyText = "Hello commander! Ready to deploy drone swarms or review mission analytics?";
      }

      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: replyText,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages((prev) => [...prev, botMsg]);
      setIsTyping(false);
    }, 900);
  };

  return (
    <>
      {/* Floating 3D Robot Container at Bottom-Right (Only visible when chat drawer is closed) */}
      {!isOpen && (
        <div
          style={{
            position: 'fixed',
            bottom: 24,
            right: 24,
            zIndex: 9999,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-end',
            pointerEvents: 'none'
          }}
        >
          {/* Floating 3D Robot Trigger Button with Circle Background */}
          <div
            onMouseEnter={() => setHasHovered(true)}
            onMouseLeave={() => setHasHovered(false)}
            onClick={() => setIsOpen(true)}
            style={{
              width: 140,
              height: 140,
              borderRadius: '50%',
              background: 'radial-gradient(circle at 40% 35%, #1a3c54, #132a3b)',
              border: '1.5px solid rgba(255, 255, 255, 0.15)',
              boxShadow: '0 10px 30px rgba(0, 0, 0, 0.4), inset 0 0 15px rgba(255, 255, 255, 0.05)',
              backdropFilter: 'blur(16px)',
              WebkitBackdropFilter: 'blur(16px)',
              cursor: 'pointer',
              pointerEvents: 'auto',
              position: 'relative',
              transition: 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
              transform: hasHovered ? 'scale(1.08)' : 'scale(1)'
            }}
          >
            <div style={{ width: '100%', height: '100%', borderRadius: '50%', overflow: 'hidden' }}>
              <Canvas camera={{ position: [0, 0, 3.4], fov: 45 }}>
                <ambientLight intensity={2.0} />
                <directionalLight position={[5, 5, 5]} intensity={2.8} />
                <pointLight position={[-5, -5, -5]} color="#2dd4bf" intensity={3.5} />
                <Suspense fallback={<FallbackRobot />}>
                  <Float speed={3} rotationIntensity={0.6} floatIntensity={0.6}>
                    <FlyingRobotModel onClick={() => setIsOpen(true)} />
                  </Float>
                </Suspense>
              </Canvas>
            </div>
          </div>
        </div>
      )}

      {/* Chat Interface Drawer / Modal */}
      {isOpen && (
        <div
          ref={chatDrawerRef}
          style={{
            position: 'fixed',
            bottom: 120,
            right: 24,
            width: 'min(420px, calc(100vw - 32px))',
            height: '560px',
            maxHeight: 'calc(100vh - 150px)',
            background: 'rgba(15, 23, 42, 0.95)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            borderRadius: 24,
            boxShadow: '0 20px 50px rgba(0, 0, 0, 0.6), 0 0 30px rgba(14, 165, 233, 0.15)',
            zIndex: 9998,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            overscrollBehavior: 'contain',
            animation: 'chatSlideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1)'
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: '16px 20px',
              background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9))',
              borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div
                style={{
                  width: 38,
                  height: 38,
                  borderRadius: 12,
                  background: 'linear-gradient(135deg, #0284c7, #0d9488)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 0 12px rgba(14, 165, 233, 0.4)'
                }}
              >
                <Bot size={22} color="#ffffff" />
              </div>
              <div>
                <h4
                  style={{
                    margin: 0,
                    fontFamily: "'Space Grotesk', sans-serif",
                    fontSize: '0.95rem',
                    fontWeight: 700,
                    color: '#f8fafc',
                    letterSpacing: '-0.01em'
                  }}
                >
                  VIHANG AI ASSISTANT
                </h4>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      background: '#10b981',
                      boxShadow: '0 0 6px #10b981'
                    }}
                  />
                  <span
                    style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: '0.68rem',
                      color: '#94a3b8',
                      textTransform: 'uppercase'
                    }}
                  >
                    Autonomous Drone Link Active
                  </span>
                </div>
              </div>
            </div>

            <button
              onClick={() => setIsOpen(false)}
              style={{
                background: 'rgba(255, 255, 255, 0.05)',
                border: 'none',
                borderRadius: '50%',
                width: 32,
                height: 32,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#94a3b8',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              <X size={18} />
            </button>
          </div>

          {/* Messages Area */}
          <div
            className="custom-chat-scroll"
            style={{
              flex: 1,
              padding: 16,
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 12
            }}
          >
            {messages.map((msg) => (
              <div
                key={msg.id}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: msg.sender === 'user' ? 'flex-end' : 'flex-start'
                }}
              >
                <div
                  style={{
                    maxWidth: '85%',
                    padding: '12px 16px',
                    borderRadius: msg.sender === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                    background: msg.sender === 'user'
                      ? 'linear-gradient(135deg, #0284c7, #0369a1)'
                      : 'rgba(30, 41, 59, 0.7)',
                    border: msg.sender === 'user' ? 'none' : '1px solid rgba(255, 255, 255, 0.08)',
                    color: '#f8fafc',
                    fontFamily: "'Inter', sans-serif",
                    fontSize: '0.85rem',
                    lineHeight: 1.45,
                    boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                  }}
                >
                  {msg.text}
                </div>
                <span
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: '0.65rem',
                    color: '#64748b',
                    marginTop: 4,
                    padding: '0 4px'
                  }}
                >
                  {msg.timestamp}
                </span>
              </div>
            ))}

            {isTyping && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 12px', background: 'rgba(30, 41, 59, 0.5)', borderRadius: 12, width: 'fit-content' }}>
                <span style={{ fontSize: '0.75rem', color: '#38bdf8', fontFamily: "'JetBrains Mono', monospace" }}>Processing tactical query...</span>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Quick Prompts */}
          <div
            className="custom-chat-scroll"
            style={{
              padding: '8px 12px',
              display: 'flex',
              gap: 6,
              overflowX: 'auto',
              borderTop: '1px solid rgba(255, 255, 255, 0.05)',
              background: 'rgba(15, 23, 42, 0.5)'
            }}
          >
            {PRESET_PROMPTS.map((prompt, i) => (
              <button
                key={i}
                onClick={() => handleSend(prompt)}
                style={{
                  whiteSpace: 'nowrap',
                  background: 'rgba(56, 189, 248, 0.1)',
                  border: '1px solid rgba(56, 189, 248, 0.25)',
                  borderRadius: 999,
                  padding: '5px 12px',
                  color: '#38bdf8',
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: '0.7rem',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  flexShrink: 0
                }}
              >
                {prompt}
              </button>
            ))}
          </div>

          {/* Input Bar */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            style={{
              padding: 12,
              background: '#0f172a',
              borderTop: '1px solid rgba(255, 255, 255, 0.08)',
              display: 'flex',
              gap: 8
            }}
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask VIHANG AI Assistant..."
              style={{
                flex: 1,
                background: 'rgba(30, 41, 59, 0.8)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: 12,
                padding: '10px 14px',
                color: '#f8fafc',
                fontFamily: "'Inter', sans-serif",
                fontSize: '0.85rem',
                outline: 'none'
              }}
            />
            <button
              type="submit"
              disabled={!input.trim()}
              style={{
                background: input.trim() ? 'linear-gradient(135deg, #0284c7, #0d9488)' : 'rgba(255, 255, 255, 0.05)',
                border: 'none',
                borderRadius: 12,
                width: 42,
                height: 42,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: input.trim() ? '#ffffff' : '#64748b',
                cursor: input.trim() ? 'pointer' : 'not-allowed',
                transition: 'all 0.2s ease'
              }}
            >
              <Send size={18} />
            </button>
          </form>
        </div>
      )}

      <style>{`
        @keyframes chatSlideUp {
          from { opacity: 0; transform: translateY(20px) scale(0.95); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        .custom-chat-scroll::-webkit-scrollbar {
          height: 4px;
          width: 4px;
        }
        .custom-chat-scroll::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.6);
        }
        .custom-chat-scroll::-webkit-scrollbar-thumb {
          background: rgba(56, 189, 248, 0.35);
          border-radius: 4px;
        }
        .custom-chat-scroll::-webkit-scrollbar-thumb:hover {
          background: rgba(56, 189, 248, 0.7);
        }
        .custom-chat-scroll {
          scrollbar-width: thin;
          scrollbar-color: rgba(56, 189, 248, 0.35) rgba(15, 23, 42, 0.6);
        }
      `}</style>
    </>
  );
}
