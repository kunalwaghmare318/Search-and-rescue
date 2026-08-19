export interface HeroTextItem {
  id: string;
  number: string;
  tagline: string;
  heading: string;
  description: string;
}

export interface MousePosition {
  x: number;
  y: number;
}

export interface HeroSectionProps {
  modelPath?: string;
  items?: HeroTextItem[];
  className?: string;
}
