import { useMemo, type CSSProperties } from "react";

type EchoDirection = "right" | "left" | "up" | "down" | "diagonal";

interface EchoTextProps {
  text: string;
  echoes?: number;
  offset?: number;
  direction?: EchoDirection;
  fade?: number;
  blur?: number;
  tint?: string;
  duration?: number;
  fontSize?: string;
  fontWeight?: number;
  color?: string;
  className?: string;
}

const directionVectors: Record<EchoDirection, { x: number; y: number }> = {
  right: { x: 1, y: 0 },
  left: { x: -1, y: 0 },
  up: { x: 0, y: -1 },
  down: { x: 0, y: 1 },
  diagonal: { x: 0.72, y: 0.72 },
};

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}

export function EchoText({
  text,
  echoes = 7,
  offset = 22,
  direction = "right",
  fade = 0.72,
  blur = 2.4,
  tint = "#6ea8ea",
  duration = 920,
  fontSize = "clamp(1.25rem, 2.2vw, 1.75rem)",
  fontWeight = 600,
  color = "#10243f",
  className = "",
}: EchoTextProps) {
  const echoCount = clamp(Math.round(echoes), 0, 12);
  const indexes = useMemo(
    () => Array.from({ length: echoCount }, (_, index) => index + 1),
    [echoCount],
  );
  const vector = directionVectors[direction];

  const rootStyle = {
    "--echo-duration": `${Math.max(180, duration)}ms`,
    "--echo-offset": `${clamp(offset, 0, 64)}px`,
    "--echo-vector-x": vector.x,
    "--echo-vector-y": vector.y,
    color,
    fontSize,
    fontWeight,
  } as CSSProperties;

  return (
    <span
      className={`echo-text ${className}`.trim()}
      style={rootStyle}
      data-testid="boot-echo-text"
    >
      {[...indexes].reverse().map((index) => {
        const depth = echoCount > 0 ? index / echoCount : 0;
        const layerStyle = {
          "--echo-depth": index,
          "--echo-opacity": Math.pow(clamp(fade, 0.1, 0.94), index),
          "--echo-blur": `${clamp(blur, 0, 8) * depth}px`,
          "--echo-x": `${vector.x * clamp(offset, 0, 64) * index}px`,
          "--echo-y": `${vector.y * clamp(offset, 0, 64) * index}px`,
          color: tint,
        } as CSSProperties;

        return (
          <span
            aria-hidden="true"
            className="echo-text__echo"
            key={index}
            style={layerStyle}
          >
            {text}
          </span>
        );
      })}
      <span
        className="echo-text__front"
        style={
          {
            "--echo-front-x": `${vector.x * clamp(offset, 0, 64) * 0.35}px`,
            "--echo-front-y": `${vector.y * clamp(offset, 0, 64) * 0.35}px`,
          } as CSSProperties
        }
      >
        {text}
      </span>
    </span>
  );
}
