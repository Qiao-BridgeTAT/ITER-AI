import {
  AnimatePresence,
  motion,
  useMotionValue,
  useReducedMotion,
  useSpring,
  useTransform,
} from "motion/react";
import {
  MouseEvent as ReactMouseEvent,
  ReactNode,
  useRef,
  useState,
} from "react";

import "./Dock.css";

export type DockItemData = {
  icon: ReactNode;
  label: string;
  onClick: () => void;
  active?: boolean;
  suppressLabel?: boolean;
};

type DockProps = {
  items: DockItemData[];
  ariaLabel: string;
  baseItemSize?: number;
  magnification?: number;
  distance?: number;
};

type DockItemProps = DockItemData & {
  mouseX: ReturnType<typeof useMotionValue<number>>;
  baseItemSize: number;
  magnification: number;
  distance: number;
};

function DockItem({
  icon,
  label,
  onClick,
  active,
  suppressLabel,
  mouseX,
  baseItemSize,
  magnification,
  distance,
}: DockItemProps) {
  const itemRef = useRef<HTMLButtonElement>(null);
  const [showLabel, setShowLabel] = useState(false);
  const reduceMotion = useReducedMotion();
  const pointerDistance = useTransform(mouseX, (value) => {
    const bounds = itemRef.current?.getBoundingClientRect();
    return bounds ? value - (bounds.left + bounds.width / 2) : distance + 1;
  });
  const targetSize = useTransform(
    pointerDistance,
    [-distance, 0, distance],
    [baseItemSize, magnification, baseItemSize],
  );
  const animatedSize = useSpring(targetSize, {
    mass: 0.12,
    stiffness: 250,
    damping: 18,
  });

  return (
    <motion.button
      ref={itemRef}
      className={`travel-dock-item${active ? " is-active" : ""}`}
      type="button"
      aria-label={label}
      aria-pressed={active}
      style={{
        width: reduceMotion ? baseItemSize : animatedSize,
        height: reduceMotion ? baseItemSize : animatedSize,
      }}
      onClick={onClick}
      onFocus={() => setShowLabel(true)}
      onBlur={() => setShowLabel(false)}
      onMouseEnter={() => setShowLabel(true)}
      onMouseLeave={() => setShowLabel(false)}
    >
      <span className="travel-dock-icon" aria-hidden="true">
        {icon}
      </span>
      <AnimatePresence>
        {showLabel && !suppressLabel ? (
          <motion.span
            className="travel-dock-label"
            initial={reduceMotion ? false : { opacity: 0, y: -3 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduceMotion ? undefined : { opacity: 0, y: -3 }}
            transition={{ duration: reduceMotion ? 0 : 0.16 }}
          >
            {label}
          </motion.span>
        ) : null}
      </AnimatePresence>
    </motion.button>
  );
}

export function Dock({
  items,
  ariaLabel,
  baseItemSize = 38,
  magnification = 50,
  distance = 104,
}: DockProps) {
  const mouseX = useMotionValue(Number.POSITIVE_INFINITY);

  const handleMouseMove = (event: ReactMouseEvent<HTMLDivElement>) => {
    mouseX.set(event.clientX);
  };

  return (
    <div
      className="travel-dock"
      role="toolbar"
      aria-label={ariaLabel}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => mouseX.set(Number.POSITIVE_INFINITY)}
    >
      {items.map((item) => (
        <DockItem
          key={item.label}
          {...item}
          mouseX={mouseX}
          baseItemSize={baseItemSize}
          magnification={magnification}
          distance={distance}
        />
      ))}
    </div>
  );
}
