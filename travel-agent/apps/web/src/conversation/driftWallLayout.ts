export const getFittedPlaneScale = (
  containerWidth: number,
  columns: number,
  tileWidth: number,
  gap: number
) => {
  const safeColumns = Math.max(1, Math.floor(columns));
  const planeWidth = safeColumns * (tileWidth + gap);
  const sideSafeArea = Math.max(40, gap * 3);
  const availableWidth = Math.max(1, containerWidth - sideSafeArea);

  return Math.min(1.18, Math.max(0.72, availableWidth / planeWidth));
};

export const getPlaneHorizontalOffset = ({
  width,
  height,
  scale,
  tilt,
  turn,
  roll,
  depth,
  perspective
}: {
  width: number;
  height: number;
  scale: number;
  tilt: number;
  turn: number;
  roll: number;
  depth: number;
  perspective: number;
}) => {
  const radians = Math.PI / 180;
  const [rx, ry, rz] = [tilt, turn, roll].map((angle) => angle * radians);
  // Match the CSS transform order. Perspective makes a geometrically centered
  // plane look off-center, so balance its projected edges without a fixed nudge.
  const corners = [-1, 1].flatMap((sx) =>
    [-1, 1].map((sy) => {
      const x = (sx * width * Math.cos(rz) - sy * height * Math.sin(rz)) / 2;
      const y = (sx * width * Math.sin(rz) + sy * height * Math.cos(rz)) / 2;
      const z =
        y * Math.sin(rx) +
        (-x * Math.sin(ry) - depth * Math.cos(ry)) * Math.cos(rx);
      return {
        x: scale * (x * Math.cos(ry) - depth * Math.sin(ry)),
        factor: perspective / (perspective - z)
      };
    })
  );
  let offset = 0;
  for (let pass = 0; pass < 2; pass++) {
    const projected = corners.map(({ x, factor }) => ({
      x: (x + offset) * factor,
      factor
    }));
    const left = projected.reduce((a, b) => (a.x < b.x ? a : b));
    const right = projected.reduce((a, b) => (a.x > b.x ? a : b));
    offset -= (left.x + right.x) / (left.factor + right.factor);
  }
  return offset;
};

export const getLoopPlaneHeight = (
  containerHeight: number,
  planeScale: number,
  tileHeight: number,
  gap: number
) => {
  const visibleHeight = Math.max(1, containerHeight) / planeScale;
  // Keep a full row (or more on tall walls) outside both viewport edges,
  // including room for the existing perspective and pointer tilt.
  const overscan = Math.max(tileHeight + gap, visibleHeight * 0.25);
  return Math.ceil(visibleHeight + overscan * 2);
};

export const getLoopTrackLayout = (
  planeHeight: number,
  itemCount: number,
  tileHeight: number,
  gap: number
) => {
  const unit = Math.max(1, tileHeight + gap);
  const copyHeight = Math.max(1, itemCount) * unit;
  // The track can move up by one whole copy before wrapping. That entire
  // copy must be EXTRA to the content covering the fixed-height plane.
  const copies = Math.max(2, Math.ceil(planeHeight / copyHeight) + 1);
  return { copyHeight, copies };
};

export const wrapLoopOffset = (offset: number, copyHeight: number) =>
  ((offset % copyHeight) + copyHeight) % copyHeight;
