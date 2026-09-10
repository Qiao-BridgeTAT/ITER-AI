import type { DriftWallItem } from "./DriftWall";

/** Each column loops a complete shuffled photo pool, never a one-photo slice. */
export function shuffledWallColumns(
  items: readonly DriftWallItem[],
  columns: number,
  seed: number,
): DriftWallItem[][] {
  const unique = [...new Map(items.map((item) => [item.image, item])).values()];
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const result: DriftWallItem[][] = [];
  for (let column = 0; column < columns; column++) {
    const shuffled = [...unique];
    for (let index = shuffled.length - 1; index > 0; index--) {
      const swap = Math.floor(random() * (index + 1));
      [shuffled[index], shuffled[swap]] = [shuffled[swap], shuffled[index]];
    }
    // Select the best cyclic shift to reduce matching neighbours across columns.
    // The vertical loop has no repeated photo at its seam unless only one exists.
    const previous = result[column - 1];
    const shifts = shuffled.map((_, offset) => ({
      offset,
      matches: previous
        ? shuffled.filter(
            (item, index) =>
              item.image ===
              previous[(index - offset + shuffled.length) % shuffled.length]
                .image,
          ).length
        : 0,
    }));
    const offset = shifts.sort((a, b) => a.matches - b.matches)[0]?.offset ?? 0;
    result.push([...shuffled.slice(offset), ...shuffled.slice(0, offset)]);
  }
  return result;
}
