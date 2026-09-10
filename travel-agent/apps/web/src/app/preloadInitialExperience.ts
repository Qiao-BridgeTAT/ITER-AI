import { waitForVideoFrame } from "./waitForVideoFrame";

const STATIC_BOOT_ASSETS = [
  "/brand/iter-mark-white-64.png",
  "/brand/iter-mark-black-64.png",
  "/icons/menu-figma.svg",
  "/icons/attachment-figma.svg",
  "/icons/send-figma.svg",
] as const;

function waitForImage(image: HTMLImageElement, signal: AbortSignal) {
  if (signal.aborted) return Promise.resolve();
  if (image.complete) {
    return image.decode?.().catch(() => undefined) ?? Promise.resolve();
  }

  return new Promise<void>((resolve) => {
    const finish = () => {
      image.removeEventListener("load", finish);
      image.removeEventListener("error", finish);
      signal.removeEventListener("abort", finish);
      resolve();
    };

    image.addEventListener("load", finish, { once: true });
    image.addEventListener("error", finish, { once: true });
    signal.addEventListener("abort", finish, { once: true });
  });
}

function preloadStaticImage(source: string, signal: AbortSignal) {
  const image = new Image();
  image.decoding = "async";
  image.src = source;
  return waitForImage(image, signal);
}

function nextPaint(signal: AbortSignal) {
  if (signal.aborted) return Promise.resolve();
  return new Promise<void>((resolve) => {
    let frame = 0;
    const finish = () => {
      if (frame) cancelAnimationFrame(frame);
      signal.removeEventListener("abort", finish);
      resolve();
    };
    frame = requestAnimationFrame(finish);
    signal.addEventListener("abort", finish, { once: true });
  });
}

export async function preloadInitialExperience(
  signal: AbortSignal,
  attempt = 0,
) {
  await nextPaint(signal);
  if (signal.aborted) return;

  const fontReady = document.fonts
    ? Promise.allSettled([
        document.fonts.ready,
        document.fonts.load("400 1rem Geist"),
        document.fonts.load("500 1rem Geist"),
        document.fonts.load("700 1rem Geist"),
      ])
    : Promise.resolve();

  const visibleImages = Array.from(document.images)
    .filter((image) => image.loading !== "lazy")
    .map((image) => waitForImage(image, signal));
  const reducedMotion =
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
  const visibleVideos = Array.from(document.querySelectorAll("video")).map(
    (video) =>
      waitForVideoFrame(video, signal, {
        retry: attempt > 0,
        reducedMotion,
      }),
  );
  const staticAssets = STATIC_BOOT_ASSETS.map((source) =>
    preloadStaticImage(source, signal),
  );

  // Warm likely next-page assets without making them block the current page.
  void Promise.allSettled(staticAssets);
  await Promise.all([fontReady, ...visibleImages, ...visibleVideos]);
}
