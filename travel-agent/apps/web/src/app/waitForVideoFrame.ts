/** A downloaded frame is not necessarily a displayed, playing background. */
export function waitForVideoFrame(
  video: HTMLVideoElement,
  signal: AbortSignal,
  { retry = false, reducedMotion = false } = {}
) {
  if (signal.aborted) return Promise.resolve();

  return new Promise<void>((resolve, reject) => {
    let settled = false;
    let painted = false;
    let playing = reducedMotion;
    let videoFrame: number | undefined;
    let paintFrame: number | undefined;
    const useVideoFrame =
      !reducedMotion && typeof video.requestVideoFrameCallback === "function";

    const cleanup = () => {
      if (videoFrame !== undefined) video.cancelVideoFrameCallback(videoFrame);
      if (paintFrame !== undefined) cancelAnimationFrame(paintFrame);
      video.removeEventListener("loadeddata", checkData);
      video.removeEventListener("playing", checkData);
      video.removeEventListener("error", fail);
      signal.removeEventListener("abort", cancel);
    };
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (error) reject(error);
      else resolve();
    };
    const completeIfReady = () => {
      if (painted && playing) finish();
    };
    const cancel = () => finish();
    const fail = () => finish(new Error("background_video_unavailable"));
    const checkData = () => {
      if (
        settled ||
        useVideoFrame ||
        paintFrame !== undefined ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        (!reducedMotion && !playing)
      ) {
        return;
      }
      // Older browsers and reduced-motion still frames need a paint opportunity.
      paintFrame = requestAnimationFrame(() => {
        paintFrame = requestAnimationFrame(() => {
          paintFrame = undefined;
          painted = true;
          completeIfReady();
        });
      });
    };

    signal.addEventListener("abort", cancel, { once: true });
    video.addEventListener("error", fail, { once: true });
    video.addEventListener("loadeddata", checkData);
    video.addEventListener("playing", checkData);

    if (retry) video.load();
    if (video.error) {
      fail();
      return;
    }
    if (useVideoFrame) {
      videoFrame = video.requestVideoFrameCallback(() => {
        videoFrame = undefined;
        painted = true;
        completeIfReady();
      });
    }

    if (reducedMotion) {
      video.pause();
    } else {
      // Set the media properties before play(), including Safari's muted default.
      video.muted = true;
      video.defaultMuted = true;
      video.playsInline = true;
      void video.play().then(() => {
        if (settled) return;
        playing = true;
        checkData();
        completeIfReady();
      }, fail);
    }
    checkData();
  });
}
