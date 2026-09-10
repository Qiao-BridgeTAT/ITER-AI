/** Display-only adaptation: keep the provider's original URL in plan evidence. */
export function placePhotoDisplayUrl(src: string): string {
  try {
    const url = new URL(src);
    if (
      url.protocol === "https:" &&
      url.host === "store.is.autonavi.com" &&
      !url.username &&
      !url.password &&
      /^\/showpic\/[a-z\d]+$/i.test(url.pathname) &&
      url.search === "?type=pic"
    ) {
      // This endpoint's default is the same photo's small preview. The full
      // type=pic JPEG can contain an HDR gain map even with an 8-bit base image.
      // Leave other query strings untouched, including signed provider URLs.
      url.search = "";
      return url.href;
    }
  } catch {
    // Local assets and opaque URLs keep their existing loading behavior.
  }
  return src;
}
