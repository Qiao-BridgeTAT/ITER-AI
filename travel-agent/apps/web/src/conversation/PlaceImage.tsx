import {
  type ComponentPropsWithoutRef,
  type ReactNode,
  forwardRef,
  useState,
} from "react";

import "./placeImage.css";
import { placePhotoDisplayUrl } from "./placePhotoSource";

type PlaceImageProps = Omit<
  ComponentPropsWithoutRef<"img">,
  "src" | "srcSet" | "sizes" | "decoding" | "alt"
> & {
  src?: string;
  alt: string;
  fallback?: ReactNode;
};

/** All provider photos share this display policy, without extra image requests. */
export const PlaceImage = forwardRef<HTMLImageElement, PlaceImageProps>(
  function PlaceImage(
    {
      src,
      alt,
      fallback = null,
      className,
      loading = "lazy",
      onError,
      ...props
    },
    ref,
  ) {
    const displaySrc = src ? placePhotoDisplayUrl(src) : undefined;
    const [failedSrc, setFailedSrc] = useState<string>();
    if (!displaySrc || failedSrc === displaySrc) return <>{fallback}</>;

    return (
      <img
        {...props}
        key={displaySrc}
        ref={ref}
        className={["poi-photo", className].filter(Boolean).join(" ")}
        src={displaySrc}
        // Never let an unchecked responsive source load the original HDR photo.
        srcSet={undefined}
        sizes={undefined}
        alt={alt}
        loading={loading}
        decoding="async"
        onError={(event) => {
          setFailedSrc(displaySrc);
          onError?.(event);
        }}
      />
    );
  },
);
