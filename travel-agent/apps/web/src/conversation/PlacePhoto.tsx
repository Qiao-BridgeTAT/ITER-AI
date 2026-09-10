import { ImageSquare } from "@phosphor-icons/react";
import { PlaceImage } from "./PlaceImage";

/** Actual POI photos only; missing or failed media keeps the existing placeholder. */
export function PlacePhoto({
  src,
  alt,
  className = "",
}: {
  src?: string;
  alt: string;
  className?: string;
}) {
  return (
    <span className={`place-photo ${className}`}>
      <PlaceImage
        src={src}
        alt={alt}
        fallback={
          <span
            className="place-photo-empty"
            role="img"
            aria-label={`${alt}暂缺`}
          >
            <ImageSquare size={24} weight="light" aria-hidden="true" />
            <span>暂无实景图</span>
          </span>
        }
      />
    </span>
  );
}
