import { EchoText } from "./EchoText";

/** Shared visual only; readiness and cancellation belong to the caller. */
export function BootLoadingVisual({
  text = "稍等一下，在为你准备中。"
}: {
  text?: string;
}) {
  return (
    <div className="app-boot-content">
      <div className="app-boot-tower" aria-hidden="true">
        {[1, 2, 3, 4].map((box) => (
          <span className={`app-boot-box app-boot-box-${box}`} key={box}>
            <span className="app-boot-side app-boot-side-left" />
            <span className="app-boot-side app-boot-side-right" />
            <span className="app-boot-side app-boot-side-top" />
          </span>
        ))}
      </div>
      <EchoText
        className="app-boot-message"
        text={text}
        echoes={7}
        offset={22}
        direction="right"
        fade={0.74}
        blur={2.4}
        tint="#7ab0ea"
        duration={980}
        color="#10243f"
        fontWeight={600}
      />
    </div>
  );
}
