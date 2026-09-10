interface MobileDeviceHints {
  userAgent?: string;
  userAgentData?: { mobile?: boolean };
}

export function isMobilePhone({
  userAgent = "",
  userAgentData,
}: MobileDeviceHints): boolean {
  if (userAgentData?.mobile === true) return true;

  // Safari and many in-app browsers have no UA Client Hints. Width and touch
  // support alone cannot distinguish a phone from a resized desktop window.
  return (
    /iPhone|iPod|Windows Phone/iu.test(userAgent) ||
    (/Android/iu.test(userAgent) && /Mobile/iu.test(userAgent))
  );
}
