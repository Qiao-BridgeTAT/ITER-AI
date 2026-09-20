/** Presentation only: preserve original messages/hashes and server diagnostics. */
export function planCompletionCopy(text: string): string {
  const withoutNotice = text.replace(
    /^\s*\*\*本版有未满足项\*\*[\s\S]*?(?=正式行程已生成)/u,
    ""
  );
  return (
    withoutNotice
      .split(/(?<=[。！？])|\n/u)
      .filter(
        (part) =>
          !/(?:未安排[：:]|未纳入|舍弃了?|未满足项|空档|碎片空闲|尚未补齐|仍有.*未安排|建议.*补[充入].*景点)/u.test(
            part
          )
      )
      .join("\n")
      .replace(/\n{3,}/gu, "\n\n")
      .trim() || "正式行程已生成，下方可以查看每日安排、交通和预算。"
  );
}
