// Compatibility for descriptions saved by the old card composer. This only
// changes presentation; never rewrite selections, source facts or safety copy.
export function discoveryOptionDescription(
  section: string | undefined,
  description: string | null | undefined
): string | null {
  if (!description) return null;
  if (
    /^走进.+感受这里的风景与人文|^以.+风味为主，可以体验这一菜系/u.test(
      description
    )
  )
    return null;
  if (section === "dining_specific") {
    if (
      /^(?:中餐厅|中式餐饮|餐饮服务|餐饮相关场所)[。.]?$/u.test(
        description.trim()
      )
    )
      return null;
    if (description === "来自当前城市真实地点候选。") return null;
    const parts = description.split(" · ");
    const rankingIndex = parts.findIndex((part) =>
      /^(用户尚未对这个地点[作做]出选择|用户明确标记为必去|用户明确表达想去|用户主动点名了这个地点)[；;]/u.test(
        part
      )
    );
    if (rankingIndex === 0 || rankingIndex === 1) {
      // Old shape: optional address · two ranking explanations · optional badge.
      return (
        parts
          .slice(rankingIndex + 1)
          .filter((part) => part !== "城市代表性补位")
          .join(" · ") || null
      );
    }
  }
  if (section === "lodging_area_preference") {
    const marker = description.indexOf(" 氛围：");
    if (marker !== -1) {
      const publicCopy = description.slice(0, marker).trim();
      const tradeoff = description
        .match(/；取舍：([\s\S]+)$/u)?.[1]
        .replace(/[。；\s]+$/u, "")
        .trim();
      return tradeoff && !publicCopy.includes(tradeoff)
        ? `${publicCopy} ${tradeoff}。`
        : publicCopy || null;
    }
  }
  if (section === "lodging_class_preference") {
    return description
      .replace("当前城市与日期的供应样本参考约为", "参考")
      .replace("当前日期暂无可信价格样本，金额不作推测。", "参考价暂缺。");
  }
  return description;
}
