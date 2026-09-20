import type { FiveLevel, PriorityGoal } from "../generated/enums";

export const TRANSIT_TAXI_OPTIONS: Readonly<Record<FiveLevel, string>> = {
  1: "公共交通优先，除非没有合适路线",
  2: "公共交通为主，换乘多、走路多或明显绕路时再打车",
  3: "逐段综合比较时间、换乘、步行和价格",
  4: "打车为主，公共交通特别直达时也可以坐",
  5: "打车优先，通常更希望门到门"
};

export const PRIORITY_GOAL_OPTIONS: ReadonlyArray<{
  value: PriorityGoal;
  label: string;
  help: string;
}> = [
  {
    value: "must_see_places",
    label: "想去的地方不留遗憾",
    help: "优先保住真正想去的地点"
  },
  {
    value: "comfortable_stay",
    label: "住得舒服",
    help: "住宿品质和休息体验更重要"
  },
  {
    value: "satisfying_food",
    label: "吃得满意",
    help: "愿意为合口味的餐厅留出空间"
  },
  {
    value: "smooth_routes",
    label: "路线顺、少折腾",
    help: "减少换乘、折返和无效奔波"
  },
  {
    value: "good_value",
    label: "整体花费划算",
    help: "在体验与花费之间更看重性价比"
  }
];
