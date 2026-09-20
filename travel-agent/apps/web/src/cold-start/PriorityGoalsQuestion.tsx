import type { PriorityGoal } from "../generated/enums";
import { PRIORITY_GOAL_OPTIONS } from "./transitOptions";

interface PriorityGoalsQuestionProps {
  value: PriorityGoal[];
  onChange: (value: PriorityGoal[]) => void;
}

const MAX_PRIORITY_GOALS = 2;

export function PriorityGoalsQuestion({
  value,
  onChange
}: PriorityGoalsQuestionProps) {
  const atLimit = value.length >= MAX_PRIORITY_GOALS;

  return (
    <fieldset className="cold-start-question priority-goals-question">
      <legend>一趟旅行里，你最希望优先保证哪两件事？</legend>
      <p className="question-help">
        选择一项或两项。它们会在景点、餐饮、住宿和交通发生冲突时帮助 Agent
        做取舍。
      </p>
      <div className="priority-goal-options">
        {PRIORITY_GOAL_OPTIONS.map((option) => {
          const selected = value.includes(option.value);
          return (
            <button
              key={option.value}
              type="button"
              aria-pressed={selected}
              disabled={!selected && atLimit}
              onClick={() => {
                if (selected) {
                  onChange(value.filter((item) => item !== option.value));
                } else if (!atLimit) {
                  onChange([...value, option.value]);
                }
              }}
            >
              <strong>{option.label}</strong>
              <span>{option.help}</span>
            </button>
          );
        })}
      </div>
      <p className="selection-count" role="status">
        已选 {value.length} 项，最多 2 项
      </p>
    </fieldset>
  );
}
