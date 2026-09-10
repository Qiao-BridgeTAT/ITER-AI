import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { ContinuousPreferenceSlider } from "../components/ContinuousPreferenceSlider";
import type { ColdStartSubmission } from "../generated/contracts";
import type {
  DayReturn,
  DayStart,
  FiveLevel,
  MobilityTolerance,
  PriorityGoal,
} from "../generated/enums";
import { PriorityGoalsQuestion } from "./PriorityGoalsQuestion";

interface ColdStartModalProps {
  onClose: () => void;
  onComplete: (submission: ColdStartSubmission) => void;
  initialSubmission?: ColdStartSubmission;
  mode?: "onboarding" | "editing";
  submitting?: boolean;
  submissionError?: string | null;
}

interface ColdStartValues {
  dayStart: DayStart;
  dayReturn: DayReturn;
  pace: number;
  classicNiche: number;
  walking: number;
  bikeTolerance: MobilityTolerance | null;
  transitTaxi: number;
  priorityGoals: PriorityGoal[];
}

const STEP_LABELS = [
  "一天的节奏",
  "旅行的步调",
  "熟悉与新鲜",
  "行走方式",
  "出行习惯",
  "最在意的事",
] as const;

const DAY_START_OPTIONS: ReadonlyArray<{ value: DayStart; label: string }> = [
  { value: "before_07", label: "7 点前" },
  { value: "around_08", label: "8 点左右" },
  { value: "around_09", label: "9 点左右" },
  { value: "around_10", label: "10 点左右" },
  { value: "after_11", label: "11 点后" },
];

const DAY_RETURN_OPTIONS: ReadonlyArray<{ value: DayReturn; label: string }> = [
  { value: "before_20", label: "20 点前" },
  { value: "around_21", label: "21 点左右" },
  { value: "after_22", label: "22 点后" },
];

const INITIAL_VALUES: ColdStartValues = {
  dayStart: "around_09",
  dayReturn: "around_21",
  pace: 38,
  classicNiche: 50,
  walking: 52,
  bikeTolerance: null,
  transitTaxi: 46,
  priorityGoals: [],
};

export function ColdStartModal({
  onClose,
  onComplete,
  initialSubmission,
  mode = "onboarding",
  submitting = false,
  submissionError = null,
}: ColdStartModalProps) {
  const isEditing = mode === "editing";
  const [step, setStep] = useState(0);
  const [values, setValues] = useState<ColdStartValues>(() =>
    initialSubmission
      ? valuesFromSubmission(initialSubmission)
      : INITIAL_VALUES,
  );
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(
    () =>
      new Set(initialSubmission ? STEP_LABELS.map((_, index) => index) : []),
  );
  const titleRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    titleRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  useEffect(() => {
    titleRef.current?.focus();
  }, [step]);

  const profileIsComplete =
    values.bikeTolerance !== null && values.priorityGoals.length > 0;
  const canContinue = isEditing
    ? profileIsComplete
    : (step !== 3 || values.bikeTolerance !== null) &&
      (step !== 5 || values.priorityGoals.length > 0);

  const summaries = useMemo(
    () => [
      `${shortTime(values.dayStart)} - ${shortTime(values.dayReturn)}`,
      paceDescription(values.pace),
      classicNicheDescription(values.classicNiche),
      walkingDescription(values.walking),
      transitDescription(values.transitTaxi),
      values.priorityGoals.length
        ? `${values.priorityGoals.length} 项优先`
        : "",
    ],
    [values],
  );

  const submitCurrentStep = (event: FormEvent) => {
    event.preventDefault();
    if (!canContinue || submitting) return;

    setCompletedSteps((current) => new Set(current).add(step));
    if (!isEditing && step < STEP_LABELS.length - 1) {
      setStep((current) => current + 1);
      return;
    }

    onComplete({
      day_start: values.dayStart,
      day_return: values.dayReturn,
      pace_level: continuousToFiveLevel(values.pace),
      classic_niche_level: continuousToFiveLevel(values.classicNiche),
      walking_tolerance: walkingToTolerance(values.walking),
      bike_tolerance: values.bikeTolerance ?? "never",
      transit_taxi_level: continuousToFiveLevel(values.transitTaxi),
      priority_goals: values.priorityGoals,
    });
  };

  return (
    <div className="cold-start-overlay" role="presentation">
      <section
        className="cold-start-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="cold-start-modal-title"
      >
        <header className="cold-start-modal-header">
          <div className="cold-start-modal-brand">
            <img src="/brand/iter-mark-black-64.png" alt="" />
            <span aria-hidden="true" />
            <strong>
              {isEditing ? "重新选择长期偏好" : "先让我们了解你一下"}
            </strong>
          </div>
          <div className="cold-start-modal-status" aria-live="polite">
            <span>{step + 1} / 6</span>
            <button
              type="button"
              aria-label={isEditing ? "关闭长期偏好" : "关闭冷启动"}
              onClick={onClose}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M6 6l12 12M18 6 6 18" />
              </svg>
            </button>
          </div>
        </header>

        <div className="cold-start-modal-body">
          <nav className="cold-start-index" aria-label="旅行习惯问题">
            {STEP_LABELS.map((label, index) => {
              const complete = completedSteps.has(index);
              return (
                <button
                  key={label}
                  type="button"
                  className={index === step ? "is-current" : ""}
                  disabled={!isEditing && index > step && !complete}
                  aria-current={index === step ? "step" : undefined}
                  onClick={() => {
                    if (isEditing || index <= step || complete) setStep(index);
                  }}
                >
                  <span className="cold-start-index-marker" aria-hidden="true">
                    {complete ? "✓" : ""}
                  </span>
                  <span>
                    <strong>{label}</strong>
                    {summaries[index] ? (
                      <small>{summaries[index]}</small>
                    ) : null}
                  </span>
                </button>
              );
            })}
          </nav>

          <form className="cold-start-stage" onSubmit={submitCurrentStep}>
            <div className="cold-start-stage-content">
              <StepContent
                step={step}
                values={values}
                setValues={setValues}
                titleRef={titleRef}
              />
            </div>
            <footer className="cold-start-modal-footer">
              {submissionError ? (
                <p className="cold-start-submit-error" role="alert">
                  {submissionError}
                </p>
              ) : null}
              <button
                className="cold-start-back"
                type="button"
                onClick={() => {
                  if (isEditing || step === 0) onClose();
                  else setStep((current) => current - 1);
                }}
              >
                {isEditing ? "关闭" : step === 0 ? "返回首页" : "返回"}
              </button>
              <button
                className="cold-start-continue"
                type="submit"
                disabled={!canContinue || submitting}
              >
                {submitting
                  ? "保存中…"
                  : isEditing
                    ? "保存偏好"
                    : step === STEP_LABELS.length - 1
                      ? "进入对话"
                      : "继续"}
              </button>
            </footer>
          </form>
        </div>
      </section>
    </div>
  );
}

function StepContent({
  step,
  values,
  setValues,
  titleRef,
}: {
  step: number;
  values: ColdStartValues;
  setValues: React.Dispatch<React.SetStateAction<ColdStartValues>>;
  titleRef: React.RefObject<HTMLHeadingElement>;
}) {
  if (step === 0) {
    return (
      <>
        <QuestionHeading
          eyebrow="一天的节奏"
          title="旅行时，你舒服的一天通常几点出门、几点回住处？"
          help="这会影响每天真正可用的时间，也可以在具体旅行里再调整。"
          titleRef={titleRef}
        />
        <div className="cold-start-time-grid">
          <TimeChoice
            label="出门时间"
            value={values.dayStart}
            options={DAY_START_OPTIONS}
            onChange={(dayStart) =>
              setValues((current) => ({ ...current, dayStart }))
            }
          />
          <TimeChoice
            label="回住处时间"
            value={values.dayReturn}
            options={DAY_RETURN_OPTIONS}
            onChange={(dayReturn) =>
              setValues((current) => ({ ...current, dayReturn }))
            }
          />
        </div>
      </>
    );
  }

  if (step === 1) {
    return (
      <>
        <QuestionHeading
          eyebrow="旅行的步调"
          title="你希望一天安排得松一点，还是尽可能多看一些？"
          help="选择更接近你的方式，也可以稍后修改。"
          titleRef={titleRef}
        />
        <div className="cold-start-slider-shell">
          <ContinuousPreferenceSlider
            label="旅行的步调"
            startLabel="想慢慢走"
            endLabel="想多看看"
            startDescription="每天留出休息和随兴时间"
            endDescription="一天安排更多地点"
            feedbackLabel="当前更接近"
            value={values.pace}
            valueText={paceFeedback(values.pace)}
            showHeading={false}
            onChange={(pace) => setValues((current) => ({ ...current, pace }))}
          />
        </div>
      </>
    );
  }

  if (step === 2) {
    return (
      <>
        <QuestionHeading
          eyebrow="熟悉与新鲜"
          title="第一次到一座城市，你更看重经典，还是自己的兴趣？"
          help="这里没有标准答案，我们只想知道怎样的地方更像你。"
          titleRef={titleRef}
        />
        <div className="cold-start-slider-shell">
          <ContinuousPreferenceSlider
            label="经典与兴趣"
            startLabel="先看经典"
            endLabel="跟着兴趣"
            startDescription="优先城市代表性的地方"
            endDescription="更看重个人喜好与小众体验"
            feedbackLabel="当前更接近"
            value={values.classicNiche}
            valueText={classicNicheFeedback(values.classicNiche)}
            showHeading={false}
            onChange={(classicNiche) =>
              setValues((current) => ({ ...current, classicNiche }))
            }
          />
        </div>
      </>
    );
  }

  if (step === 3) {
    return (
      <>
        <QuestionHeading
          eyebrow="行走方式"
          title="在景点之间移动时，你对步行和骑行的接受度如何？"
          help="步行用无级滑杆表达；共享单车只需告诉我们会不会考虑。"
          titleRef={titleRef}
        />
        <div className="cold-start-slider-shell">
          <ContinuousPreferenceSlider
            label="步行接受度"
            startLabel="少走一些"
            endLabel="多走也可以"
            startDescription="尽量减少连续步行和长距离接驳"
            endDescription="为了体验愿意多走一段"
            feedbackLabel="当前更接近"
            value={values.walking}
            valueText={walkingFeedback(values.walking)}
            showHeading={false}
            onChange={(walking) =>
              setValues((current) => ({ ...current, walking }))
            }
          />
        </div>
        <div className="cold-start-bike-choice">
          <span>共享单车</span>
          <div role="group" aria-label="是否考虑共享单车">
            <button
              type="button"
              aria-pressed={
                values.bikeTolerance !== null &&
                values.bikeTolerance !== "never"
              }
              onClick={() =>
                setValues((current) => ({
                  ...current,
                  bikeTolerance: "around_10",
                }))
              }
            >
              会考虑
            </button>
            <button
              type="button"
              aria-pressed={values.bikeTolerance === "never"}
              onClick={() =>
                setValues((current) => ({
                  ...current,
                  bikeTolerance: "never",
                }))
              }
            >
              一般不考虑
            </button>
          </div>
        </div>
      </>
    );
  }

  if (step === 4) {
    return (
      <>
        <QuestionHeading
          eyebrow="出行习惯"
          title="需要乘车时，你更偏向公共交通，还是门到门的打车？"
          help="具体路线仍会综合比较时间、换乘、步行和价格。"
          titleRef={titleRef}
        />
        <div className="cold-start-slider-shell">
          <ContinuousPreferenceSlider
            label="公共交通与打车"
            startLabel="更常坐公交地铁"
            endLabel="更常打车"
            startDescription="接受换乘，更看重成本和稳定"
            endDescription="少换乘，更看重门到门和省时"
            feedbackLabel="当前更接近"
            value={values.transitTaxi}
            valueText={transitFeedback(values.transitTaxi)}
            showHeading={false}
            onChange={(transitTaxi) =>
              setValues((current) => ({ ...current, transitTaxi }))
            }
          />
        </div>
      </>
    );
  }

  return (
    <div className="cold-start-priorities">
      <QuestionHeading
        eyebrow="最在意的事"
        title="一趟旅行里，你最希望优先保证哪两件事？"
        help="至少选择一项，最多两项。发生取舍时，我们会优先保住它们。"
        titleRef={titleRef}
      />
      <PriorityGoalsQuestion
        value={values.priorityGoals}
        onChange={(priorityGoals) =>
          setValues((current) => ({ ...current, priorityGoals }))
        }
      />
    </div>
  );
}

function QuestionHeading({
  eyebrow,
  title,
  help,
  titleRef,
}: {
  eyebrow: string;
  title: string;
  help: string;
  titleRef: React.RefObject<HTMLHeadingElement>;
}) {
  return (
    <header className="cold-start-question-heading">
      <p>{eyebrow}</p>
      <h2 id="cold-start-modal-title" ref={titleRef} tabIndex={-1}>
        {title}
      </h2>
      <span>{help}</span>
    </header>
  );
}

function TimeChoice<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: ReadonlyArray<{ value: T; label: string }>;
  onChange: (value: T) => void;
}) {
  return (
    <fieldset className="cold-start-time-choice">
      <legend>{label}</legend>
      <div>
        {options.map((option) => (
          <label key={option.value}>
            <input
              type="radio"
              name={label}
              value={option.value}
              checked={value === option.value}
              onChange={() => onChange(option.value)}
            />
            <span>{option.label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function continuousToFiveLevel(value: number): FiveLevel {
  return Math.min(5, Math.max(1, Math.floor(value / 20) + 1)) as FiveLevel;
}

function fiveLevelToContinuous(value: FiveLevel): number {
  return value * 20 - 10;
}

function toleranceToWalking(value: MobilityTolerance): number {
  const positions: Record<MobilityTolerance, number> = {
    never: 8,
    within_5: 30,
    around_10: 56,
    "15_plus": 86,
  };
  return positions[value];
}

function valuesFromSubmission(
  submission: ColdStartSubmission,
): ColdStartValues {
  return {
    dayStart: submission.day_start,
    dayReturn: submission.day_return,
    pace: fiveLevelToContinuous(submission.pace_level),
    classicNiche: fiveLevelToContinuous(submission.classic_niche_level),
    walking: toleranceToWalking(submission.walking_tolerance),
    bikeTolerance: submission.bike_tolerance,
    transitTaxi: fiveLevelToContinuous(submission.transit_taxi_level),
    priorityGoals: [...submission.priority_goals],
  };
}

function walkingToTolerance(value: number): MobilityTolerance {
  if (value <= 15) return "never";
  if (value <= 40) return "within_5";
  if (value <= 72) return "around_10";
  return "15_plus";
}

function paceDescription(value: number): string {
  if (value <= 20) return "很松弛";
  if (value <= 40) return "轻松一点";
  if (value <= 60) return "松紧平衡";
  if (value <= 80) return "充实一些";
  return "尽量多看";
}

function paceFeedback(value: number): string {
  if (value <= 20) return "慢慢走";
  if (value <= 40) return "轻松一点";
  if (value <= 60) return "松紧适中";
  if (value <= 80) return "安排得充实些";
  return "尽可能多看看";
}

function classicNicheDescription(value: number): string {
  if (value <= 20) return "经典优先";
  if (value <= 40) return "经典多一些";
  if (value <= 60) return "两边平衡";
  if (value <= 80) return "兴趣多一些";
  return "跟着兴趣走";
}

function classicNicheFeedback(value: number): string {
  if (value <= 20) return "明显偏经典";
  if (value <= 40) return "经典多一些";
  if (value <= 60) return "经典和兴趣都要";
  if (value <= 80) return "兴趣多一些";
  return "主要跟着兴趣";
}

function walkingDescription(value: number): string {
  if (value <= 15) return "尽量不步行接驳";
  if (value <= 40) return "少走一点";
  if (value <= 72) return "十分钟左右可以";
  return "多走一会也可以";
}

function walkingFeedback(value: number): string {
  if (value <= 15) return "希望尽量少走";
  if (value <= 40) return "走一小段可以";
  if (value <= 72) return "十分钟左右没问题";
  return "多走一会儿也可以";
}

function transitDescription(value: number): string {
  if (value <= 20) return "公共交通优先";
  if (value <= 40) return "公交地铁多一些";
  if (value <= 60) return "逐段比较";
  if (value <= 80) return "打车多一些";
  return "打车优先";
}

function transitFeedback(value: number): string {
  if (value <= 20) return "更偏公交地铁";
  if (value <= 40) return "公交地铁多一些";
  if (value <= 60) return "看当段路线决定";
  if (value <= 80) return "打车多一些";
  return "更偏门到门打车";
}

function shortTime(value: DayStart | DayReturn): string {
  const labels: Record<DayStart | DayReturn, string> = {
    before_07: "07:00 前",
    around_08: "08:00",
    around_09: "09:00",
    around_10: "10:00",
    after_11: "11:00 后",
    before_20: "20:00 前",
    around_21: "21:00",
    after_22: "22:00 后",
    flexible: "灵活",
  };
  return labels[value];
}
