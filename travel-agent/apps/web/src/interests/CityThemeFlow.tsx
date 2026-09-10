import { useState } from "react";

import { validatePublicContract } from "../contracts/validation";
import type {
  CityThemeContent,
  CityThemeSelection,
} from "../generated/contracts";

interface CityThemeFlowProps {
  cityName: string;
  themes: CityThemeContent[];
  onSubmit: (selection: CityThemeSelection) => boolean;
  initialSelection?: CityThemeSelection;
  disabled?: boolean;
}

export function CityThemeFlow({
  cityName,
  themes,
  onSubmit,
  initialSelection,
  disabled = false,
}: CityThemeFlowProps) {
  const [selectedThemeIds, setSelectedThemeIds] = useState<string[]>(
    initialSelection?.selected_theme_ids ?? [],
  );
  const [openToAny, setOpenToAny] = useState(
    initialSelection?.mode === "open_to_any",
  );
  const [freeText, setFreeText] = useState(initialSelection?.free_text ?? "");
  const [submitError, setSubmitError] = useState<string | null>(null);

  const canSubmit = openToAny || selectedThemeIds.length > 0;
  const submit = () => {
    const note = freeText.trim();
    const selection: CityThemeSelection = {
      mode: openToAny ? "open_to_any" : "selected",
      selected_theme_ids: openToAny ? [] : selectedThemeIds,
      ...(note.length === 0 ? {} : { free_text: note }),
    };
    if (!validatePublicContract("city_theme_selection", selection).success) {
      setSubmitError("至少选一个方向，或者明确告诉我“都可以”。");
      return;
    }
    if (!onSubmit(selection)) {
      setSubmitError("这些兴趣还没有保存成功，请稍后再试。");
    }
  };

  return (
    <section className="city-theme-flow" aria-labelledby="city-theme-title">
      <header className="flow-intro">
        <p className="city-context">接下来决定先看哪些方向</p>
        <h2 id="city-theme-title">来到{cityName}，哪些体验更吸引你？</h2>
        <p>
          可以多选，不限数量。这里决定首轮景点候选，不会把你锁死在某一种玩法里。
        </p>
      </header>

      <div className="theme-bubbles" aria-label={`${cityName}旅行主题`}>
        {themes.map((theme) => {
          const selected = selectedThemeIds.includes(theme.theme_id);
          return (
            <button
              key={theme.theme_id}
              type="button"
              disabled={disabled}
              aria-pressed={selected}
              onClick={() => {
                setOpenToAny(false);
                setSubmitError(null);
                setSelectedThemeIds((current) =>
                  selected
                    ? current.filter((themeId) => themeId !== theme.theme_id)
                    : [...current, theme.theme_id],
                );
              }}
            >
              <strong>{theme.label}</strong>
              <span>{theme.summary}</span>
            </button>
          );
        })}
      </div>

      <button
        className="open-theme-choice"
        type="button"
        disabled={disabled}
        aria-pressed={openToAny}
        onClick={() => {
          setOpenToAny((current) => !current);
          setSelectedThemeIds([]);
          setSubmitError(null);
        }}
      >
        <strong>都可以，先带我看看</strong>
        <span>由系统保留代表性体验，再根据下一步反馈收敛。</span>
      </button>

      <label className="theme-free-text" htmlFor="city-theme-note">
        <span>还有没写在上面的兴趣？（选填）</span>
        <textarea
          id="city-theme-note"
          value={freeText}
          disabled={disabled}
          placeholder="例如：想看城市里的近代工业遗产"
          onChange={(event) => setFreeText(event.currentTarget.value)}
        />
      </label>

      {submitError ? (
        <p className="event-error" role="alert">
          {submitError}
        </p>
      ) : null}
      <div className="flow-completion">
        <p>
          {openToAny
            ? "已选择开放推荐"
            : `已选择 ${selectedThemeIds.length} 个方向`}
        </p>
        <button
          className="primary-action"
          type="button"
          disabled={disabled || !canSubmit}
          onClick={submit}
        >
          继续看景点
        </button>
      </div>
    </section>
  );
}
