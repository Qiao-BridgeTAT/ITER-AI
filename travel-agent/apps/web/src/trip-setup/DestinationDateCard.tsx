import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent
} from "react";
import { createPortal } from "react-dom";
import {
  CaretLeft,
  CaretRight,
  MagnifyingGlass,
  X,
  XCircle
} from "@phosphor-icons/react";
import { useModalFocus } from "../accessibility/useModalFocus";
import {
  addCalendarDays,
  calendarDayOfWeek,
  destinationToday,
  formatCalendarDate,
  parseCalendarDate
} from "./dateMath";
import {
  selectRangeDate,
  tripLength,
  tripSetupFields,
  type TripSetupFields,
  type DateRangeSelection
} from "./dateRangeSelection";
import { searchDestinationCities, type DestinationCity } from "./citySearch";
import "./destination-date-card.css";

interface Props {
  today?: string;
  connected: boolean;
  onSubmit: (fields: TripSetupFields) => boolean;
}

export function DestinationDateCard(props: Props) {
  const [open, setOpen] = useState(true);
  const [city, setCity] = useState("");
  const [selectedCity, setSelectedCity] = useState<DestinationCity | null>(
    null
  );
  const [range, setRange] = useState<DateRangeSelection>({
    start: null,
    end: null
  });
  const triggerRef = useRef<HTMLButtonElement>(null);
  const close = useCallback(() => {
    setOpen(false);
    requestAnimationFrame(() => triggerRef.current?.focus());
  }, []);
  return (
    <>
      <div className="destination-date-entry">
        <button type="button" ref={triggerRef} onClick={() => setOpen(true)}>
          设置目的地和日期
        </button>
      </div>
      {open && (
        <DestinationDateDialog
          {...props}
          city={city}
          selectedCity={selectedCity}
          onSelectedCityChange={setSelectedCity}
          range={range}
          onCityChange={setCity}
          onRangeChange={setRange}
          onClose={close}
        />
      )}
    </>
  );
}

function DestinationDateDialog({
  today: fixedToday,
  connected,
  city,
  selectedCity,
  onSelectedCityChange,
  range,
  onCityChange,
  onRangeChange,
  onClose,
  onSubmit
}: Props & {
  city: string;
  selectedCity: DestinationCity | null;
  onSelectedCityChange: (city: DestinationCity | null) => void;
  range: DateRangeSelection;
  onCityChange: (city: string) => void;
  onRangeChange: (range: DateRangeSelection) => void;
  onClose: () => void;
}) {
  const [today, setToday] = useState(() => fixedToday ?? destinationToday());
  useEffect(() => {
    if (fixedToday) {
      setToday(fixedToday);
      return;
    }
    const timer = window.setInterval(
      () => setToday(destinationToday()),
      30_000
    );
    return () => window.clearInterval(timer);
  }, [fixedToday]);
  const [month, setMonth] = useState(
    () => (range.start ?? today).slice(0, 7) + "-01"
  );
  const [focusDate, setFocusDate] = useState(range.end ?? range.start ?? today);
  const [hoverDate, setHoverDate] = useState<string | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [activeCityIndex, setActiveCityIndex] = useState(0);
  const matches = searchDestinationCities(city);
  const showMatches = searchOpen && Boolean(city.trim());
  const selectCity = (value: DestinationCity) => {
    onSelectedCityChange(value);
    onCityChange(value.name);
    setSearchOpen(false);
    setSubmitError(null);
  };
  const [submitError, setSubmitError] = useState<string | null>(null);
  const submitted = useRef(false);
  const dialogRef = useModalFocus<HTMLElement>(onClose);
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    inputRef.current?.focus();
  }, []);
  const parsedMonth = parseCalendarDate(month)!;
  const nextMonth = formatCalendarDate({
    year: parsedMonth.year + (parsedMonth.month === 12 ? 1 : 0),
    month: parsedMonth.month === 12 ? 1 : parsedMonth.month + 1,
    day: 1
  });
  const previousMonth = formatCalendarDate({
    year: parsedMonth.year - (parsedMonth.month === 1 ? 1 : 0),
    month: parsedMonth.month === 1 ? 12 : parsedMonth.month - 1,
    day: 1
  });
  const offset = (calendarDayOfWeek(month) + 6) % 7;
  const lastDay = parseCalendarDate(addCalendarDays(nextMonth, -1))!.day;
  const length = tripLength(range);
  const past = Boolean(range.start && range.start < today);
  const tooLong = Boolean(length && length.days > 5);
  const valid = Boolean(selectedCity && length && !tooLong && !past);
  const previewEnd =
    !range.end && range.start && hoverDate && hoverDate >= range.start
      ? hoverDate
      : null;
  const error = past
    ? "起始日期已过，请重新选择。"
    : tooLong
      ? "目前支持 1–5 天旅行，请重新选择日期。"
      : submitError;
  const shortDate = (date: string) => {
    const value = parseCalendarDate(date)!;
    return `${value.month}月${value.day}日`;
  };
  const choose = (date: string) => {
    onRangeChange(selectRangeDate(range, date));
    setFocusDate(date);
    setHoverDate(null);
    setSubmitError(null);
  };
  const moveFocus = (date: string) => {
    const target = date < today ? today : date;
    setMonth(target.slice(0, 7) + "-01");
    setFocusDate(target);
    requestAnimationFrame(() =>
      dialogRef.current
        ?.querySelector<HTMLButtonElement>(`[data-date="${target}"]`)
        ?.focus()
    );
  };
  const dateKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    date: string
  ) => {
    const delta = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -7, ArrowDown: 7 }[
      event.key
    ];
    if (delta !== undefined) {
      event.preventDefault();
      moveFocus(addCalendarDays(date, delta));
    }
    if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      const weekDay = (calendarDayOfWeek(date) + 6) % 7;
      moveFocus(
        addCalendarDays(date, event.key === "Home" ? -weekDay : 6 - weekDay)
      );
    }
  };
  return createPortal(
    <div className="destination-date-backdrop">
      <section
        className="destination-date-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="destination-date-title"
        aria-describedby="destination-date-subtitle"
        ref={dialogRef}
        tabIndex={-1}
      >
        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (!valid || submitted.current) return;
            if (!connected) {
              setSubmitError("连接尚未就绪，你的选择已保留，请稍后重试。");
              return;
            }
            submitted.current = true;
            if (!onSubmit(tripSetupFields(selectedCity!.cityId, range))) {
              submitted.current = false;
              setSubmitError("暂时未能发送，你的选择已保留，请重试。");
            }
          }}
        >
          <header className="destination-date-heading">
            <h2 id="destination-date-title">先把目的地和日期设计好吧</h2>
            <p id="destination-date-subtitle">
              选好这两项，就开始选择景点偏好。
            </p>
            <button
              type="button"
              className="destination-date-close"
              aria-label="关闭目的地和日期"
              onClick={onClose}
            >
              <X size={24} />
            </button>
          </header>
          <div className="destination-date-body">
            <div className="destination-date-details">
              <label htmlFor="trip-destination">目的地</label>
              <div className="destination-date-city-control">
                <div className="destination-date-input">
                  <MagnifyingGlass size={23} aria-hidden="true" />
                  <input
                    ref={inputRef}
                    id="trip-destination"
                    value={city}
                    maxLength={80}
                    autoComplete="off"
                    role="combobox"
                    aria-autocomplete="list"
                    aria-expanded={showMatches}
                    aria-controls={
                      showMatches ? "destination-city-results" : undefined
                    }
                    aria-activedescendant={
                      showMatches && matches[activeCityIndex]
                        ? `destination-city-${activeCityIndex}`
                        : undefined
                    }
                    onFocus={() => {
                      if (!selectedCity) setSearchOpen(true);
                    }}
                    onBlur={() => setSearchOpen(false)}
                    onKeyDown={(event) => {
                      if (event.nativeEvent.isComposing) return;
                      if (event.key === "Escape" && showMatches) {
                        event.preventDefault();
                        event.stopPropagation();
                        setSearchOpen(false);
                      } else if (
                        event.key === "ArrowDown" ||
                        event.key === "ArrowUp"
                      ) {
                        event.preventDefault();
                        setSearchOpen(true);
                        setActiveCityIndex((index) =>
                          matches.length
                            ? (index +
                                (event.key === "ArrowDown" ? 1 : -1) +
                                matches.length) %
                              matches.length
                            : 0
                        );
                      } else if (event.key === "Enter" && showMatches) {
                        event.preventDefault();
                        if (matches[activeCityIndex])
                          selectCity(matches[activeCityIndex]);
                      }
                    }}
                    onChange={(event) => {
                      onCityChange(event.target.value);
                      onSelectedCityChange(null);
                      setSearchOpen(true);
                      setActiveCityIndex(0);
                      setSubmitError(null);
                    }}
                  />
                  {city && (
                    <button
                      type="button"
                      aria-label="清空目的地"
                      onClick={() => {
                        onCityChange("");
                        onSelectedCityChange(null);
                        setSearchOpen(false);
                        inputRef.current?.focus();
                      }}
                    >
                      <XCircle size={21} weight="fill" />
                    </button>
                  )}
                </div>
                {showMatches && (
                  <div className="destination-city-results">
                    <ul
                      id="destination-city-results"
                      role="listbox"
                      aria-label="相近城市"
                    >
                      {matches.map((match, index) => (
                        <li
                          key={match.cityId}
                          id={`destination-city-${index}`}
                          role="option"
                          aria-selected={index === activeCityIndex}
                          onMouseDown={(event) => event.preventDefault()}
                          onMouseEnter={() => setActiveCityIndex(index)}
                          onClick={() => selectCity(match)}
                        >
                          {match.label}
                        </li>
                      ))}
                    </ul>
                    {matches.length === 0 && (
                      <p role="status">未找到相近城市，请换个名称</p>
                    )}
                  </div>
                )}
              </div>
              <div
                className="destination-date-summary"
                aria-live="polite"
                aria-atomic="true"
              >
                <p className="destination-date-caption">行程天数</p>
                {range.start ? (
                  <>
                    <p className="destination-date-range">
                      {range.end &&
                      range.start.slice(0, 4) !== range.end.slice(0, 4)
                        ? `${range.start.replaceAll("-", "/")} — ${range.end.replaceAll("-", "/")}`
                        : `${shortDate(range.start)}${range.end ? ` — ${shortDate(range.end)}` : "出发"}`}
                    </p>
                    <p className="destination-date-count">
                      {length
                        ? `${length.days}天${length.nights}晚`
                        : "请选择结束日期"}
                    </p>
                  </>
                ) : (
                  <p className="destination-date-empty">请选择起始日期</p>
                )}
              </div>
              {error && (
                <p className="destination-date-error" role="alert">
                  {error}
                </p>
              )}
            </div>
            <div
              className="destination-date-calendar"
              onMouseLeave={() => setHoverDate(null)}
            >
              <div className="destination-date-month">
                <button
                  type="button"
                  aria-label="上个月"
                  disabled={previousMonth.slice(0, 7) < today.slice(0, 7)}
                  onClick={() => {
                    setMonth(previousMonth);
                    setFocusDate(previousMonth < today ? today : previousMonth);
                    setHoverDate(null);
                  }}
                >
                  <CaretLeft size={21} weight="bold" />
                </button>
                <h3 aria-live="polite">
                  {parsedMonth.year}年{parsedMonth.month}月
                </h3>
                <button
                  type="button"
                  aria-label="下个月"
                  onClick={() => {
                    setMonth(nextMonth);
                    setFocusDate(nextMonth);
                    setHoverDate(null);
                  }}
                >
                  <CaretRight size={21} weight="bold" />
                </button>
              </div>
              <div className="destination-date-weekdays" aria-hidden="true">
                {["一", "二", "三", "四", "五", "六", "日"].map((day) => (
                  <span key={day}>{day}</span>
                ))}
              </div>
              <div
                className="destination-date-days"
                role="group"
                aria-label={
                  range.start && !range.end ? "选择结束日期" : "选择起始日期"
                }
              >
                {Array.from({ length: offset }, (_, index) => (
                  <span key={`blank-${index}`} />
                ))}
                {Array.from({ length: lastDay }, (_, index) => {
                  const date = formatCalendarDate({
                    ...parsedMonth,
                    day: index + 1
                  });
                  const endpoint = date === range.start || date === range.end;
                  const inRange = Boolean(
                    range.start &&
                    range.end &&
                    date > range.start &&
                    date < range.end
                  );
                  const preview = Boolean(
                    range.start &&
                    previewEnd &&
                    date > range.start &&
                    date <= previewEnd
                  );
                  return (
                    <button
                      type="button"
                      key={date}
                      data-date={date}
                      className={[
                        "destination-date-day",
                        endpoint && "is-endpoint",
                        inRange && "is-in-range",
                        preview && "is-preview",
                        date === today && "is-today"
                      ]
                        .filter(Boolean)
                        .join(" ")}
                      aria-label={`${parsedMonth.year}年${parsedMonth.month}月${index + 1}日${date === range.start ? "，起始日" : ""}${date === range.end ? "，结束日" : ""}`}
                      aria-current={date === today ? "date" : undefined}
                      aria-pressed={endpoint || inRange}
                      disabled={date < today}
                      tabIndex={date === focusDate ? 0 : -1}
                      onFocus={() => setFocusDate(date)}
                      onKeyDown={(event) => dateKeyDown(event, date)}
                      onMouseEnter={() => setHoverDate(date)}
                      onClick={() => choose(date)}
                    >
                      <span>{index + 1}</span>
                      {date === today && <i aria-hidden="true" />}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
          <footer className="destination-date-footer">
            {!connected && <span role="status">正在连接，选好后即可继续</span>}
            <button
              type="submit"
              className="destination-date-submit"
              disabled={!valid || !connected}
            >
              选择景点偏好
            </button>
          </footer>
        </form>
      </section>
    </div>,
    document.body
  );
}
