import { Link } from "react-router-dom";

import { useTripShell } from "../session/TripShellContext";

export function NotFoundPage() {
  const { shell } = useTripShell();
  const currentTripPath = shell.trip_id ? `/trips/${shell.trip_id}` : "/";

  return (
    <div className="not-found-page">
      <a className="skip-link" href="#not-found-main">
        跳到主要内容
      </a>

      <header className="not-found-header">
        <Link className="not-found-brand" to="/" aria-label="ITER AI 首页">
          <img src="/brand/iter-mark-black-64.png" alt="" />
          <strong>ITER AI</strong>
        </Link>
        <span>页面未找到</span>
      </header>

      <main className="not-found-main" id="not-found-main" tabIndex={-1}>
        <section className="not-found-copy" aria-labelledby="not-found-title">
          <p className="not-found-status">
            <span>404</span>
            这一步没有抵达
          </p>
          <h1 id="not-found-title">这条路线，没有找到对应页面。</h1>
          <p className="not-found-description">
            链接可能已经失效，或者页面移动了位置。你的旅行仍然保留着，可以回到当前旅程继续规划。
          </p>
          <div className="not-found-actions">
            <Link className="not-found-primary-action" to={currentTripPath}>
              继续当前旅行
            </Link>
            <Link className="not-found-secondary-action" to="/">
              返回首页
            </Link>
          </div>
        </section>

        <figure className="not-found-route" aria-label="未抵达的旅行路线示意">
          <svg viewBox="0 0 420 300" role="img" aria-hidden="true">
            <path
              className="not-found-route-guide"
              d="M58 229C102 215 111 159 157 151C205 143 208 93 256 88C291 84 308 103 330 85"
            />
            <path
              className="not-found-route-progress"
              d="M58 229C102 215 111 159 157 151C181 147 194 132 207 117"
            />
            <circle className="not-found-route-origin" cx="58" cy="229" r="8" />
            <circle className="not-found-route-stop" cx="157" cy="151" r="6" />
            <circle
              className="not-found-route-break"
              cx="220"
              cy="105"
              r="13"
            />
            <path
              className="not-found-route-cross"
              d="m215 100 10 10m0-10-10 10"
            />
            <circle
              className="not-found-route-destination"
              cx="330"
              cy="85"
              r="20"
            />
            <path
              className="not-found-route-pin"
              d="M330 73a8 8 0 0 0-8 8c0 6 8 14 8 14s8-8 8-14a8 8 0 0 0-8-8Zm0 11a3 3 0 1 1 0-6 3 3 0 0 1 0 6Z"
            />
          </svg>
          <figcaption>
            <span>当前旅程</span>
            <strong>页面在途中走失了</strong>
          </figcaption>
        </figure>
      </main>

      <footer className="not-found-footer">
        <span>ITER AI</span>
        <span>和你一起把想法变成旅程</span>
      </footer>
    </div>
  );
}
