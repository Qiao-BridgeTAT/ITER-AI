import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./styles/app-boot.css";
import "./styles/app-shell.css";
import "./styles/conversation-plan-ready.css";
import "./styles/global.css";
import "./styles/tokens.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Application root element was not found");
}

document.documentElement.classList.add("app-is-booting");

// Catch entry-module failures too; a React boundary cannot catch static imports.
void import("./app/App")
  .then(({ App }) => {
    createRoot(root).render(
      <StrictMode>
        <App />
      </StrictMode>,
    );
  })
  .catch(() => {
    document.documentElement.classList.remove("app-is-booting");
    const message = document.createElement("p");
    message.setAttribute("role", "alert");
    message.textContent =
      "页面暂时未能加载，请刷新重试。旅行内容不会因此被修改。";
    const retry = document.createElement("button");
    retry.type = "button";
    retry.textContent = "重新加载";
    retry.addEventListener("click", () => window.location.reload());
    root.replaceChildren(message, retry);
  });
