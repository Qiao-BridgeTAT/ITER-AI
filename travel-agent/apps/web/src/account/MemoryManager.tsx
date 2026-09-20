import { useEffect, useState } from "react";
import { TravelApiClient } from "../backend/travelApiClient";
import type { UserMemoryView } from "../generated/v4/contracts";

const api = new TravelApiClient();

export function MemoryManager() {
  const [items, setItems] = useState<UserMemoryView[]>([]);
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let mounted = true;
    void api
      .getMemories()
      .then((result) => {
        if (mounted) setItems(result.memories ?? []);
      })
      .catch(() => {
        if (mounted) setError("暂时无法读取记忆，请稍后重试。");
      });
    return () => {
      mounted = false;
    };
  }, []);
  async function save() {
    if (!text.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await api.createMemory({
        kind: "preference",
        text: text.trim(),
        explicitly_confirmed: true
      });
      setItems(result.memories ?? []);
      setText("");
    } catch {
      setError("记忆未保存，请重试。");
    } finally {
      setBusy(false);
    }
  }
  async function remove(id: string) {
    setBusy(true);
    setError("");
    try {
      await api.deleteMemory(id);
      setItems((items) => items.filter((item) => item.memory_id !== id));
    } catch {
      setError("删除未完成，请重试。");
    } finally {
      setBusy(false);
    }
  }
  return (
    <details className="long-term-memory-manager">
      <summary>已记住的偏好与旅行反馈（{items.length}）</summary>
      <p>用于以后的旅行。本次已确认的行程要求优先。</p>
      {error ? <p role="alert">{error}</p> : null}
      <ul>
        {items.map((item) => (
          <li key={item.memory_id}>
            <span>
              {item.kind === "feedback" ? "旅行反馈：" : ""}
              {item.text}
            </span>
            <button
              type="button"
              disabled={busy}
              onClick={() => void remove(item.memory_id)}
              aria-label={`删除记忆：${item.text}`}
            >
              删除
            </button>
          </li>
        ))}
      </ul>
      <label>
        希望以后记住什么？
        <input
          maxLength={2000}
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              void save();
            }
          }}
        />
      </label>
      <button
        type="button"
        disabled={busy || !text.trim()}
        onClick={() => void save()}
      >
        保存为长期偏好
      </button>
      <p>也可以在对话中说“以后请记住……”，或用“旅行反馈：……”记录这次体验。</p>
    </details>
  );
}
