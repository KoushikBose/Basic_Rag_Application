const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");

function addMessage(text, cls) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addSources(msgEl, sources) {
  if (!sources || !sources.length) return;
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `Sources (${sources.length})`;
  details.appendChild(summary);
  sources.forEach((s) => {
    const el = document.createElement("div");
    el.className = "source";
    const title = document.createElement("b");
    title.textContent = s.source + (s.page != null ? ` — page ${s.page + 1}` : "");
    const body = document.createElement("div");
    body.textContent = s.text;
    el.append(title, body);
    details.appendChild(el);
  });
  msgEl.appendChild(details);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";
  send.disabled = input.disabled = true;
  const typing = addMessage("Thinking…", "bot typing");

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    typing.remove();
    if (!res.ok) {
      addMessage(data.error || "Something went wrong.", "bot error");
    } else {
      const msg = addMessage(data.answer, "bot");
      addSources(msg, data.sources);
    }
  } catch (err) {
    typing.remove();
    addMessage("Could not reach the server: " + err.message, "bot error");
  } finally {
    send.disabled = input.disabled = false;
    input.focus();
    chat.scrollTop = chat.scrollHeight;
  }
});
