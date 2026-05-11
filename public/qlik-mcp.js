/**
 * Replaces Chainlit's MCP dialog with Qlik Cloud OAuth form.
 *
 * Plug icon → Tenant URL + Client ID → Connect → OAuth PKCE redirect.
 * After OAuth, JS calls /auth/qlik/connect to store token under the current
 * tab's session key. Next chat message triggers MCP connection server-side.
 */
(function () {
  "use strict";

  let defaults = { tenant_url: "", client_id: "" };
  fetch("/auth/qlik/defaults").then(r => r.json()).then(d => { defaults = d; }).catch(() => {});

  /**
   * Per-tab session key. Stable for the lifetime of a browser tab so that
   * multiple OAuth flows from the same tab land in the same pending slot.
   * Different tabs get different keys, isolating their state.
   */
  function getTabSessionId() {
    let id = sessionStorage.getItem("qlik_tab_session_id");
    if (!id) {
      id = (crypto.randomUUID && crypto.randomUUID()) ||
           (Date.now().toString(36) + Math.random().toString(36).slice(2));
      sessionStorage.setItem("qlik_tab_session_id", id);
    }
    return id;
  }

  // If Chainlit emits its real session ID via send_window_message, prefer that.
  let chainlitSessionId = null;
  window.addEventListener("message", (e) => {
    const data = e.data;
    if (typeof data === "string" && data.startsWith("qlik_session_id:")) {
      chainlitSessionId = data.slice("qlik_session_id:".length);
    } else if (data && typeof data === "object" && data.type === "qlik_session_id") {
      chainlitSessionId = data.id;
    }
  });

  function getSessionKey() {
    return chainlitSessionId || getTabSessionId();
  }

  function replaceDialog(dialog) {
    const title = dialog.querySelector("h2");
    const isMcpDialog = title && (title.textContent.trim() === "MCP Servers");
    const hasConnectTab = Array.from(dialog.querySelectorAll("button")).some(
      b => b.textContent.trim() === "Connect an MCP" || b.textContent.trim() === "Connect to Qlik"
    );
    if (!isMcpDialog && !hasConnectTab) return;
    if (dialog.querySelector("#qlik-form")) return;

    Array.from(dialog.children).forEach(c => (c.style.display = "none"));
    dialog.querySelectorAll('[role="tablist"], [class*="Tabs"]').forEach(t => (t.style.display = "none"));

    const tenantVal = localStorage.getItem("qlik_tenant_url") || defaults.tenant_url || "";
    const clientVal = localStorage.getItem("qlik_client_id") || defaults.client_id || "";

    const form = document.createElement("div");
    form.id = "qlik-form";
    form.style.cssText = "padding:24px;";

    const titleRow = document.createElement("div");
    titleRow.style.cssText = "display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;";
    const h2 = document.createElement("h2");
    h2.textContent = "Connect to Qlik Cloud";
    h2.style.cssText = "font-size:18px;font-weight:700;margin:0;color:#e0e0e0;";
    const xBtn = document.createElement("button");
    xBtn.textContent = "×";
    xBtn.style.cssText = "background:none;border:none;color:#888;font-size:20px;cursor:pointer;";
    titleRow.appendChild(h2);
    titleRow.appendChild(xBtn);
    form.appendChild(titleRow);

    form.appendChild(makeLabel("Qlik Tenant URL"));
    const urlInput = makeInput("https://your-tenant.us.qlikcloud.com", tenantVal);
    form.appendChild(urlInput);

    form.appendChild(makeLabel("OAuth Client ID"));
    const cidInput = makeInput("Client ID from your Qlik tenant admin", clientVal);
    form.appendChild(cidInput);

    const info = document.createElement("div");
    info.style.cssText = "font-size:12px;color:#888;margin-bottom:8px;line-height:1.5;";
    info.innerHTML = 'MCP endpoint: <code style="color:#009845;">&lt;tenant&gt;/api/ai/mcp</code><br/>' +
      'Transport: <code style="color:#009845;">streamable-http</code> with OAuth PKCE';
    form.appendChild(info);

    const help = document.createElement("a");
    help.href = "https://help.qlik.com/en-US/cloud-services/Subsystems/Hub/Content/Sense_Hub/QlikMCP/Connecting-Qlik-MCP-server.htm";
    help.target = "_blank";
    help.textContent = "Qlik MCP setup guide";
    help.style.cssText = "display:block;font-size:12px;color:#006580;margin-bottom:20px;text-decoration:none;";
    form.appendChild(help);

    const status = document.createElement("div");
    status.id = "qlik-status";
    status.style.cssText = "font-size:12px;color:#d32f2f;margin-bottom:12px;min-height:16px;";
    form.appendChild(status);

    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex;justify-content:flex-end;gap:10px;";
    const cancelBtn = makeButton("Cancel", false);
    const connectBtn = makeButton("Connect", true);
    btnRow.appendChild(cancelBtn);
    btnRow.appendChild(connectBtn);
    form.appendChild(btnRow);

    dialog.appendChild(form);

    const closeDialog = () => {
      dialog.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
      const overlay = dialog.closest("[data-state]") || dialog.parentElement;
      if (overlay) overlay.style.display = "none";
    };
    xBtn.onclick = closeDialog;
    cancelBtn.onclick = closeDialog;

    const resetButton = () => {
      connectBtn.textContent = "Connect";
      connectBtn.disabled = false;
      connectBtn.style.background = "#009845";
      connectBtn.style.cursor = "pointer";
    };

    connectBtn.onclick = () => {
      const url = urlInput.value.trim();
      const cid = cidInput.value.trim();
      status.textContent = "";
      if (!url || !cid) {
        urlInput.style.borderColor = url ? "#333" : "#d32f2f";
        cidInput.style.borderColor = cid ? "#333" : "#d32f2f";
        return;
      }
      if (!/^https:\/\//i.test(url)) {
        status.textContent = "Tenant URL must start with https://";
        urlInput.style.borderColor = "#d32f2f";
        return;
      }

      localStorage.setItem("qlik_tenant_url", url);
      localStorage.setItem("qlik_client_id", cid);

      const state = crypto.randomUUID();
      const sessionKey = getSessionKey();
      const params = new URLSearchParams({
        tenant_url: url, client_id: cid, state: state, session_id: sessionKey,
      });
      window.open("/auth/qlik/start?" + params.toString(), "_blank");

      connectBtn.textContent = "Waiting for Qlik approval...";
      connectBtn.disabled = true;
      connectBtn.style.background = "#54565A";
      connectBtn.style.cursor = "wait";

      pollForCompletion(state, sessionKey, closeDialog, (errMsg) => {
        status.textContent = errMsg;
        resetButton();
      });
    };

    setTimeout(() => {
      if (!urlInput.value) urlInput.focus();
      else if (!cidInput.value) cidInput.focus();
    }, 150);
  }

  async function pollForCompletion(state, sessionKey, closeDialog, onError) {
    for (let i = 0; i < 90; i++) {
      await new Promise(r => setTimeout(r, 2000));
      try {
        const resp = await fetch("/auth/qlik/status?state=" + encodeURIComponent(state));
        if (!resp.ok) continue;
        const data = await resp.json();
        if (!data.complete) continue;
        if (!data.access_token) {
          onError("OAuth completed but no access token returned");
          return;
        }

        // Primary path: postMessage to Chainlit's on_window_message hook so
        // the MCP connection happens inside the user's WebSocket session
        // context — no extra chat message required to trigger it.
        try {
          window.postMessage({
            type: "qlik_oauth_complete",
            access_token: data.access_token,
            tenant_url: data.tenant_url,
            client_id: data.client_id,
          }, window.location.origin);
        } catch (e) {
          // Origin mismatch or other issue — fall back to the POST below.
        }

        // Fallback: POST to /auth/qlik/connect. Server stores the token keyed
        // by session and the user's next chat message picks it up. Harmless
        // duplicate if the window-message handler already ran.
        const sessionId = data.session_id || sessionKey;
        try {
          const connectResp = await fetch("/auth/qlik/connect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              access_token: data.access_token,
              tenant_url: data.tenant_url,
              client_id: data.client_id,
              session_id: sessionId,
            }),
          });
          if (!connectResp.ok) {
            // Best-effort: log but don't surface, since the window-message
            // path may have succeeded.
            let errText = "Server rejected the token";
            try { errText = (await connectResp.json()).error || errText; } catch {}
            console.warn("Qlik fallback connect failed:", errText);
          }
        } catch (e) {
          console.warn("Qlik fallback connect threw:", e);
        }
        closeDialog();
        return;
      } catch (e) {
        // Network blip — keep polling. Will hit timeout below if persistent.
      }
    }
    onError("Timed out waiting for Qlik approval (3 minutes).");
  }

  function makeLabel(text) {
    const l = document.createElement("label");
    l.textContent = text;
    l.style.cssText = "display:block;font-size:14px;font-weight:600;margin-bottom:6px;color:#e0e0e0;";
    return l;
  }

  function makeInput(placeholder, value) {
    const i = document.createElement("input");
    i.type = "text";
    i.placeholder = placeholder;
    i.value = value || "";
    i.style.cssText = "width:100%;padding:10px 12px;border-radius:6px;border:1px solid #333;background:#1a2632;color:#e0e0e0;font-size:14px;margin-bottom:16px;box-sizing:border-box;outline:none;";
    i.addEventListener("keydown", e => e.stopPropagation());
    i.addEventListener("keyup", e => e.stopPropagation());
    i.addEventListener("keypress", e => e.stopPropagation());
    i.addEventListener("focus", () => i.style.borderColor = "#009845");
    i.addEventListener("blur", () => i.style.borderColor = "#333");
    return i;
  }

  function makeButton(text, primary) {
    const b = document.createElement("button");
    b.textContent = text;
    b.style.cssText = primary
      ? "padding:8px 20px;border-radius:6px;border:none;background:#009845;color:white;font-size:14px;font-weight:600;cursor:pointer;"
      : "padding:8px 20px;border-radius:6px;border:1px solid #444;background:transparent;color:#e0e0e0;font-size:14px;cursor:pointer;";
    return b;
  }

  new MutationObserver(mutations => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node.nodeType !== 1) continue;
        const dlg = node.getAttribute?.("role") === "dialog" ? node : node.querySelector?.('[role="dialog"]');
        if (dlg) {
          setTimeout(() => replaceDialog(dlg), 80);
          setTimeout(() => replaceDialog(dlg), 250);
        }
      }
    }
  }).observe(document.body, { childList: true, subtree: true });
})();
