"""Chainlit chat app with AWS Bedrock LLM and Qlik MCP integration.

Plug icon → Qlik Cloud form → OAuth PKCE → streamable-http MCP connection.
Gear icon → LLM provider (Bedrock), model, and settings.
Supports chart rendering via Plotly when data is returned from Qlik MCP.
"""

import hashlib
import json
import os
import sys
import tempfile
import traceback

import chainlit as cl
from chainlit.input_widget import Select, Slider, TextInput
from chainlit.server import app as fastapi_app
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from typing import cast

from qlik_oauth import register_oauth_routes, pending_connections

load_dotenv()

logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

# Register OAuth routes
register_oauth_routes(fastapi_app)

BEDROCK_MODELS = {
    "Claude 4 Sonnet": "anthropic.claude-sonnet-4-20250514-v1:0",
    "Amazon Nova Pro": "amazon.nova-pro-v1:0",
    "Meta Llama 3.3 70B": "meta.llama3-3-70b-instruct-v1:0",
}

AWS_REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "ap-northeast-1"]

# Cross-region inference profile prefixes by AWS region group.
# See: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
_BEDROCK_REGION_PREFIX = {
    "us": "us",
    "eu": "eu",
    "ap": "apac",
}

# System prompt for the ReAct agent. The tool list is supplied at runtime by
# MCP; the prompt only provides routing hints — keep these generic so the
# prompt doesn't drift if Qlik adds or renames tools.
SYSTEM_PROMPT = """You are a Qlik Cloud data analyst assistant. Use the available tools to answer questions about the user's Qlik Cloud tenant. Always call tools — never guess or make up data.

QUESTION ROUTING — match the user's question to the right tool:

FINDING THINGS:
- "What apps/datasets/spaces/data products do I have?" → qlik_search(query='*', resourceType='app|dataset|space|dataproduct')
- "Find [something]" → qlik_search(query='search term')
- "Tell me about app X" → qlik_search to find ID, then qlik_describe_app(appId=ID)

APP EXPLORATION:
- "What sheets are in app X?" → qlik_list_sheets(appId=ID)
- "What fields are available?" → qlik_get_fields(appId=ID)
- "What dimensions/measures exist?" → qlik_list_dimensions(appId=ID) / qlik_list_measures(appId=ID)
- "Show me chart data" → qlik_get_chart_data(appId=ID, chartId=ID)

DATA ANALYSIS:
- "What values are in field X?" → qlik_get_field_values(appId=ID, fieldName='X')
- "Filter by X=Y" → qlik_select_values(appId=ID, selections=[{field:'X', values:['Y']}])
- "Clear filters" → qlik_clear_selections(appId=ID)
- "Calculate/aggregate something" → qlik_create_data_object with dimensions and measures

DATASETS:
- "Show me dataset schema/columns" → qlik_get_dataset_schema(datasetId=ID)
- "Preview dataset" → qlik_get_dataset_sample(datasetId=ID)
- "Data quality/trust score" → qlik_get_dataset_trust_score(datasetId=ID)
- "When was it last updated?" → qlik_get_dataset_freshness(datasetId=ID)
- "Data lineage" → qlik_get_lineage(qri=QRI)

DATA PRODUCTS:
- "List data products" → qlik_search(query='*', resourceType='dataproduct')
- "Data product details" → qlik_get_data_product(dataProductId=ID)
- "Create data product" → qlik_create_data_product(name='...')

GLOSSARY:
- "Search glossary terms" → qlik_search_glossary_terms(glossaryId=ID)
- "Create glossary term" → qlik_create_glossary_term(glossaryId=ID, name='...')

BUILDING VISUALIZATIONS:
- "Create a sheet" → qlik_create_sheet(appId=ID, title='...')
- "Add a bar/line/pie chart" → qlik_add_chart(appId=ID, sheetId=ID, chartType='barchart', ...)
- "Add a filter" → qlik_add_filter(appId=ID, sheetId=ID, ...)

CHART RENDERING:
- When you get data from qlik_get_chart_data or qlik_create_data_object, present it as a table
- If the user asks to "chart", "plot", or "visualize" data, include a JSON code block with chart spec:
  ```chart
  {"type": "bar", "title": "Sales by Region", "x": ["North", "South", "East"], "y": [100, 200, 150], "xlabel": "Region", "ylabel": "Sales"}
  ```
  Supported types: bar, line, pie, scatter

RULES:
- Always call qlik_search FIRST to find IDs before using other tools
- The query parameter is REQUIRED for qlik_search — use '*' to match everything
- Present results clearly with counts, tables, and bullet points
- If a tool returns empty results, say so clearly
- Chain multiple tool calls when needed (search → describe → list sheets)"""

QLIK_MCP_HELP_URL = "https://help.qlik.com/en-US/cloud-services/Subsystems/Hub/Content/Sense_Hub/QlikMCP/Connecting-Qlik-MCP-server.htm"


# ---------------------------------------------------------------------------
# LLM Provider Factory
# ---------------------------------------------------------------------------

def _bedrock_inference_prefix(region: str) -> str:
    """Map an AWS region to its cross-region inference profile prefix."""
    group = region.split("-", 1)[0].lower()
    return _BEDROCK_REGION_PREFIX.get(group, "us")


def get_chat_model(model_name, region="us-east-1", temperature=0.2,
                   max_tokens=4096, api_key=None):
    """Create a Bedrock chat model.

    The Bedrock bearer token (AWS_BEARER_TOKEN_BEDROCK) is a process-wide
    credential picked up by boto3. To avoid one user's UI-supplied key
    clobbering another's, the env var is set only once via setdefault — the
    first non-empty value wins. Operators should prefer setting it in .env.
    """
    from langchain_aws.chat_models import ChatBedrockConverse
    if api_key:
        if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
        elif os.environ["AWS_BEARER_TOKEN_BEDROCK"] != api_key:
            logger.warning(
                "AWS_BEARER_TOKEN_BEDROCK already set; ignoring UI-supplied key. "
                "Configure via .env to change."
            )
    client = boto3.client(
        "bedrock-runtime", region_name=region,
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=60),
    )
    model_id = BEDROCK_MODELS.get(model_name, model_name)
    prefix = _bedrock_inference_prefix(region)
    return ChatBedrockConverse(
        model=f"{prefix}.{model_id}", client=client,
        temperature=temperature, max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Chart Rendering
# ---------------------------------------------------------------------------

async def render_charts(text: str) -> str:
    """Extract ```chart blocks and render as Plotly images inline."""
    if "```chart" not in text:
        return text

    import plotly.graph_objects as go
    import plotly.io as pio

    parts = text.split("```chart")
    result = parts[0]

    for part in parts[1:]:
        if "```" in part:
            chart_json, rest = part.split("```", 1)
            try:
                spec = json.loads(chart_json.strip())
                chart_type = spec.get("type", "bar")
                title = spec.get("title", "Chart")
                x = spec.get("x", [])
                y = spec.get("y", [])
                xlabel = spec.get("xlabel", "")
                ylabel = spec.get("ylabel", "")

                fig = go.Figure()

                if chart_type == "bar":
                    fig.add_trace(go.Bar(x=x, y=y, marker_color="#009845"))
                elif chart_type == "line":
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color="#009845")))
                elif chart_type == "pie":
                    fig.add_trace(go.Pie(labels=x, values=y))
                elif chart_type == "scatter":
                    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color="#009845", size=10)))

                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    template="plotly_dark",
                    paper_bgcolor="#0f1a24",
                    plot_bgcolor="#1a2632",
                    font=dict(color="#e0e0e0"),
                    margin=dict(l=40, r=40, t=60, b=40),
                )

                # Save as image — stable filename across processes
                chart_id = hashlib.md5(chart_json.strip().encode("utf-8")).hexdigest()[:12]
                img_path = os.path.join(tempfile.gettempdir(), f"chart_{chart_id}.png")
                pio.write_image(fig, img_path, width=700, height=400)

                # Send as Chainlit image element. Chainlit copies the file into
                # the session's element store, so we can remove our temp copy.
                elements = [cl.Image(name=title, path=img_path, display="inline")]
                await cl.Message(content=f"**{title}**", elements=elements).send()
                try:
                    os.remove(img_path)
                except OSError:
                    pass

                result += rest
            except Exception as e:
                logger.warning(f"Chart rendering failed: {e}")
                result += f"\n(Chart rendering failed: {e})\n" + rest
        else:
            result += part

    return result


# ---------------------------------------------------------------------------
# Qlik MCP Connection (streamable-http + OAuth Bearer token)
# ---------------------------------------------------------------------------

def _validate_qlik_tenant(tenant_url: str) -> str:
    """Reject non-https URLs and hostnames outside the Qlik Cloud allowlist.

    Prevents SSRF: a user-supplied URL flows into outbound HTTP calls (token
    exchange + MCP requests). Restrict to qlikcloud.com/qlik-stage.com domains
    or an explicit allowlist via QLIK_TENANT_ALLOWLIST (comma-separated suffixes).
    """
    from urllib.parse import urlparse
    parsed = urlparse(tenant_url)
    if parsed.scheme != "https":
        raise ValueError("Qlik tenant URL must use https://")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Qlik tenant URL is missing a hostname")
    extra = os.getenv("QLIK_TENANT_ALLOWLIST", "")
    allowed_suffixes = [".qlikcloud.com", ".qlik-stage.com"] + [
        s.strip().lower() for s in extra.split(",") if s.strip()
    ]
    if not any(host == s.lstrip(".") or host.endswith(s) for s in allowed_suffixes):
        raise ValueError(
            f"Qlik tenant host {host!r} not in allowlist. "
            f"Set QLIK_TENANT_ALLOWLIST=.your-domain.com to permit it."
        )
    # Preserve netloc so self-hosted tenants on non-443 ports keep their port.
    return f"https://{parsed.netloc.lower()}"


def _flatten_exception_group(eg: BaseException) -> list[str]:
    """Walk a possibly nested ExceptionGroup into flat 'Type: msg' strings."""
    out: list[str] = []

    def visit(e: BaseException) -> None:
        if isinstance(e, BaseExceptionGroup):  # noqa: F821 — 3.11+ builtin
            for sub in e.exceptions:
                visit(sub)
        else:
            out.append(f"{type(e).__name__}: {e}")

    visit(eg)
    return out


def _escape_for_codeblock(text: str) -> str:
    """Make text safe to drop inside a fenced markdown code block."""
    # Replace any triple-backtick that would close the fence early.
    return str(text).replace("```", "ʼʼʼ")


class _ChartFenceFilter:
    """Stream-time filter that strips ```chart …``` fenced blocks token by token.

    The model is instructed to emit chart JSON inside ```chart fences; we render
    those as separate image messages and don't want the raw JSON to appear in
    the streamed text. Because tokens are arbitrary chunks (the marker can
    span multiple tokens), the filter buffers a short tail until it knows
    whether it's about to enter a fence.
    """

    _OPEN = "```chart"
    _CLOSE = "```"

    def __init__(self) -> None:
        self._buf = ""
        self._in_chart = False

    def feed(self, token: str) -> str:
        """Consume a token; return only the portion safe to stream out now."""
        out_parts: list[str] = []
        self._buf += token

        while True:
            if not self._in_chart:
                idx = self._buf.find(self._OPEN)
                if idx >= 0:
                    out_parts.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self._OPEN):]
                    self._in_chart = True
                    continue
                # Hold back enough chars that a partial _OPEN can't slip through.
                hold = len(self._OPEN) - 1
                if len(self._buf) > hold:
                    out_parts.append(self._buf[:-hold])
                    self._buf = self._buf[-hold:]
                break
            else:
                idx = self._buf.find(self._CLOSE)
                if idx >= 0:
                    self._buf = self._buf[idx + len(self._CLOSE):]
                    self._in_chart = False
                    continue
                # Stay inside the fence; drop everything except a possible
                # partial closing marker.
                hold = len(self._CLOSE) - 1
                self._buf = self._buf[-hold:] if hold else ""
                break

        return "".join(out_parts)

    def flush(self) -> str:
        """Emit any trailing buffered text once streaming ends."""
        if self._in_chart:
            return ""  # unclosed fence — drop, the chart will be rendered separately
        tail, self._buf = self._buf, ""
        return tail


async def connect_qlik_mcp(tenant_url: str, access_token: str, client_id: str):
    """Connect to Qlik MCP using streamable-http with Bearer token + X-Agent-Id."""
    tenant_url = _validate_qlik_tenant(tenant_url)
    mcp_url = f"{tenant_url}/api/ai/mcp"
    logger.info(f"Connecting to MCP: {mcp_url}")

    mcp_client = MultiServerMCPClient({
        "qlik": {
            "url": mcp_url,
            "transport": "streamable_http",
            "headers": {
                "Authorization": f"Bearer {access_token}",
                "X-Agent-Id": client_id,
            },
        },
    })
    try:
        tools = await mcp_client.get_tools()
    except BaseExceptionGroup as eg:  # noqa: F821 — 3.11+ builtin
        raise RuntimeError("; ".join(_flatten_exception_group(eg))) from eg
    logger.info(f"MCP connected with {len(tools)} tools")
    return mcp_client, tools


async def disconnect_qlik_mcp():
    """Close the MCP client and clear session state."""
    mcp_client = cl.user_session.get("mcp_client")
    if mcp_client is not None:
        for close_method in ("aclose", "close", "__aexit__"):
            fn = getattr(mcp_client, close_method, None)
            if fn is None:
                continue
            try:
                result = fn() if close_method != "__aexit__" else fn(None, None, None)
                if hasattr(result, "__await__"):
                    await result
                break
            except Exception as e:
                logger.debug(f"MCP client {close_method}() failed: {e}")
    cl.user_session.set("mcp_client", None)
    cl.user_session.set("mcp_tools", None)
    cl.user_session.set("agent", None)


def build_agent_if_ready():
    chat_model = cl.user_session.get("chat_model")
    tools = cl.user_session.get("mcp_tools")
    if chat_model and tools:
        agent = create_react_agent(chat_model, tools, prompt=SYSTEM_PROMPT)
        cl.user_session.set("agent", agent)
        return agent
    return None


# ---------------------------------------------------------------------------
# Chat Lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    default_api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")

    settings = cl.ChatSettings([
        TextInput(id="bedrock_api_key", label="Bedrock API Key", initial=default_api_key,
                  placeholder="bedrock-api-key-...", description="Generate from Bedrock console > API keys"),
        Select(id="aws_region", label="AWS Region", values=AWS_REGIONS, initial_value=default_region,
               description="Must match the region where you generated the API key"),
        Select(id="bedrock_model", label="Bedrock Model", values=list(BEDROCK_MODELS.keys()),
               initial_value="Claude 4 Sonnet", description="Claude Sonnet 4 recommended for tool calling"),
        Slider(id="temperature", label="Temperature", initial=0.2, min=0.0, max=1.0, step=0.1,
               description="Controls randomness in responses"),
        Slider(id="max_tokens", label="Max Tokens", initial=4096, min=256, max=32768, step=256,
               description="Maximum response length"),
    ])
    await settings.send()

    chat_model = get_chat_model("Claude 4 Sonnet", default_region, 0.2, 4096, default_api_key)
    cl.user_session.set("chat_model", chat_model)

    await cl.Message(
        content=(
            "## Your Friendly Neighborhood AI Assistant\n\n"
            "Ask me anything — or connect to Qlik Cloud for data access."
        )
    ).send()


@cl.on_chat_end
async def on_chat_end():
    """Release the MCP client when the user closes the chat."""
    try:
        await disconnect_qlik_mcp()
    except Exception as e:
        logger.debug(f"on_chat_end cleanup failed: {e}")


async def _connect_mcp_for_session(token: str, tenant: str, client_id: str) -> None:
    """Disconnect any existing MCP client and (re)connect with the given creds.

    Runs in the user's Chainlit session context — used by both
    on_window_message (immediate) and on_message (deferred fallback).
    """
    cl.user_session.set("qlik_access_token", token)
    cl.user_session.set("qlik_tenant_url", tenant)
    cl.user_session.set("qlik_client_id", client_id)
    try:
        await disconnect_qlik_mcp()
        mcp_client, tools = await connect_qlik_mcp(tenant, token, client_id)
        cl.user_session.set("mcp_client", mcp_client)
        cl.user_session.set("mcp_tools", tools)
        build_agent_if_ready()
        await cl.Message(
            content=f"Connected to Qlik MCP — **{len(tools)} tools** available. Ask me anything!"
        ).send()
    except Exception as e:
        logger.error(f"MCP connection failed: {e}")
        safe = _escape_for_codeblock(e)
        await cl.Message(content=f"Qlik MCP connection failed:\n```\n{safe}\n```").send()


@cl.on_window_message
async def on_window_message(message) -> None:
    """Receive OAuth completion directly from JS, bound to this WebSocket session.

    Chainlit forwards `window.postMessage` events from the page to this hook,
    which runs in the user's session context — so the MCP connection is bound
    to the correct cl.user_session without needing the pending_connections
    side-channel.
    """
    data = message
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return
    if not isinstance(data, dict) or data.get("type") != "qlik_oauth_complete":
        return
    token = data.get("access_token")
    tenant = data.get("tenant_url")
    if not token or not tenant:
        return
    try:
        _validate_qlik_tenant(tenant)
    except ValueError as e:
        await cl.Message(content=f"Qlik tenant rejected: {_escape_for_codeblock(e)}").send()
        return
    # Mark this session as already connected so the deferred fallback in
    # on_message doesn't try to connect again with the same token.
    cl.user_session.set("qlik_connected_via_window_msg", True)
    await _connect_mcp_for_session(token, tenant, data.get("client_id", ""))


@cl.on_settings_update
async def on_settings_update(settings: dict):
    temperature = settings.get("temperature") or 0.2
    max_tokens = int(settings.get("max_tokens") or 4096)

    api_key = (settings.get("bedrock_api_key") or "").strip()
    model_name = settings.get("bedrock_model") or "Claude 4 Sonnet"
    region = settings.get("aws_region") or "us-east-1"
    chat_model = get_chat_model(model_name, region, temperature, max_tokens, api_key)
    cl.user_session.set("chat_model", chat_model)
    build_agent_if_ready()
    await cl.Message(content=f"Settings updated: **Bedrock** → **{model_name}** in **{region}**").send()


def _consume_pending_connection() -> dict | None:
    """Pop the pending OAuth result keyed strictly to this Chainlit session.

    The OAuth popup stores tokens keyed by the per-tab session ID that the JS
    passed through /auth/qlik/start. Each Chainlit session matches a single
    browser tab; with on_window_message also firing immediately, this
    fallback only runs when window.postMessage didn't reach the backend
    (e.g. JS disabled in a downstream proxy). No cross-session fallback —
    that would let user A pick up user B's pending token.
    """
    session_id = cl.context.session.id
    return pending_connections.pop(session_id, None)


@cl.on_message
async def on_message(message: cl.Message):
    # Fallback path: pick up any pending OAuth result that the window-message
    # channel missed. Skipped if the window-message handler already connected.
    if not cl.user_session.get("qlik_connected_via_window_msg"):
        pending = _consume_pending_connection()
        if pending:
            await _connect_mcp_for_session(
                pending["access_token"], pending["tenant_url"], pending["client_id"],
            )

    agent = cast(CompiledStateGraph | None, cl.user_session.get("agent"))

    # No agent — use LLM directly
    if not agent:
        chat_model = cl.user_session.get("chat_model")
        if not chat_model:
            await cl.Message(content="No LLM configured. Check your settings.").send()
            return
        try:
            resp = await chat_model.ainvoke([HumanMessage(content=message.content)])
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            text += "\n\n---\n*No Qlik MCP connected. Click the **plug icon** to connect.*"
            text = await render_charts(text)
            await cl.Message(content=text).send()
        except Exception as e:
            await cl.Message(content=f"LLM Error: {str(e)}").send()
            logger.error(traceback.format_exc())
        return

    # Agent with MCP tools
    config = RunnableConfig(configurable={"thread_id": cl.context.session.id})
    response_message = cl.Message(content="")
    full_text = ""
    response_sent = False
    chart_filter = _ChartFenceFilter()

    async def emit(token: str) -> None:
        nonlocal full_text
        full_text += token
        visible = chart_filter.feed(token)
        if visible:
            await response_message.stream_token(visible)

    try:
        async for msg, metadata in agent.astream(
            {"messages": message.content}, stream_mode="messages", config=config,
        ):
            if isinstance(msg, AIMessageChunk) and msg.content:
                if isinstance(msg.content, str):
                    await emit(msg.content)
                elif (isinstance(msg.content, list) and len(msg.content) > 0
                      and isinstance(msg.content[0], dict) and msg.content[0].get("type") == "text"):
                    await emit(msg.content[0]["text"])
        # Flush any tail held back by the filter.
        tail = chart_filter.flush()
        if tail:
            await response_message.stream_token(tail)
        await response_message.send()
        response_sent = True

        # Render any charts in the response
        if "```chart" in full_text:
            await render_charts(full_text)

    except Exception as e:
        # Flush partial stream once, before error message, so content isn't lost.
        if full_text and not response_sent:
            try:
                await response_message.send()
            except Exception:
                logger.debug("Failed to flush partial response after error")
        error_str = str(e).lower()
        if any(kw in error_str for kw in ["timeout", "closed", "connection", "eof", "reset"]):
            cl.user_session.set("agent", None)
            await cl.Message(content="Connection lost. Click the **plug icon** to reconnect.").send()
        else:
            await cl.Message(content=f"Error: {_escape_for_codeblock(e)}").send()
            logger.error(traceback.format_exc())
