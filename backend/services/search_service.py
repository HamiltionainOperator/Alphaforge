from __future__ import annotations

import html
import os
import re
from urllib.parse import unquote

import httpx


# Free web search for the research brief. No OpenRouter credits involved.
# Provider auto-selection (first that is configured wins):
#   - Tavily  (TAVILY_API_KEY)  — free tier ~1000 searches/mo, best quality
#   - Brave   (BRAVE_API_KEY)   — free tier ~2000 searches/mo
#   - DuckDuckGo (no key)       — zero setup, but can rate-limit / get flaky
# Everything is best-effort: any failure returns [] so research() degrades to
# model knowledge rather than erroring the whole forge.

_TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)
_TAG = re.compile(r"<[^>]+>")
# Lookahead requires the class anywhere in the <a> tag, so href may appear in any order.
_DDG_LINK = re.compile(r'<a\b(?=[^>]*class="result__a")[^>]*\bhref="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)
_DDG_SNIP = re.compile(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.S | re.I)
_LITE_LINK = re.compile(r'<a\b(?=[^>]*class="result-link")[^>]*\bhref="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)
_LITE_SNIP = re.compile(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', re.S | re.I)
# A realistic browser UA reduces DuckDuckGo blocking the scrape.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def search_provider() -> str:
    if os.getenv("TAVILY_API_KEY", "").strip():
        return "tavily"
    if os.getenv("BRAVE_API_KEY", "").strip():
        return "brave"
    return "duckduckgo"


async def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Return up to max_results [{title, url, content}] from the active free provider."""
    q = (query or "").strip()
    if not q:
        return []
    n = max(1, min(int(max_results or 5), 10))
    tavily = os.getenv("TAVILY_API_KEY", "").strip()
    brave = os.getenv("BRAVE_API_KEY", "").strip()
    try:
        if tavily:
            return await _tavily(q, n, tavily)
        if brave:
            return await _brave(q, n, brave)
        return await _duckduckgo(q, n)
    except Exception:  # noqa: BLE001 — search is best-effort
        return []


async def _tavily(query: str, n: int, key: str) -> list[dict[str, str]]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": query, "max_results": n, "search_depth": "basic"},
        )
    if r.status_code >= 400:
        return []
    out: list[dict[str, str]] = []
    for item in (r.json().get("results") or [])[:n]:
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        out.append({"title": (str(item.get("title") or url))[:160], "url": url, "content": (str(item.get("content") or ""))[:600]})
    return out


async def _brave(query: str, n: int, key: str) -> list[dict[str, str]]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": n},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
        )
    if r.status_code >= 400:
        return []
    results = ((r.json().get("web") or {}).get("results")) or []
    out: list[dict[str, str]] = []
    for item in results[:n]:
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        out.append({"title": (str(item.get("title") or url))[:160], "url": url, "content": _strip_html(str(item.get("description") or ""))[:600]})
    return out


async def _duckduckgo(query: str, n: int) -> list[dict[str, str]]:
    # The main html endpoint is the best source; fall back to the lite endpoint
    # (different host/markup) when it returns nothing or gets blocked.
    results = await _ddg_endpoint("https://html.duckduckgo.com/html/", query, n, _DDG_LINK, _DDG_SNIP)
    if results:
        return results
    return await _ddg_endpoint("https://lite.duckduckgo.com/lite/", query, n, _LITE_LINK, _LITE_SNIP)


async def _ddg_endpoint(url: str, query: str, n: int, link_re: "re.Pattern[str]", snip_re: "re.Pattern[str]") -> list[dict[str, str]]:
    headers = {
        "User-Agent": _BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers, follow_redirects=True) as client:
            r = await client.post(url, data={"q": query})
    except Exception:  # noqa: BLE001
        return []
    if r.status_code >= 400:
        return []
    body = r.text or ""
    links = link_re.findall(body)
    snippets = snip_re.findall(body)
    out: list[dict[str, str]] = []
    for i, (href, title_html) in enumerate(links):
        clean_url = _ddg_unwrap(href)
        if not clean_url:  # skip ads / internal redirects
            continue
        snippet = _strip_html(snippets[i]) if i < len(snippets) else ""
        out.append({"title": _strip_html(title_html)[:160] or clean_url, "url": clean_url, "content": snippet[:600]})
        if len(out) >= n:
            break
    return out


def _ddg_unwrap(href: str) -> str:
    h = html.unescape(href or "")
    if h.startswith("//"):
        h = "https:" + h
    match = re.search(r"[?&]uddg=([^&]+)", h)
    if match:
        h = unquote(match.group(1))
    if not h.startswith("http"):
        return ""
    # Skip DuckDuckGo ad / redirect / internal links (e.g. .../y.js?ad_domain=...).
    if re.match(r"https?://[^/]*duckduckgo\.com/", h):
        return ""
    return h


def _strip_html(text: str) -> str:
    return html.unescape(_TAG.sub("", text or "")).strip()
