"""
kernel_app/main.py
==================
A Kernel App: "library vs. the live web." Given a question, it asks the
raglib library (on Railway) and, in parallel, has a Kernel browser search the
web and read the top results — returning both so you can compare what the
canon says against what the internet says today.

Kept in its own folder so `kernel deploy` uploads only this app, not the DBs.

Deploy & run:
  cd kernel_app
  kernel deploy main.py
  kernel invoke raglib-vs-web compare --payload '{"query": "What is the sea a metaphor for?"}'
"""

import asyncio
import json
from urllib.parse import quote_plus

import requests
from kernel import App, Kernel, KernelContext
from playwright.async_api import async_playwright

RAGLIB_URL = "https://onlychr.is/raglib"

kernel = Kernel()
app = App("raglib-vs-web")


def ask_library(query: str) -> dict:
    answer, sources = [], []
    with requests.post(f"{RAGLIB_URL}/api/query", json={"query": query}, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "token":
                answer.append(event["data"])
            elif event["type"] == "sources":
                sources = [s["metadata"].get("title") or s["metadata"].get("source") for s in event["data"]]
    return {"answer": "".join(answer), "sources": sources}


async def ask_web(ctx: KernelContext, query: str, n: int = 3) -> list[dict]:
    kb = kernel.browsers.create(invocation_id=ctx.invocation_id, stealth=True)
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(kb.cdp_ws_url)
        try:
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(f"https://duckduckgo.com/html/?q={quote_plus(query)}")
            links = await page.eval_on_selector_all(
                "a.result__a", f"as => as.slice(0, {n}).map(a => ({{title: a.textContent, url: a.href}}))"
            )
            for link in links:
                try:
                    await page.goto(link["url"], wait_until="domcontentloaded", timeout=20000)
                    text = await page.inner_text("body")
                    results.append({**link, "excerpt": " ".join(text.split())[:1500]})
                except Exception as e:
                    results.append({**link, "error": str(e)})
        finally:
            await browser.close()
    return results


@app.action("compare")
async def compare(ctx: KernelContext, payload: dict) -> dict:
    query = payload["query"]
    library, web = await asyncio.gather(asyncio.to_thread(ask_library, query), ask_web(ctx, query))
    return {"query": query, "library": library, "web": web}
