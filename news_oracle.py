"""
news_oracle.py
==============
This week's news, read against the library. A Kernel browser collects stories
published in the last N days from a few news sites; each story is matched to
the literary passages and Leiden community it most resembles, and Claude
writes one line on what the old text saw coming.

Usage:
  python news_oracle.py                   # last 3 days, top 10 stories
  python news_oracle.py --days 2 --top 6 --no-explain

Writes news_oracle_results.json. Requires KERNEL_API_KEY, OPENAI_API_KEY and
ANTHROPIC_API_KEY (the last only for the explanations).
"""

import argparse
import asyncio
import json
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

import discover                      # Kernel client, robots.txt helpers, taste vectors
import passage_match as pm

# Section front → pattern that article URLs on that site follow.
SOURCES = {
    "https://www.theguardian.com/world":       r"/\d{4}/[a-z]{3}/\d{2}/",
    "https://www.theguardian.com/us-news":     r"/\d{4}/[a-z]{3}/\d{2}/",
    "https://www.theguardian.com/technology":  r"/\d{4}/[a-z]{3}/\d{2}/",
    "https://www.theguardian.com/culture":     r"/\d{4}/[a-z]{3}/\d{2}/",
    "https://www.npr.org/sections/news/":      r"/\d{4}/\d{2}/\d{2}/",
    "https://apnews.com/":                     r"/article/",
    "https://www.bbc.com/news":                r"/news/articles/",
    "https://www.wired.com/":                  r"/story/",
    "https://www.cnn.com/":                    r"cnn\.com/\d{4}/\d{2}/\d{2}/",
    "https://www.propublica.org/":             r"/article/",
    "https://www.politico.com/":               r"politico\.com/news/\d{4}/\d{2}/\d{2}/",
}
SKIP_URL = re.compile(r"/(live|video|audio|gallery|podcasts?|crosswords?|newsletters?)/|/live-", re.I)
CLAUDE_MODEL = "claude-sonnet-4-6"


def story_links(html: str, front: str, pattern: str, limit: int) -> list[str]:
    host = urlparse(front).netloc
    links = []
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        href = urljoin(front, a["href"]).split("#")[0].split("?")[0]
        if (urlparse(href).netloc == host and re.search(pattern, href)
                and not SKIP_URL.search(href) and href not in links):
            links.append(href)
    return links[:limit]


def published_at(soup: BeautifulSoup, url: str) -> datetime | None:
    """Publish time from meta tags, JSON-LD, <time>, or a date in the URL."""
    candidates = [
        (soup.find("meta", property="article:published_time") or {}).get("content"),
        (soup.find("meta", attrs={"name": re.compile(r"^(publish-date|pubdate|date|parsely-pub-date)$", re.I)}) or {}).get("content"),
    ]
    for s in soup.find_all("script", type="application/ld+json"):
        m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', s.string or "")
        if m:
            candidates.append(m.group(1))
    t = soup.find("time", datetime=True)
    if t:
        candidates.append(t["datetime"])
    for c in candidates:
        if not c:
            continue
        try:
            dt = datetime.fromisoformat(c.strip().replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    m = re.search(r"/(\d{4})/(\d{2}|[a-z]{3})/(\d{2})/", url)
    if m:
        y, mo, d = m.groups()
        fmt = "%Y/%b/%d" if mo.isalpha() else "%Y/%m/%d"
        return datetime.strptime(f"{y}/{mo}/{d}", fmt).replace(tzinfo=timezone.utc)
    return None


def extract_story(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    og = lambda p: (soup.find("meta", property=p) or {}).get("content", "")
    title = og("og:title") or (soup.title.string if soup.title and soup.title.string else url)
    when = published_at(soup, url)
    for tag in soup(["script", "style", "nav", "noscript", "svg", "iframe", "aside", "form", "figure"]):
        tag.decompose()
    body = soup.find("article") or soup.find("main") or soup.body or soup
    text = " ".join(p.get_text(" ", strip=True) for p in body.find_all("p") if len(p.get_text(strip=True)) > 60)
    return {"url": url, "site": urlparse(url).netloc.removeprefix("www."), "title": title.strip(),
            "description": og("og:description").strip(), "published": when.isoformat() if when else None,
            "text": text[:2000]}


async def crawl(per_front: int, days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    kb = discover.kernel.browsers.create(stealth=True)
    print(f"Kernel browser {kb.session_id} — watch live:\n  {kb.browser_live_view_url}\n")
    robots, stories = {}, []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(kb.cdp_ws_url)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            front_page = context.pages[0] if context.pages else await context.new_page()

            candidates = []
            for front, pattern in SOURCES.items():
                try:
                    try:
                        await front_page.goto(front, wait_until="domcontentloaded", timeout=30000)
                    except Exception:
                        await front_page.goto(front, wait_until="commit", timeout=45000)   # one slower retry
                    await front_page.wait_for_timeout(1500)
                    links = [u for u in story_links(await front_page.content(), front, pattern, per_front * 2)
                             if u not in candidates and discover.robots_ok(u, robots)][:per_front]
                    print(f"  {front.split('//')[1]:<36} {len(links)} stories")
                    candidates += links
                except Exception as e:
                    print(f"  {front.split('//')[1]:<36} ✗ {type(e).__name__}")

            sem = asyncio.Semaphore(discover.CONCURRENCY)

            async def fetch(url):
                async with sem:
                    page = await context.new_page()
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        s = extract_story(await page.content(), url)
                        recent = s["published"] and datetime.fromisoformat(s["published"]) >= cutoff
                        if recent and len(s["text"]) > 300:
                            stories.append(s)
                    except Exception:
                        pass
                    finally:
                        await page.close()

            await asyncio.gather(*(fetch(u) for u in candidates))
            await browser.close()
    finally:
        discover.kernel.browsers.delete_by_id(kb.session_id)
    return stories


def pick_diverse(ranked: list[dict], top: int, same_story: float = 0.70, per_work: int = 1,
                 min_similarity: float = 0.30) -> list[dict]:
    """Best matches first, but skip a story another outlet already covered and
    let no single book answer more than `per_work` stories — falling back to a
    story's next-best passage when its first choice is used up. Matches below
    `min_similarity` are dropped rather than forced."""
    picks, used = [], {}
    for s in ranked:
        if any(float(s["_vec"] @ p["_vec"]) > same_story for p in picks):
            continue
        options = [p for p in s["passages"]
                   if used.get(p["title"], 0) < per_work and p["similarity"] >= min_similarity]
        if not options:
            continue
        s["passages"] = options[:3]
        s["score"] = options[0]["similarity"]
        used[options[0]["title"]] = used.get(options[0]["title"], 0) + 1
        picks.append(s)
        if len(picks) == top:
            break
    return sorted(picks, key=lambda s: s["score"], reverse=True)


def explain(story: dict, passage: dict) -> str:
    import anthropic
    msg = anthropic.Anthropic().messages.create(
        model=CLAUDE_MODEL, max_tokens=120,
        messages=[{"role": "user", "content":
            f"News story: {story['title']}\n{story['description']}\n\n"
            f"Passage from {passage['title']} ({passage['author']}): {passage['passage']}\n\n"
            "In one sentence of plain English (under 35 words), say what this older passage "
            "anticipates or illuminates about the story. If the link is loose, say what it "
            "reframes rather than claiming prophecy. No preamble."}])
    return msg.content[0].text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--per-front", type=int, default=6)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--no-explain", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    stories = asyncio.run(crawl(args.per_front, args.days))
    print(f"\nRead {len(stories)} stories from the last {args.days} days in {time.time() - t0:.0f}s — matching…\n")
    if not stories:
        return

    taste = discover.load_taste()
    vecs = pm.embed([f"{s['title']}\n{s['description']}\n{s['text']}" for s in stories])
    for s, v in zip(stories, vecs):
        s["passages"] = pm.find_passages(v, k=5)
        s["cluster"] = pm.closest_clusters(v, taste)[0]
        s["score"] = s["passages"][0]["similarity"] if s["passages"] else 0
        s["_vec"] = v
        s.pop("text")

    ranked = sorted(stories, key=lambda s: s["score"], reverse=True)
    top = pick_diverse(ranked, args.top)
    for s in ranked:
        s.pop("_vec")
    if not args.no_explain:
        for s in top:
            s["why"] = explain(s, s["passages"][0])

    for i, s in enumerate(top, 1):
        p = s["passages"][0]
        print(f"{i:>2}. {s['title']}  ({s['site']}, {s['published'][:10]})")
        print(f"    ↳ {p['title']}{' — ' + p['author'] if p['author'] else ''}  ({p['similarity']:.3f})")
        print(f"      “{p['passage'][:220]}…”")
        if s.get("why"):
            print(f"      {s['why']}")
        print(f"    cluster: {s['cluster']['title']}\n")

    with open("news_oracle_results.json", "w") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "days": args.days,
                   "top": top, "all": ranked}, f, indent=2, ensure_ascii=False)
    print("Saved news_oracle_results.json")


if __name__ == "__main__":
    main()
