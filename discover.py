"""
discover.py
===========
Reading recommendations from the live web, ranked against the library's own
taste. Each Leiden community summary in `literary_communities` is a taste
vector; a Kernel browser crawls literary magazines, and every article is
scored against those vectors and explained by its closest community.

Usage:
  python discover.py                     # crawl all sites, print top picks
  python discover.py --per-site 6 --top 15
  python discover.py --sites https://aeon.co/essays

Writes discover_results.json. Requires KERNEL_API_KEY and OPENAI_API_KEY.
"""

import argparse
import asyncio
import json
import os
import re
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import chromadb
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from kernel import Kernel
from openai import OpenAI
from playwright.async_api import async_playwright

load_dotenv()

SITES = [
    "https://www.theparisreview.org/blog/",
    "https://www.nplusonemag.com/online-only/",
    "https://thepointmag.com/",
    "https://aeon.co/essays",
    "https://www.laphamsquarterly.org/roundtable",
    "https://granta.com/",
    "https://electricliterature.com/",
    "https://www.themarginalian.org/",
    "https://www.lrb.co.uk/blog",
]

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMS  = 3072
CLAUDE_MODEL    = "claude-sonnet-4-6"   # same model graphrag_app.py uses
LABELS_PATH     = "taste_labels.json"
EXCERPT_CHARS   = 1500
CONCURRENCY     = 4
USER_AGENT      = "Mozilla/5.0 (compatible; raglib-discover/1.0)"
NAV_WORDS = re.compile(
    r"\b(subscribe|sign in|log in|newsletter|donate|shop|about|contact|privacy|"
    r"terms|advertis|masthead|careers|events|podcast|see more|read more|archive)\b", re.I)
SECTION_PATHS = re.compile(
    r"/(tags?|subjects?|category|categories|products?|shop|store|authors?|contributors?|"
    r"donate|support|events?|about|subscribe|newsletter|issues?|editions?|page)(/|$)", re.I)

kernel = Kernel()
oai    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Taste profile ──────────────────────────────────────────────────────────

def load_taste() -> list[dict]:
    """Literary community summaries as taste vectors (skips nautical/code clusters)."""
    col = chromadb.PersistentClient(path="chroma_db").get_collection("literary_communities")
    r = col.get(include=["metadatas", "documents", "embeddings"])
    taste = []
    for id_, meta, doc, emb in zip(r["ids"], r["metadatas"], r["documents"], r["embeddings"]):
        genres = meta.get("genres", "")
        if "nautical" in genres or not meta.get("themes") or meta.get("themes") == "language_and_narrative":
            continue
        title = doc.splitlines()[0].lstrip("# ").strip()
        works = [re.sub(r"\s+\d{3}\b.*|\.txt$", "", w) for w in json.loads(meta.get("works", "[]"))]
        taste.append({
            "id": id_,
            "title": title,
            "summary": doc,
            "works": list(dict.fromkeys(works))[:3],
            "vec": np.asarray(emb, dtype=np.float32) / np.linalg.norm(emb),
        })
    name_generic_clusters(taste)
    return taste


def name_generic_clusters(taste: list[dict]):
    """Many summaries are headed "The Cluster's Conceptual Thread"; ask Claude
    for a real name once and cache it in taste_labels.json."""
    try:
        labels = json.load(open(LABELS_PATH))
    except FileNotFoundError:
        labels = {}
    generic = [t for t in taste if t["title"].startswith("The Cluster") and t["id"] not in labels]
    if generic:
        import anthropic
        client = anthropic.Anthropic()
        for t in generic:
            msg = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=30,
                messages=[{"role": "user", "content":
                    "Name this cluster of books in 3-7 words, in the style of "
                    "'The Exhaustion of Heroic Consciousness'. Reply with the name only.\n\n"
                    + t["summary"][:2500]}])
            labels[t["id"]] = msg.content[0].text.strip().strip('"*# ')
        json.dump(labels, open(LABELS_PATH, "w"), indent=2)
    for t in taste:
        t["title"] = labels.get(t["id"], t["title"])
        t.pop("summary")


# ── Crawl ──────────────────────────────────────────────────────────────────

def robots_for(url: str, cache: dict) -> RobotFileParser | None:
    # Fetched with requests, not RobotFileParser.read(): many sites 403 Python's
    # default urllib agent, which the parser treats as "disallow everything".
    root = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    if root not in cache:
        try:
            r = requests.get(f"{root}/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=10)
            rp = RobotFileParser()
            rp.parse(r.text.splitlines() if r.ok else [])   # no robots.txt → no restrictions
        except Exception:
            rp = None
        cache[root] = rp
    return cache[root]


def robots_ok(url: str, cache: dict) -> bool:
    rp = robots_for(url, cache)
    return rp is None or rp.can_fetch(USER_AGENT, url)


def crawl_delay(url: str, cache: dict) -> float:
    rp = robots_for(url, cache)
    return float((rp and rp.crawl_delay(USER_AGENT)) or 0)


def article_links(html: str, index_url: str, limit: int) -> list[str]:
    """Headline-looking links on the same domain, deepest/sluggiest first."""
    host = urlparse(index_url).netloc
    seen, links = set(), []
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = urljoin(index_url, a["href"]).split("#")[0].split("?")[0].rstrip("/")
        path = urlparse(href).path.rstrip("/")
        segments = [s for s in path.split("/") if s]
        slug = segments[-1] if segments else ""
        # Article slugs are hyphenated: either nested (/criticism/death-is-not-the-end)
        # or long at the top level (/the-coziest-place-on-the-moon).
        is_article = "-" in slug and (len(segments) >= 2 or slug.count("-") >= 3)
        if (urlparse(href).netloc != host or href in seen or href.rstrip("/") == index_url.rstrip("/")
                or not (20 <= len(text) <= 200) or NAV_WORDS.search(text)
                or SECTION_PATHS.search(path) or not is_article):
            continue
        seen.add(href)
        links.append(href)
    return links[:limit]


def extract_article(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    og = lambda p: (soup.find("meta", property=p) or {}).get("content", "")
    title = og("og:title") or (soup.title.string.strip() if soup.title and soup.title.string else url)
    desc  = og("og:description") or (soup.find("meta", attrs={"name": "description"}) or {}).get("content", "")
    for tag in soup(["script", "style", "nav", "noscript", "svg", "iframe", "aside", "form"]):
        tag.decompose()
    body = soup.find("article") or soup.find("main") or soup.body or soup
    paras = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    text = " ".join(p for p in paras if len(p) > 60)
    return {"url": url, "site": urlparse(url).netloc.removeprefix("www."),
            "title": title.strip(), "description": desc.strip(), "excerpt": text[:EXCERPT_CHARS]}


async def crawl(sites: list[str], per_site: int) -> list[dict]:
    kb = kernel.browsers.create(stealth=True)
    print(f"Kernel browser {kb.session_id} — watch live:\n  {kb.browser_live_view_url}\n")
    robots, articles = {}, []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(kb.cdp_ws_url)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            index_page = context.pages[0] if context.pages else await context.new_page()

            # 1. Index pages → candidate article URLs
            candidates = []
            for site in sites:
                try:
                    await index_page.goto(site, wait_until="domcontentloaded", timeout=30000)
                    await index_page.wait_for_timeout(1500)
                    links = [u for u in article_links(await index_page.content(), site, per_site * 2)
                             if robots_ok(u, robots)][:per_site]
                    print(f"  {urlparse(site).netloc:<28} {len(links)} articles")
                    candidates += links
                except Exception as e:
                    print(f"  {urlparse(site).netloc:<28} ✗ {type(e).__name__}")

            # 2. Articles, a few tabs at a time — but one at a time per site that
            #    sets a robots.txt Crawl-delay, waiting that long between requests.
            sem = asyncio.Semaphore(CONCURRENCY)
            host_locks: dict[str, asyncio.Lock] = {}

            async def fetch(url):
                delay = crawl_delay(url, robots)
                lock = host_locks.setdefault(urlparse(url).netloc, asyncio.Lock()) if delay else None
                if lock:
                    await lock.acquire()
                try:
                    async with sem:
                        page = await context.new_page()
                        try:
                            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                            art = extract_article(await page.content(), url)
                            is_listing = re.search(r"\barchives?\b", art["title"], re.I)
                            if not is_listing and len(art["excerpt"]) + len(art["description"]) > 200:
                                articles.append(art)
                        except Exception:
                            pass
                        finally:
                            await page.close()
                    if delay:
                        await asyncio.sleep(delay)
                finally:
                    if lock:
                        lock.release()

            await asyncio.gather(*(fetch(u) for u in candidates))
            await browser.close()
    finally:
        kernel.browsers.delete_by_id(kb.session_id)
    return articles


# ── Rank ───────────────────────────────────────────────────────────────────

def embed(texts: list[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), 50):
        resp = oai.embeddings.create(input=texts[i:i + 50], model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMS)
        vecs += [d.embedding for d in resp.data]
    m = np.asarray(vecs, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def rank(articles: list[dict], taste: list[dict]) -> list[dict]:
    texts = [f"{a['title']}\n{a['description']}\n{a['excerpt']}" for a in articles]
    sims = embed(texts) @ np.stack([t["vec"] for t in taste]).T      # (articles, communities)
    for a, row in zip(articles, sims):
        best = int(row.argmax())
        a["score"] = round(float(row[best]), 4)
        a["match"] = {k: taste[best][k] for k in ("id", "title", "works")}
        a.pop("excerpt")
    return sorted(articles, key=lambda a: a["score"], reverse=True)


def pick_top(ranked: list[dict], top: int, max_per_site: int = 3) -> list[dict]:
    picks, per_site = [], {}
    for a in ranked:
        if per_site.get(a["site"], 0) < max_per_site:
            picks.append(a)
            per_site[a["site"]] = per_site.get(a["site"], 0) + 1
        if len(picks) == top:
            break
    return picks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=SITES)
    parser.add_argument("--per-site", type=int, default=8)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    taste = load_taste()
    print(f"Taste profile: {len(taste)} literary communities\n")

    t0 = time.time()
    articles = asyncio.run(crawl(args.sites, args.per_site))
    print(f"\nRead {len(articles)} articles in {time.time() - t0:.0f}s — ranking…\n")
    if not articles:
        return

    ranked = rank(articles, taste)
    picks = pick_top(ranked, args.top)

    for i, a in enumerate(picks, 1):
        print(f"{i:>2}. {a['title']}  ({a['site']}, {a['score']:.3f})")
        print(f"    ↳ closest to “{a['match']['title']}” — {', '.join(a['match']['works'])}")
        print(f"    {a['url']}\n")

    with open("discover_results.json", "w") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "taste": [
            {k: t[k] for k in ("id", "title", "works")} for t in taste],
            "picks": picks, "all": ranked}, f, indent=2, ensure_ascii=False)
    print("Saved discover_results.json")


if __name__ == "__main__":
    main()
