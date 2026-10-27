"""
kernel_ingest.py
================
Same pipeline as ingest_url.py, but pages are rendered in a Kernel cloud
browser instead of fetched with requests — so JS-rendered sites and pages
that block plain HTTP clients come through with their real text.

Usage:
  # Public pages (JS-rendered, bot-protected)
  python kernel_ingest.py https://example.com/docs --depth 1

  # Pages behind a login: sign in once by hand through Kernel's live view,
  # the session is saved to a Kernel profile, then reused on every ingest.
  python kernel_ingest.py --login substack https://substack.com/sign-in
  python kernel_ingest.py --profile substack https://substack.com/inbox

Requires KERNEL_API_KEY in .env, plus: pip install -r requirements-kernel.txt
"""

import argparse
import asyncio
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from kernel import ConflictError, Kernel
from playwright.async_api import async_playwright

load_dotenv()

import ingest_url as base  # reuses its chunking, embedding and ChromaDB insert

kernel = Kernel()


def html_to_text(html: str, url: str) -> tuple[str, str]:
    """Same cleaning as ingest_url.fetch_text, applied to rendered HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else urlparse(url).path
    for tag in soup(["script", "style", "nav", "noscript", "svg", "iframe"]):
        tag.decompose()
    # Only drop site chrome — many magazines wrap each article card's title
    # in its own <header>, which ingest_url's blanket removal would delete.
    for tag in soup(["header", "footer"]):
        if not tag.find_parent(["main", "article"]):
            tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return title, text.strip()


def create_browser(profile: str | None, save: bool = False):
    opts = {"stealth": True}
    if profile:
        opts["profile"] = {"name": profile, "save_changes": True} if save else {"name": profile}
    return kernel.browsers.create(**opts)


async def render_site(seeds: list[str], depth: int, profile: str | None) -> dict[str, tuple[str, str]]:
    """Crawl seeds in one Kernel browser; return {url: (title, text)}."""
    kb = create_browser(profile)
    print(f"Kernel browser {kb.session_id} — watch live: {kb.browser_live_view_url}")
    pages: dict[str, tuple[str, str]] = {}
    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(kb.cdp_ws_url)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = context.pages[0] if context.pages else await context.new_page()

            for seed in seeds:
                root = f"{urlparse(seed).scheme}://{urlparse(seed).netloc}"
                queue = [(seed, 0)]
                while queue:
                    url, d = queue.pop(0)
                    if url in pages:
                        continue
                    try:
                        await page.goto(url, wait_until="networkidle", timeout=30000)
                        html = await page.content()
                    except Exception as e:
                        print(f"  ✗ {url}: {e}")
                        continue
                    pages[url] = html_to_text(html, url)
                    if d < depth:
                        for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
                            href = urljoin(url, a["href"]).split("#")[0].rstrip("/")
                            if href.startswith(root) and href not in pages:
                                queue.append((href, d + 1))
            await browser.close()
    finally:
        kernel.browsers.delete_by_id(kb.session_id)
    return pages


def login(profile: str, url: str):
    """Open a profile-saving browser, let the user sign in via live view, then persist it."""
    try:
        kernel.profiles.create(name=profile)
    except ConflictError:
        pass
    kb = create_browser(profile, save=True)
    print(f"Open this live view, go to {url} and sign in:\n  {kb.browser_live_view_url}")
    input("Press Enter once you're signed in to save the session... ")
    # Deleting (not closing) the browser is what writes the profile.
    kernel.browsers.delete_by_id(kb.session_id)
    print(f"✓ Saved session to Kernel profile '{profile}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    parser.add_argument("--depth", type=int, default=0, help="Link-hops to follow (default 0)")
    parser.add_argument("--label", default="", help="Source label (default: domain name)")
    parser.add_argument("--profile", help="Reuse a saved Kernel profile (logged-in session)")
    parser.add_argument("--login", metavar="PROFILE", help="Sign in once and save to this profile")
    args = parser.parse_args()

    if args.login:
        login(args.login, args.urls[0])
        return

    pages = asyncio.run(render_site(args.urls, args.depth, args.profile))
    print(f"Rendered {len(pages)} pages")

    # ingest_url.ingest_url() calls fetch_text(url); serve it the rendered text instead.
    base.fetch_text = lambda url: pages[url]
    total = 0
    for url in pages:
        label = args.label or urlparse(url).netloc
        total += base.ingest_url(url, label)

    print(f"\n✓ Done — {total} new chunks added to '{base.COLLECTION_NAME}'")
    print(f"  Collection now has {base.collection.count()} total documents")


if __name__ == "__main__":
    main()
