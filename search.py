"""Web search across three providers -- Tavily, Exa, Linkup -- picked by
task complexity/speed needs, or forced by Ruk saying a provider name
explicitly. Falls back through the other two if the chosen one fails,
so one bad/expired key or a rate limit doesn't kill the whole search.
"""
import concurrent.futures
import os

from exa_py import Exa
from linkup import LinkupClient
from tavily import TavilyClient


def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Fastest, cheapest -- good default for normal/quick research."""
    client = TavilyClient(api_key=os.environ["TAVILY"].strip())
    data = client.search(query, search_depth="basic", max_results=max_results)
    return [{"title": r.get("title", ""), "url": r["url"], "content": r.get("content", "")} for r in data["results"]]


def _exa_search(query: str, max_results: int = 5) -> list[dict]:
    """Neural/semantic search -- better for conceptual, nuanced research.

    Bypasses exa_py's search_and_contents() wrapper on purpose. The
    pinned SDK (1.0.2) is many major versions behind Exa's live API
    (latest is 2.16.2 as of writing) -- the API now returns an 'image'
    field on results that this old SDK's Result dataclass doesn't
    define, so constructing it crashes with
    "Result.__init__() got an unexpected keyword argument 'image'"
    before we ever see a single result. Calling the same underlying
    request() the SDK's own search_and_contents() uses internally, but
    building a plain dict ourselves instead of that dataclass, sidesteps
    the version mismatch without a risky major-version bump (2.x changed
    enough that it isn't a safe drop-in replacement)."""
    client = Exa(api_key=os.environ["EXA"].strip())
    data = client.request("/search", {"query": query, "numResults": max_results, "contents": {"text": True}})
    return [
        {"title": r.get("title") or "", "url": r.get("url", ""), "content": (r.get("text") or "")[:1500]}
        for r in data.get("results", [])
    ]


def _linkup_search(query: str, max_results: int = 5) -> list[dict]:
    """Deep/structured search -- best for the most thorough research needs."""
    client = LinkupClient(api_key=os.environ["LINKUP"].strip())
    data = client.search(query=query, depth="deep", output_type="searchResults")
    results = data.get("results", []) if isinstance(data, dict) else getattr(data, "results", [])
    out = []
    for r in results[:max_results]:
        r = r if isinstance(r, dict) else r.model_dump()
        out.append({"title": r.get("name", ""), "url": r.get("url", ""), "content": (r.get("content") or "")[:1500]})
    return out


PROVIDERS = {"tavily": _tavily_search, "exa": _exa_search, "linkup": _linkup_search}

# ponytail: simple complexity -> provider mapping, not a learned/tunable
# router -- Tavily for quick/general lookups (fastest, cheapest), Exa for
# deeper conceptual research (neural search surfaces more relevant
# matches on nuanced queries), Linkup for the most thorough/structured
# research needs.
_COMPLEXITY_DEFAULT = {
    "simple": "tavily",
    "medium": "tavily",
    "complex": "exa",
    "very_complex": "linkup",
}


def search(query: str, provider: str | None = None, complexity: str = "simple") -> list[dict]:
    """provider: explicit choice ("tavily"/"exa"/"linkup") overrides the
    complexity-based default. Falls back through the other two providers
    if the chosen/default one fails, so one bad key or outage doesn't
    kill the whole search."""
    chosen = provider if provider in PROVIDERS else _COMPLEXITY_DEFAULT.get(complexity, "tavily")
    order = [chosen] + [p for p in PROVIDERS if p != chosen]
    errors = []
    for p in order:
        try:
            return PROVIDERS[p](query)
        except Exception as e:
            errors.append(f"{p}: {e}")
            continue
    raise RuntimeError("All search providers failed -- " + "; ".join(errors))


def search_all(query: str) -> dict[str, list[dict] | str]:
    """Actually calls Tavily, Exa, AND Linkup in parallel -- for when Ruk
    explicitly wants to see/compare all three separately (e.g. 'use all
    three search models'). This is the real version of that request --
    without it, main.py only ever called ONE provider per message and
    the model had nothing genuine to split three ways when asked, so it
    fabricated a fake per-provider breakdown to match the request's
    shape. Returns {"tavily": [...], "exa": "error: ...", ...} -- a
    failed provider's value is an error string, not silently dropped,
    so the model can honestly report 'linkup failed' instead of making
    something up to fill the gap."""
    out: dict[str, list[dict] | str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        future_to_provider = {pool.submit(fn, query): name for name, fn in PROVIDERS.items()}
        for future in concurrent.futures.as_completed(future_to_provider):
            name = future_to_provider[future]
            try:
                out[name] = future.result()
            except Exception as e:
                out[name] = f"error: {e}"
    return out


if __name__ == "__main__":
    results = search("what is Mem0", complexity="simple")
    assert results, "search returned nothing"
    assert all("url" in r and "content" in r for r in results)
    print(f"search.py: got {len(results)} results ->", results[0]["title"])
