"""Web search across three providers -- Tavily, Exa, Linkup -- picked by
task complexity/speed needs, or forced by Ruk saying a provider name
explicitly. Falls back through the other two if the chosen one fails,
so one bad/expired key or a rate limit doesn't kill the whole search.
"""
import os

from exa_py import Exa
from linkup import LinkupClient
from tavily import TavilyClient


def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Fastest, cheapest -- good default for normal/quick research."""
    client = TavilyClient(api_key=os.environ["TAVILY"])
    data = client.search(query, search_depth="basic", max_results=max_results)
    return [{"title": r.get("title", ""), "url": r["url"], "content": r.get("content", "")} for r in data["results"]]


def _exa_search(query: str, max_results: int = 5) -> list[dict]:
    """Neural/semantic search -- better for conceptual, nuanced research."""
    client = Exa(api_key=os.environ["EXA"])
    data = client.search_and_contents(query, num_results=max_results, text=True)
    return [
        {"title": r.title or "", "url": r.url, "content": (r.text or "")[:1500]}
        for r in data.results
    ]


def _linkup_search(query: str, max_results: int = 5) -> list[dict]:
    """Deep/structured search -- best for the most thorough research needs."""
    client = LinkupClient(api_key=os.environ["LINKUP"])
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
    last_err = None
    for p in order:
        try:
            return PROVIDERS[p](query)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All search providers failed: {last_err}")


if __name__ == "__main__":
    results = search("what is Mem0", complexity="simple")
    assert results, "search returned nothing"
    assert all("url" in r and "content" in r for r in results)
    print(f"search.py: got {len(results)} results ->", results[0]["title"])
