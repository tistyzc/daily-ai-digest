#!/usr/bin/env python3
"""Daily AI Digest — main pipeline entry point.

Orchestrates the full pipeline:
  Collect → Content Gate → Dedup → Quality Score → Relevance Filter
  → AI Summarize → Critic Validation → Render → Output

Usage:
    python src/main.py                          # Full pipeline
    python src/main.py --date 2026-07-31        # Specific date
    python src/main.py --skip-collect            # Use cached data
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.critics.bias import BiasCritic
from src.ai.critics.factual import FactualCritic
from src.ai.critics.privacy import PrivacyCritic
from src.ai.critics.safety import SafetyCritic
from src.ai.deepseek_client import DeepSeekClient
from src.ai.summarizer import BilingualSummarizer
from src.ai.topic_classifier import TopicClassifier
from src.collect.github_trending import GitHubTrendingFetcher
from src.collect.reddit_fetcher import RedditFetcher
from src.collect.rss_fetcher import RSSFetcher
from src.collect.web_search import WebSearchFetcher
from src.process.content_gate import ContentGate
from src.process.deduplicator import Deduplicator
from src.process.quality_scorer import QualityScorer
from src.render.renderer import DigestRenderer

logger = logging.getLogger(__name__)


def load_config(config_dir: str = "config") -> dict:
    """Load all YAML configuration files."""
    config_dir = Path(config_dir)

    def _load(name: str) -> dict:
        path = config_dir / name
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        logger.warning(f"Config file not found: {path}")
        return {}

    settings = _load("settings.yaml")
    sources_en = _load("sources_en.yaml")
    sources_cn = _load("sources_cn.yaml")

    return {
        "pipeline": settings.get("pipeline", {}),
        "deepseek": settings.get("deepseek", {}),
        "reddit": settings.get("reddit", {}),
        "github": settings.get("github", {}),
        "web_search": settings.get("web_search", {}),
        "output": settings.get("output", {}),
        "logging": settings.get("logging", {}),
        "sources_en": sources_en.get("sources", []),
        "sources_cn": sources_cn.get("sources", []),
    }


def setup_logging(config: dict) -> None:
    """Configure logging from config."""
    log_config = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format=log_config.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_collect(config: dict) -> list:
    """Step 1: Collect articles from all sources."""
    from src.collect.rss_fetcher import Article

    all_articles: list[Article] = []

    # RSS feeds (English + Chinese)
    all_sources = config["sources_en"] + config["sources_cn"]
    if all_sources:
        rss = RSSFetcher(all_sources)
        articles = rss.fetch_all()
        all_articles.extend(articles)
        logger.info(f"[Collect] RSS: {len(articles)} articles from {len(all_sources)} sources")

    # Reddit
    reddit_cfg = config["reddit"]
    if reddit_cfg.get("subreddits"):
        reddit = RedditFetcher(
            subreddits=reddit_cfg["subreddits"],
            post_limit=reddit_cfg.get("post_limit", 15),
            min_score=reddit_cfg.get("min_score", 10),
        )
        posts = reddit.fetch_all()
        articles = [p.to_article() for p in posts]
        all_articles.extend(articles)
        logger.info(f"[Collect] Reddit: {len(articles)} posts")

    # GitHub Trending
    gh_cfg = config["github"]
    if gh_cfg.get("languages"):
        gh = GitHubTrendingFetcher(
            languages=gh_cfg.get("languages"),
            min_stars_today=gh_cfg.get("min_stars_today", 20),
            max_repos=gh_cfg.get("max_repos", 10),
        )
        repos = gh.fetch_all()
        articles = [r.to_article() for r in repos]
        all_articles.extend(articles)
        logger.info(f"[Collect] GitHub Trending: {len(articles)} repos")

    # Web Search
    ws_cfg = config["web_search"]
    if ws_cfg.get("queries_en") or ws_cfg.get("queries_zh"):
        ws = WebSearchFetcher(
            queries_en=ws_cfg.get("queries_en", []),
            queries_zh=ws_cfg.get("queries_zh", []),
            max_results=ws_cfg.get("max_results_per_query", 5),
        )
        results = ws.fetch_all()
        articles = [r.to_article() for r in results]
        all_articles.extend(articles)
        logger.info(f"[Collect] Web Search: {len(articles)} results")

    logger.info(f"[Collect] Total: {len(all_articles)} articles from all sources")
    return all_articles


def step_process(articles: list, config: dict) -> list:
    """Steps 2-5: Content gate → Dedup → Quality score → Relevance filter."""
    pipe = config["pipeline"]

    # Step 2: Content gate
    gate = ContentGate(min_content_length=pipe.get("min_content_length", 600))
    articles = gate.filter(articles)

    # Step 3: Deduplicate
    dedup = Deduplicator(threshold=pipe.get("dedup_similarity_threshold", 85) / 100.0)
    articles = dedup.deduplicate(articles)

    # Step 4: Quality score
    scorer = QualityScorer(
        authority_boost=pipe.get("scoring", {}).get("authority_boost", 3),
        cross_source_boost=pipe.get("scoring", {}).get("cross_source_boost", 5),
        recency_boost=pipe.get("scoring", {}).get("recency_boost", 2),
        already_reported_penalty=pipe.get("scoring", {}).get("already_reported_penalty", -5),
        lookback_days=pipe.get("lookback_days", 2),
    )
    scored = scorer.score(articles)
    # Sort by score and cap
    max_articles = pipe.get("max_articles", 30)
    articles = [a for a, _ in scored[:max_articles]]

    return articles


def step_ai_process(articles: list, config: dict) -> tuple[list[dict], list[dict]]:
    """Steps 6-7: AI Summarize + Critic validation."""
    ds_cfg = config["deepseek"]
    model = ds_cfg.get("summary_model", "deepseek-chat")
    critic_model = ds_cfg.get("critic_model", "deepseek-chat")

    client = DeepSeekClient(default_model=model)

    # Step 6: Summarize
    summarizer = BilingualSummarizer(client)
    summaries = summarizer.summarize_batch(articles)

    # Step 7: Critic validation (4 layers)
    factual = FactualCritic(client, model=critic_model)
    safety = SafetyCritic(client, model=critic_model)
    bias = BiasCritic(client, model=critic_model)
    privacy = PrivacyCritic(client, model=critic_model)

    critical_flags: list[dict] = []

    for result in summaries:
        article = result["_article"]
        title = article.title
        content = article.content
        en_summary = result.get("en_summary", "")
        zh_summary = result.get("zh_summary", "")

        critic_issues: list[str] = []
        result["has_factual_issue"] = False
        result["has_bias_issue"] = False
        result["has_safety_issue"] = False
        result["has_privacy_issue"] = False
        result["critic_details"] = {}

        # Factual check
        fc = factual.check(title, content, en_summary)
        if not fc.get("is_factually_correct"):
            result["has_factual_issue"] = True
            result["critic_details"]["factual"] = fc.get("issues", [])
            critic_issues.append(f"Factual: {fc.get('issues', [])}")

        # Safety check
        sc = safety.check(title, content, result)
        if not sc.get("is_safe"):
            result["has_safety_issue"] = True
            result["critic_details"]["safety"] = sc.get("flags", [])
            critic_issues.append(f"Safety: {sc.get('flags', [])}")

        # Bias check
        bc = bias.check(title, en_summary, zh_summary)
        if bc.get("has_bias_issues"):
            result["has_bias_issue"] = True
            result["critic_details"]["bias"] = {"level": bc.get("bias_level", "mild"), "issues": bc.get("issues", [])}
            critic_issues.append(f"Bias ({bc.get('bias_level')}): {bc.get('issues', [])}")

        # Privacy check
        pc = privacy.check(title, content, result)
        if pc.get("has_pii"):
            result["has_privacy_issue"] = True
            result["critic_details"]["privacy"] = pc.get("pii_types", [])
            critic_issues.append(f"Privacy: {pc.get('pii_types', [])}")

        result["critic_flags"] = critic_issues

        if critic_issues:
            critical_flags.append(
                {
                    "article_title": title[:100],
                    "reason": "; ".join(critic_issues),
                }
            )
            logger.warning(f"[Critics] {len(critic_issues)} issues for '{title[:80]}'")

    logger.info(f"[Critics] {len(critical_flags)} articles flagged out of {len(summaries)}")
    return summaries, critical_flags


def step_classify(summaries: list[dict], config: dict) -> dict[str, list[dict]]:
    """Classify articles into topics."""
    client = DeepSeekClient(default_model=config["deepseek"].get("summary_model", "deepseek-chat"))
    classifier = TopicClassifier(client)

    articles = [s["_article"] for s in summaries]
    topic_buckets = classifier.classify(articles)

    # Reconstruct: map topic -> list of summary dicts
    result: dict[str, list[dict]] = {}
    for topic, arts in topic_buckets.items():
        result[topic] = []
        for art in arts:
            # Find matching summary
            for s in summaries:
                if s["_article"] is art:
                    result[topic].append(s)
                    break

    return result


def step_render(
    topics: dict[str, list[dict]],
    stats: dict,
    critical_flags: list[dict],
    date_str: str,
    config: dict,
) -> None:
    """Step 8: Render to HTML and write output files."""
    out_cfg = config["output"]
    docs_dir = Path(out_cfg.get("docs_dir", "docs"))
    archive_dir = Path(out_cfg.get("archive_dir", "docs/archive"))
    template_dir = out_cfg.get("template_dir", "src/render/templates")

    renderer = DigestRenderer(template_dir=template_dir)

    # Render index page
    index_html = renderer.render_index(
        date_str=date_str,
        topics=topics,
        stats=stats,
        critical_flags=critical_flags,
    )
    renderer.write_file(index_html, str(docs_dir / "index.html"))

    # Render archive page
    archives = build_archive_index(archive_dir)
    archive_html = renderer.render_archive(archives)
    archive_dir_path = docs_dir / "archive"
    archive_dir_path.mkdir(parents=True, exist_ok=True)
    renderer.write_file(archive_html, str(archive_dir_path / "index.html"))

    # Save archive for today
    today_archive = archive_dir_path / f"{date_str}.html"
    renderer.write_file(index_html, str(today_archive))

    logger.info(f"[Render] Output written to {docs_dir}/")


def build_archive_index(archive_dir: Path) -> list[dict]:
    """Build archive index by scanning existing archive files."""
    archives = []
    archive_path = Path("docs/archive")
    if archive_path.exists():
        for f in sorted(archive_path.glob("*.html"), reverse=True):
            if f.name == "index.html":
                continue
            date_str = f.stem
            archives.append(
                {
                    "date": date_str,
                    "title": f"Daily AI Digest — {date_str}",
                    "article_count": 0,  # Could parse HTML but non-trivial
                    "url": f"archive/{f.name}",
                }
            )
    return archives


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily AI Digest Pipeline")
    parser.add_argument("--date", type=str, default=None, help="Date string YYYY-MM-DD (default: today)")
    parser.add_argument("--skip-collect", action="store_true", help="Skip collection, use cached data")
    parser.add_argument("--config-dir", type=str, default="config", help="Config directory path")
    args = parser.parse_args()

    date_str = args.date or datetime.now(UTC).strftime("%Y-%m-%d")

    # Load config
    config = load_config(args.config_dir)
    setup_logging(config)

    logger.info("=" * 60)
    logger.info(f"Daily AI Digest — {date_str}")
    logger.info("=" * 60)

    # Step 1: Collect
    if not args.skip_collect:
        articles = step_collect(config)
        # Cache for debugging
        cache_path = Path("docs/.cache")
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_data = [
            {
                "title": a.title,
                "url": a.url,
                "source_name": a.source_name,
                "source_category": a.source_category,
                "authority_score": a.authority_score,
                "language": a.language,
                "published_at": a.published_at.isoformat() if a.published_at else None,
                "summary": a.summary[:500],
                "content_length": a.content_length,
            }
            for a in articles
        ]
        with open(cache_path / f"raw_{date_str}.json", "w") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info(f"[Cache] Saved raw data to docs/.cache/raw_{date_str}.json")
    else:
        logger.info("Skipping collection (--skip-collect)")
        articles = []

    if not articles:
        logger.warning("No articles collected — aborting")
        sys.exit(0)

    # Steps 2-5: Process
    articles = step_process(articles, config)
    logger.info(f"[Process] {len(articles)} articles after processing")

    if not articles:
        logger.warning("No articles after processing — aborting")
        sys.exit(0)

    # Steps 6-7: AI processing
    summaries, critical_flags = step_ai_process(articles, config)
    logger.info(f"[AI] {len(summaries)} articles summarized")

    # Classify
    topics = step_classify(summaries, config)
    logger.info(f"[Classify] {len(topics)} topics: {list(topics.keys())}")

    # Stats
    stats = {
        "total_articles": len(summaries),
        "sources_count": len({s["_article"].source_name for s in summaries}),
        "topics_count": len(topics),
        "languages": {
            "en": len([s for s in summaries if s["_article"].language == "en"]),
            "zh": len([s for s in summaries if s["_article"].language == "zh"]),
        },
        "critic_flags_count": len(critical_flags),
    }

    # Step 8: Render
    step_render(topics, stats, critical_flags, date_str, config)

    logger.info("=" * 60)
    logger.info(f"✅ Digest complete: {stats['total_articles']} articles, {stats['topics_count']} topics")
    logger.info(f"⚠️  {len(critical_flags)} articles with critic flags")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
