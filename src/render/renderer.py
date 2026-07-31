"""Jinja2 renderer — generates the static HTML site."""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown import markdown

logger = logging.getLogger(__name__)


class DigestRenderer:
    """Renders the daily digest into static HTML pages."""

    def __init__(self, template_dir: str) -> None:
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )
        # Custom filters
        self.env.filters["format_date"] = self._format_date
        self.env.filters["markdown"] = self._render_markdown

    def render_index(
        self,
        date_str: str,
        topics: dict[str, list[dict]],
        stats: dict,
        critical_flags: list[dict],
    ) -> str:
        """Render the main index page (latest digest).

        Args:
            date_str: ISO date string like "2026-07-31".
            topics: Dict of topic_name -> list of article summary dicts.
            stats: Dict with total_articles, sources_count, etc.
            critical_flags: List of critic findings that need attention.
        """
        template = self.env.get_template("index.html.j2")
        return template.render(
            date=date_str,
            topics=topics,
            stats=stats,
            critical_flags=critical_flags,
            generated_at=datetime.now(UTC).isoformat(),
        )

    def render_archive(
        self,
        archives: list[dict],
    ) -> str:
        """Render the archive page listing all past digests.

        Args:
            archives: List of {date: str, title: str, article_count: int, url: str}.
        """
        template = self.env.get_template("archive.html.j2")
        return template.render(archives=archives)

    def write_file(self, content: str, path: str) -> None:
        """Write rendered HTML to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        logger.info(f"Written: {path}")

    @staticmethod
    def _format_date(iso_str: str, fmt: str = "%B %d, %Y") -> str:
        try:
            dt = datetime.fromisoformat(iso_str)
            return dt.strftime(fmt)
        except (ValueError, TypeError):
            return iso_str

    @staticmethod
    def _render_markdown(text: str) -> str:
        return markdown(text, extensions=["fenced_code", "tables"])
