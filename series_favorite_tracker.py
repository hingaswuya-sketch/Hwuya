#!/usr/bin/env python3
"""Series Favorite Tracker.

A small command-line tool for keeping track of your favorite TV series:
add series you're watching, rate them, mark how many episodes you've seen,
tag them as favorites, and see quick statistics.

Data is stored as JSON in ``~/.series_favorite_tracker.json`` by default
(override with the ``--data-file`` option or the ``SERIES_TRACKER_FILE``
environment variable).

Examples
--------
    python series_favorite_tracker.py add "Breaking Bad" --total 62 --rating 10 --favorite
    python series_favorite_tracker.py watch "Breaking Bad" 5
    python series_favorite_tracker.py rate "Breaking Bad" 9
    python series_favorite_tracker.py favorite "Breaking Bad"
    python series_favorite_tracker.py list --favorites
    python series_favorite_tracker.py stats
    python series_favorite_tracker.py remove "Breaking Bad"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_DATA_FILE = Path.home() / ".series_favorite_tracker.json"
STATUSES = ("watching", "completed", "paused", "dropped", "plan-to-watch")


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Series:
    """A single tracked TV series."""

    title: str
    status: str = "watching"
    rating: Optional[int] = None
    favorite: bool = False
    episodes_watched: int = 0
    total_episodes: Optional[int] = None
    notes: str = ""
    added_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def touch(self) -> None:
        """Update the ``updated_at`` timestamp."""
        self.updated_at = _now_iso()

    @property
    def progress(self) -> Optional[float]:
        """Fraction watched (0.0-1.0), or ``None`` if the total is unknown."""
        if not self.total_episodes:
            return None
        return min(self.episodes_watched / self.total_episodes, 1.0)

    def progress_str(self) -> str:
        if self.total_episodes:
            pct = (self.progress or 0) * 100
            return f"{self.episodes_watched}/{self.total_episodes} ({pct:.0f}%)"
        return f"{self.episodes_watched} ep"

    @classmethod
    def from_dict(cls, data: dict) -> "Series":
        known = {f: data.get(f) for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        # Drop keys whose value is None so dataclass defaults apply.
        known = {k: v for k, v in known.items() if v is not None}
        return cls(**known)


class Tracker:
    """Loads, mutates, and saves a collection of :class:`Series`."""

    def __init__(self, data_file: Path):
        self.data_file = data_file
        self.series: dict[str, Series] = {}
        self.load()

    # ---- persistence -------------------------------------------------
    def load(self) -> None:
        if not self.data_file.exists():
            return
        try:
            raw = json.loads(self.data_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SystemExit(f"error: could not read data file {self.data_file}: {exc}")
        for item in raw.get("series", []):
            s = Series.from_dict(item)
            self.series[s.title.lower()] = s

    def save(self) -> None:
        payload = {"series": [asdict(s) for s in self.sorted()]}
        tmp = self.data_file.with_suffix(self.data_file.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.data_file)

    # ---- helpers -----------------------------------------------------
    def sorted(self) -> list[Series]:
        return sorted(self.series.values(), key=lambda s: s.title.lower())

    def get(self, title: str) -> Series:
        s = self.series.get(title.lower())
        if s is None:
            raise SystemExit(f"error: no series titled {title!r} (try 'list')")
        return s

    def add(self, series: Series) -> Series:
        key = series.title.lower()
        if key in self.series:
            raise SystemExit(f"error: {series.title!r} is already tracked")
        self.series[key] = series
        return series

    def remove(self, title: str) -> Series:
        s = self.get(title)
        del self.series[title.lower()]
        return s


# ---- validation ------------------------------------------------------
def _validate_rating(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if not 1 <= value <= 10:
        raise SystemExit("error: rating must be between 1 and 10")
    return value


# ---- command handlers ------------------------------------------------
def cmd_add(tracker: Tracker, args: argparse.Namespace) -> None:
    if args.status not in STATUSES:
        raise SystemExit(f"error: status must be one of {', '.join(STATUSES)}")
    series = Series(
        title=args.title,
        status=args.status,
        rating=_validate_rating(args.rating),
        favorite=args.favorite,
        total_episodes=args.total,
        episodes_watched=args.watched,
        notes=args.notes or "",
    )
    tracker.add(series)
    tracker.save()
    print(f"Added {series.title!r}.")


def cmd_remove(tracker: Tracker, args: argparse.Namespace) -> None:
    series = tracker.remove(args.title)
    tracker.save()
    print(f"Removed {series.title!r}.")


def cmd_rate(tracker: Tracker, args: argparse.Namespace) -> None:
    series = tracker.get(args.title)
    series.rating = _validate_rating(args.rating)
    series.touch()
    tracker.save()
    print(f"Rated {series.title!r} {series.rating}/10.")


def cmd_favorite(tracker: Tracker, args: argparse.Namespace) -> None:
    series = tracker.get(args.title)
    series.favorite = not args.unset
    series.touch()
    tracker.save()
    state = "a favorite" if series.favorite else "no longer a favorite"
    print(f"{series.title!r} is now {state}.")


def cmd_watch(tracker: Tracker, args: argparse.Namespace) -> None:
    series = tracker.get(args.title)
    if args.set is not None:
        series.episodes_watched = max(args.set, 0)
    else:
        series.episodes_watched = max(series.episodes_watched + args.count, 0)
    if series.total_episodes:
        series.episodes_watched = min(series.episodes_watched, series.total_episodes)
        if series.episodes_watched == series.total_episodes:
            series.status = "completed"
    series.touch()
    tracker.save()
    print(f"{series.title!r}: {series.progress_str()} [{series.status}]")


def cmd_status(tracker: Tracker, args: argparse.Namespace) -> None:
    if args.status not in STATUSES:
        raise SystemExit(f"error: status must be one of {', '.join(STATUSES)}")
    series = tracker.get(args.title)
    series.status = args.status
    series.touch()
    tracker.save()
    print(f"{series.title!r} is now {series.status}.")


def _matches(series: Series, args: argparse.Namespace) -> bool:
    if args.favorites and not series.favorite:
        return False
    if args.status and series.status != args.status:
        return False
    return True


def cmd_list(tracker: Tracker, args: argparse.Namespace) -> None:
    rows = [s for s in tracker.sorted() if _matches(s, args)]
    if not rows:
        print("No series match." if (args.favorites or args.status) else "No series tracked yet.")
        return
    print(_format_table(rows))


def cmd_show(tracker: Tracker, args: argparse.Namespace) -> None:
    s = tracker.get(args.title)
    star = "★" if s.favorite else " "
    rating = f"{s.rating}/10" if s.rating is not None else "unrated"
    lines = [
        f"{star} {s.title}",
        f"  status:   {s.status}",
        f"  rating:   {rating}",
        f"  progress: {s.progress_str()}",
        f"  added:    {s.added_at}",
        f"  updated:  {s.updated_at}",
    ]
    if s.notes:
        lines.append(f"  notes:    {s.notes}")
    print("\n".join(lines))


def cmd_stats(tracker: Tracker, args: argparse.Namespace) -> None:
    items = tracker.sorted()
    if not items:
        print("No series tracked yet.")
        return
    rated = [s.rating for s in items if s.rating is not None]
    by_status: dict[str, int] = {}
    for s in items:
        by_status[s.status] = by_status.get(s.status, 0) + 1
    total_eps = sum(s.episodes_watched for s in items)

    print(f"Total series:      {len(items)}")
    print(f"Favorites:         {sum(1 for s in items if s.favorite)}")
    print(f"Episodes watched:  {total_eps}")
    if rated:
        print(f"Average rating:    {sum(rated) / len(rated):.1f}/10  (over {len(rated)})")
    print("By status:")
    for status in STATUSES:
        if by_status.get(status):
            print(f"  {status:<14} {by_status[status]}")

    top = sorted(
        (s for s in items if s.rating is not None),
        key=lambda s: s.rating,  # type: ignore[arg-type]
        reverse=True,
    )[:3]
    if top:
        print("Top rated:")
        for s in top:
            print(f"  {s.rating:>2}/10  {s.title}")


def _format_table(rows: Iterable[Series]) -> str:
    rows = list(rows)
    headers = ("", "Title", "Status", "Rating", "Progress")
    table = []
    for s in rows:
        table.append(
            (
                "★" if s.favorite else "",
                s.title,
                s.status,
                f"{s.rating}/10" if s.rating is not None else "-",
                s.progress_str(),
            )
        )
    widths = [len(h) for h in headers]
    for r in table:
        widths = [max(w, len(c)) for w, c in zip(widths, r)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    out.extend(fmt.format(*r) for r in table)
    return "\n".join(out)


# ---- argument parsing ------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="series_favorite_tracker",
        description="Track your favorite TV series from the command line.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Path to the JSON data file (default: ~/.series_favorite_tracker.json).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a new series.")
    p_add.add_argument("title")
    p_add.add_argument("--status", default="watching", help=f"One of: {', '.join(STATUSES)}.")
    p_add.add_argument("--rating", type=int, help="Rating from 1 to 10.")
    p_add.add_argument("--favorite", action="store_true", help="Mark as a favorite.")
    p_add.add_argument("--total", type=int, help="Total number of episodes.")
    p_add.add_argument("--watched", type=int, default=0, help="Episodes already watched.")
    p_add.add_argument("--notes", help="Free-form notes.")
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser("remove", help="Remove a series.")
    p_rm.add_argument("title")
    p_rm.set_defaults(func=cmd_remove)

    p_rate = sub.add_parser("rate", help="Set a rating (1-10).")
    p_rate.add_argument("title")
    p_rate.add_argument("rating", type=int)
    p_rate.set_defaults(func=cmd_rate)

    p_fav = sub.add_parser("favorite", help="Mark (or unmark) a series as a favorite.")
    p_fav.add_argument("title")
    p_fav.add_argument("--unset", action="store_true", help="Remove the favorite flag.")
    p_fav.set_defaults(func=cmd_favorite)

    p_watch = sub.add_parser("watch", help="Record watched episodes.")
    p_watch.add_argument("title")
    p_watch.add_argument("count", type=int, nargs="?", default=1, help="Episodes to add (default 1).")
    p_watch.add_argument("--set", type=int, help="Set the watched count to an absolute value.")
    p_watch.set_defaults(func=cmd_watch)

    p_status = sub.add_parser("status", help="Change a series' status.")
    p_status.add_argument("title")
    p_status.add_argument("status", help=f"One of: {', '.join(STATUSES)}.")
    p_status.set_defaults(func=cmd_status)

    p_list = sub.add_parser("list", help="List tracked series.")
    p_list.add_argument("--favorites", action="store_true", help="Only favorites.")
    p_list.add_argument("--status", help="Only series with this status.")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show details for one series.")
    p_show.add_argument("title")
    p_show.set_defaults(func=cmd_show)

    p_stats = sub.add_parser("stats", help="Show summary statistics.")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def resolve_data_file(cli_value: Optional[Path]) -> Path:
    if cli_value is not None:
        return cli_value
    env = os.environ.get("SERIES_TRACKER_FILE")
    if env:
        return Path(env)
    return DEFAULT_DATA_FILE


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tracker = Tracker(resolve_data_file(args.data_file))
    args.func(tracker, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
