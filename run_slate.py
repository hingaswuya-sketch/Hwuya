#!/usr/bin/env python3
"""
run_slate.py

Batch runner for the CFB scoring engine: reads a slate of games from JSON,
runs each through ``cfb_model.project_game``, and prints a ranked card.

The engine itself (cfb_model.py) scores one game at a time. This is the
harness that walks a week's worth of them and sorts the output by edge so
the playable games surface at the top.

Usage
-----
    python run_slate.py slate_example.json
    python run_slate.py slate.json --min-edge 0.06
    python run_slate.py slate.json --all        # include no-plays

Slate format
------------
A JSON object with an optional ``season``/``week`` label and a ``games``
list. Each game supplies the same fields the dataclasses in cfb_model.py
declare::

    {
      "season": 2026,
      "week": 1,
      "synthetic": true,
      "games": [
        {
          "home": {"name": "...", "efficiency": {...}, "talent": {...},
                   "situational": {...}},
          "away": {...},
          "context": {...},
          "pinnacle_home_odds": -280,
          "pinnacle_away_odds": 230
        }
      ]
    }

Set ``"synthetic": true`` on any slate whose numbers are placeholders. The
runner prints a loud banner so illustrative output is never mistaken for a
real card.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from cfb_model import (
    GameContext,
    MIN_EDGE_THRESHOLD,
    ProjectionResult,
    TeamEfficiency,
    TeamInputs,
    TeamSituational,
    TeamTalent,
    project_game,
)


def _build(cls, data: dict, where: str):
    """Instantiate a dataclass from a dict, reporting missing/extra keys."""
    expected = set(cls.__dataclass_fields__)
    got = set(data)
    missing = expected - got
    extra = got - expected
    if missing:
        raise SystemExit(f"error: {where}: missing field(s): {', '.join(sorted(missing))}")
    if extra:
        raise SystemExit(f"error: {where}: unknown field(s): {', '.join(sorted(extra))}")
    return cls(**data)


def parse_team(data: dict, where: str) -> TeamInputs:
    for key in ("name", "efficiency", "talent", "situational"):
        if key not in data:
            raise SystemExit(f"error: {where}: missing '{key}'")
    return TeamInputs(
        name=data["name"],
        efficiency=_build(TeamEfficiency, data["efficiency"], f"{where}.efficiency"),
        talent=_build(TeamTalent, data["talent"], f"{where}.talent"),
        situational=_build(TeamSituational, data["situational"], f"{where}.situational"),
    )


def load_slate(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"error: could not read slate {path}: {exc}")
    if "games" not in raw:
        raise SystemExit(f"error: {path}: slate has no 'games' list")
    return raw


def run_slate(slate: dict) -> list[ProjectionResult]:
    results = []
    for i, game in enumerate(slate["games"]):
        where = f"games[{i}]"
        home = parse_team(game.get("home", {}), f"{where}.home")
        away = parse_team(game.get("away", {}), f"{where}.away")
        context = _build(GameContext, game.get("context", {}), f"{where}.context")
        for key in ("pinnacle_home_odds", "pinnacle_away_odds"):
            if key not in game:
                raise SystemExit(f"error: {where}: missing '{key}'")
        results.append(
            project_game(
                home, away, context,
                pinnacle_home_odds=game["pinnacle_home_odds"],
                pinnacle_away_odds=game["pinnacle_away_odds"],
            )
        )
    return results


def format_card(results: list[ProjectionResult], min_edge: float, show_all: bool) -> str:
    ranked = sorted(results, key=lambda r: abs(r.edge), reverse=True)
    rows = []
    for r in ranked:
        playable = abs(r.edge) >= min_edge and r.recommended_stake_fraction > 0
        if not playable and not show_all:
            continue
        side = r.home_team if r.edge > 0 else r.away_team
        rows.append((
            f"{r.away_team} @ {r.home_team}",
            f"{r.projected_margin:+.1f}",
            f"{r.model_home_win_prob:.3f}",
            f"{r.market_home_win_prob:.3f}",
            f"{r.edge*100:+.2f}%",
            f"{r.confidence:.2f}",
            f"{r.recommended_stake_fraction*100:.2f}%",
            side if playable else "-- no play",
        ))

    if not rows:
        return ("No games cleared the edge/confidence filter.\n"
                "Re-run with --all to see every game and why it was passed.")

    headers = ("Matchup", "Margin", "Model", "Market", "Edge", "Conf", "Stake", "Side")
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    out.extend(fmt.format(*row) for row in rows)
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_slate",
        description="Run a slate of CFB games through the scoring engine.",
    )
    parser.add_argument("slate", type=Path, help="Path to the slate JSON file.")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE_THRESHOLD,
                        help=f"Edge threshold to call a game playable (default {MIN_EDGE_THRESHOLD}).")
    parser.add_argument("--all", action="store_true", dest="show_all",
                        help="Show every game, including no-plays.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the engine's per-game INFO logging.")
    args = parser.parse_args(argv)

    if args.quiet:
        logging.getLogger("cfb_model").setLevel(logging.WARNING)

    slate = load_slate(args.slate)
    results = run_slate(slate)

    label = []
    if slate.get("season"):
        label.append(str(slate["season"]))
    if slate.get("week"):
        label.append(f"Week {slate['week']}")
    header = " ".join(label) or args.slate.name

    print()
    if slate.get("synthetic"):
        print("!" * 72)
        print("!! SYNTHETIC SLATE — placeholder inputs, NOT real team data or lines.")
        print("!! Output is for wiring/validation only. Do not bet this card.")
        print("!" * 72)
    print(f"\n{header}  —  {len(results)} game(s)\n")
    print(format_card(results, args.min_edge, args.show_all))

    played = sum(1 for r in results
                 if abs(r.edge) >= args.min_edge and r.recommended_stake_fraction > 0)
    print(f"\n{played} play(s) / {len(results)} game(s) at min edge {args.min_edge:.3f}.")
    if played == 0 and results:
        worst = max(r.confidence for r in results)
        if worst == 0.0:
            print("NOTE: every game scored confidence 0.00 — with no plays logged and no\n"
                  "      common opponents, confidence_score() floors at zero and forces\n"
                  "      every stake to 0 regardless of edge. This is the model working\n"
                  "      as designed, not a data problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
