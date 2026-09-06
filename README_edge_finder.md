# Daily Edge Finder

Finds college-football spread edges the way the reference card did: model line
(SP+) vs. the market line, converted to a cover-probability edge, calibrated,
then gated by a single **raw-overshoot cap** and ranked strongest → weakest.

## The method (what the card was doing)

1. **Model line** — from 2026 final SP+ ratings:
   `home_margin = (home_SP+ − away_SP+) / 16.2 + 3` (home-field).
   You can also feed `model_margin` directly per game.
2. **Market line** — the book's home spread and price (DraftKings via ESPN).
3. **Raw edge** — the model's cover probability minus the price's break-even:
   `edge% = P(side covers) − break_even(price)`, using a normal margin model
   with `sigma` (SD of actual-margin-minus-spread, ~18.7 pts for CFB).
4. **Calibrated edge** — `raw × 0.80`, shrinking the edge toward the market.
5. **Gate & rank** — drop already-played games, keep games whose **raw** edge is
   `≤ cap`, rank high → low, keep the top `N` (default 5). The old 3–5% band is
   gone; the cap is the only gate.

**The cap is now 10%, not 9%.** A game like TULN @ DUKE (raw +9.2%) used to sit
*just over* the 9% cap and get flagged `NOT PLAYED`; at 10% it lands in range and
becomes a live, ranked play automatically.

Anything still **over** the cap is reported as a `FLAG` — the model disagrees so
hard with the market that it's more likely stale data than a real edge.

## Usage

```bash
# Ranked plays for a slate
python edge_finder.py scan --data games.json

# JSON out (for a notifier / spreadsheet / bot)
python edge_finder.py scan --data games.json --json

# Every scored game, not just the top plays
python edge_finder.py scan --data games.json --show-all

# The math for one matchup
python edge_finder.py explain --data games.json --home DUKE
```

Override any knob without editing the file:
`--cap 10 --shrink 0.80 --sigma 18.7 --top 5 --floor 0`.

## Data format (`games.json`)

```json
{
  "config": { "cap_pct": 10.0, "shrink": 0.80, "sigma": 18.7, "top_n": 5 },
  "games": [
    { "away": "TULN", "home": "DUKE", "home_spread": -7.5,
      "home_price": -110, "away_price": -110, "model_margin": 13.0 },
    { "away": "BAY", "home": "SMU", "home_spread": -2.5,
      "home_rating": 12.4, "away_rating": 3.7 }
  ]
}
```

- Give each game **either** `model_margin` (expected home margin, points) **or**
  both `home_rating` / `away_rating` (SP+, and the `/16.2 + 3` formula is applied).
- `played: true` filters a game out before scoring.
- `home_price` / `away_price` default to −110 when omitted.

See `sample_games.json` for a full worked slate that reproduces the card.

## The data feed (`fetch_slate.py`)

`fetch_slate.py` builds a **real** `games.json` so no matchup or line is ever
typed by hand:

- **Schedule + spreads** — ESPN's public CFB scoreboard API (no key). Reads the
  preferred sportsbook's line (default DraftKings) and records which book.
- **SP+ ratings** — a local table you maintain, joined by team name.

```bash
# Real upcoming-Saturday slate, scored:
python fetch_slate.py --saturday --ratings sp_ratings.json --out games.json
python edge_finder.py scan --data games.json

# Or piped in one line:
python fetch_slate.py --date 2026-09-05 --ratings sp_ratings.json | \
    python edge_finder.py scan --data -
```

Ratings file — JSON `{"Duke": 12.4, ...}` or CSV `team,rating`
(see `sp_ratings.sample.json`). Games with no line or a missing rating are
flagged and counted, not silently dropped; completed games are marked played and
skipped.

> **Network note:** some sandboxes (including Claude Code on the web) block
> outbound egress, so `fetch_slate.py` must run where `site.api.espn.com` is
> reachable — your machine, or a scheduled job with normal network access.

### ⚠️ Calibrate the divisor before you trust a number

The `/16.2` divisor only makes sense for the SP+ scale it was tuned on. With
ordinary SP+ overall ratings (e.g. Duke +12, Tulane +5) the formula yields
*Duke by ~3*, **not** the "Duke by ~13" from the reference card. Before acting
on any slate, confirm the model reproduces a line you trust:

```bash
python edge_finder.py explain --data games.json --home DUKE
# adjust with --divisor / a config block until "model margin" looks right
```

If your model line is off, every edge downstream is off. This is the single
knob that matters most.

## Making it a daily habit

Chain the two scripts and schedule them (cron, launchd, or a Claude Code
trigger), piping the JSON to a text / email / Slack:

```bash
python fetch_slate.py --saturday --ratings sp_ratings.json --out games.json && \
python edge_finder.py scan --data games.json --json > slate.json
```

*Forward-only research, not betting advice. Re-verify every line at kickoff.*
