# CFB Scoring Engine

A college-football betting model that projects a game, converts the projection
to a win probability, compares it against the sharpest available market
(Pinnacle) with the vig stripped out, and sizes a bet from the resulting edge.

The pipeline is three cooperating modules, all standard-library only (no
third-party runtime dependency):

```
fetch_lines_pinnacle.py   market side   -> Pinnacle moneylines (American odds)
devig_pinnacle.py         fair value    -> no-vig probabilities + edge
cfb_model.py              model side    -> projection, win prob, EV, stake
```

## Flow

```
team stats ─► composite scores ─► projected margin ─► model win prob
                                                            │
Pinnacle odds ─► devig() ─► market no-vig prob ────────────┤
                                                            ▼
                                          edge ─► confidence-adjusted EV / stake
```

## Modules

### `devig_pinnacle.py`
Strips the bookmaker margin from a two-outcome market quoted in American odds.

```python
from devig_pinnacle import devig, compute_edge

r = devig(-280, 230, method="shin")   # methods: multiplicative, additive, power, shin
r.home_prob, r.away_prob              # sum to exactly 1.0
r.overround                           # raw margin, e.g. 0.0399
compute_edge(model_prob, r.home_prob) # model_prob - market_prob
```

`shin` (the default) is the Shin (1992) model, which accounts for informed
money and tends to match Pinnacle's closing lines best. The Shin exponent `z`
is solved by bisection, so the method is robust without a closed-form derivation
to get wrong.

### `fetch_lines_pinnacle.py`
Pulls Pinnacle NCAAF moneylines. Pinnacle's own public API was retired, so this
targets [The Odds API](https://the-odds-api.com) (Pinnacle is a bookmaker key
there) using stdlib `urllib`.

```python
from fetch_lines_pinnacle import fetch_pinnacle_lines
lines = fetch_pinnacle_lines()   # reads ODDS_API_KEY from the environment
```

Network I/O and parsing are split: `parse_odds_api(payload)` and
`load_lines_from_file(path)` are pure and unit-tested against a saved fixture,
so nothing here needs a live key or a network to test.

### `cfb_model.py`
The scoring engine. Blends in-season efficiency (EPA, success rate,
explosiveness, havoc — regressed to the mean by sample size) with a preseason
talent/roster prior, weighting the prior more early in the season. Adds
situational (pace, third down, red zone, special teams) and contextual
(home field, travel, weather, injuries, trap games, opt-outs) adjustments,
converts the projected margin to a win probability with a CFB-calibrated
logistic, and produces edge, EV, and a **confidence-adjusted fractional-Kelly
stake**. Confidence takes the *minimum* of sample-size, schedule-connectivity,
and season-timing factors, so one thin input (e.g. zero common opponents) alone
zeroes the stake.

```python
from cfb_model import project_game
result = project_game(home, away, context,
                      pinnacle_home_odds=-280, pinnacle_away_odds=230)
```

## Run it

```bash
python devig_pinnacle.py     # de-vig demo across all four methods
python cfb_model.py          # full projection on a sample matchup
ODDS_API_KEY=xxx python fetch_lines_pinnacle.py   # live Pinnacle lines
```

## Tests

Pure stdlib `unittest` — no pytest, no network, no API key required:

```bash
python -m unittest discover -s tests -v
```

## Tuning notes

The weights and constants (`REGRESSION_K`, `CFB_LOGISTIC_SCALE`, the composite
coefficients, `MIN_EDGE_THRESHOLD`, `BASE_KELLY_FRACTION`) are documented
starting points, not fitted values — refit them against your own backtest
before staking real money. One item is flagged inline in
`context_adjustment`: confirm the sign convention of `travel_distance_penalty`
matches however your precompute step produces it.
