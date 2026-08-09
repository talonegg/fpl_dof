"""Turning bookmaker prices into the numbers an FPL model can use.

Three steps, each of which is wrong to skip:

**De-vigging.** A bookmaker's prices imply probabilities summing to more than
1 — typically 1.05 for a football match. That excess is their margin, and
treating raw implied probabilities as real ones makes every outcome look about
5% more likely than it is. Removing it proportionally is the standard
approach and the one used here.

**Consensus.** One bookmaker is one opinion, and a stale or eccentric price
distorts everything downstream. The median across bookmakers is used because
it ignores an outlier entirely, where a mean does not.

**Translating to FPL quantities.** A match probability is not a clean sheet.
Assuming each team's goals are Poisson, the market's total-goals line and its
win probabilities together pin down each team's expected goals — and once you
have those, a clean sheet is just the chance the opponent scores zero. That
independence assumption is not quite true (goals correlate within a match) but
it is the standard working model and it is stated rather than hidden.

Nothing here fetches anything; it all operates on the frame
:mod:`fpl.sources.odds` produces.
"""

from __future__ import annotations

import math

import pandas as pd

from fpl.sources.odds import DRAW, MARKET_H2H, MARKET_TOTALS

# When the market gives no total-goals line, fall back to the long-run Premier
# League average. Roughly 2.75 goals a game across recent seasons.
DEFAULT_TOTAL_GOALS = 2.75

# Bounds for the goals split search. A team is never expected to score less
# than a fifth or more than four fifths of the goals in a match.
MIN_HOME_SHARE = 0.2
MAX_HOME_SHARE = 0.8

MAX_GOALS = 10


def implied_probability(price: pd.Series) -> pd.Series:
    """Raw implied probability of a decimal price, margin included."""
    return 1.0 / price


def devig(prices: pd.Series) -> pd.Series:
    """Remove the bookmaker's margin from one book's set of prices.

    Proportional (multiplicative) de-vigging: scale every implied probability
    by the same factor so they sum to 1. Must be applied within a single
    bookmaker's market for a single match -- across bookmakers the sum has no
    meaning and the result is not a probability.
    """
    implied = implied_probability(prices)
    total = implied.sum()
    if total <= 0:
        return implied
    return implied / total


def overround(prices: pd.Series) -> float:
    """How much more than 1 a book's implied probabilities sum to.

    1.05 means a 5% margin. Useful as a sanity check: a book that does not
    overround has been misparsed.
    """
    return float(implied_probability(prices).sum())


def match_probabilities(odds: pd.DataFrame) -> pd.DataFrame:
    """Consensus, de-vigged win/draw/loss probabilities per match.

    Returns one row per match with ``home_win``, ``draw``, ``away_win``, plus
    the ``bookmakers`` count behind it so a thin market can be spotted.
    """
    columns = [
        "match_id",
        "commence_time",
        "home_team",
        "away_team",
        "home_win",
        "draw",
        "away_win",
        "bookmakers",
    ]
    if odds.empty or MARKET_H2H not in set(odds.get("market", [])):
        return pd.DataFrame(columns=columns)

    head_to_head = odds[odds["market"] == MARKET_H2H].copy()
    # De-vig within each bookmaker's book, which is the only place the margin
    # is well defined.
    head_to_head["probability"] = head_to_head.groupby(["match_id", "bookmaker"])[
        "price"
    ].transform(devig)

    consensus = (
        head_to_head.groupby(["match_id", "home_team", "away_team", "outcome"])
        .agg(probability=("probability", "median"), bookmakers=("bookmaker", "nunique"))
        .reset_index()
    )

    rows = []
    for (match_id, home, away), group in consensus.groupby(["match_id", "home_team", "away_team"]):
        by_outcome = dict(zip(group["outcome"], group["probability"], strict=True))
        probabilities = {
            "home_win": by_outcome.get(home, float("nan")),
            "draw": by_outcome.get(DRAW, float("nan")),
            "away_win": by_outcome.get(away, float("nan")),
        }
        # The median is taken per outcome, so the three need not sum to 1.
        total = sum(value for value in probabilities.values() if value == value)
        if total > 0:
            probabilities = {key: value / total for key, value in probabilities.items()}

        commence = odds.loc[odds["match_id"] == match_id, "commence_time"]
        rows.append(
            {
                "match_id": match_id,
                "commence_time": commence.iloc[0] if len(commence) else None,
                "home_team": home,
                "away_team": away,
                **probabilities,
                "bookmakers": int(group["bookmakers"].max()),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def total_goals_line(odds: pd.DataFrame) -> pd.DataFrame:
    """Market-implied expected total goals per match, from the totals market.

    The line itself (2.5, 3.0) is the bookmaker's estimate of the median, and
    the prices either side say which way it leans. Inverting the Poisson gives
    a mean rather than a median, which is what the goal split needs.
    """
    columns = ["match_id", "expected_total_goals"]
    if odds.empty or MARKET_TOTALS not in set(odds.get("market", [])):
        return pd.DataFrame(columns=columns)

    totals = odds[odds["market"] == MARKET_TOTALS].copy()
    totals["probability"] = totals.groupby(["match_id", "bookmaker", "point"])["price"].transform(
        devig
    )

    overs = totals[totals["outcome"].str.lower() == "over"]
    if overs.empty:
        return pd.DataFrame(columns=columns)

    consensus = overs.groupby(["match_id", "point"])["probability"].median().reset_index()

    rows = []
    for match_id, group in consensus.groupby("match_id"):
        # Use the line closest to the usual 2.5, which is the most liquid.
        closest = group.iloc[(group["point"] - 2.5).abs().argsort().iloc[0]]
        rows.append(
            {
                "match_id": match_id,
                "expected_total_goals": _total_goals_from_over_probability(
                    float(closest["point"]), float(closest["probability"])
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _poisson_pmf(k: int, mean: float) -> float:
    if mean <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-mean) * mean**k / math.factorial(k)


def _poisson_cdf(k: int, mean: float) -> float:
    return sum(_poisson_pmf(i, mean) for i in range(k + 1))


def _total_goals_from_over_probability(line: float, over_probability: float) -> float:
    """Expected goals consistent with P(total > line) under a Poisson.

    Solved by bisection because the relationship has no closed form. Bounded
    well outside anything football produces, so it always converges.
    """
    if not 0 < over_probability < 1:
        return DEFAULT_TOTAL_GOALS

    below = math.floor(line)
    low, high = 0.2, 8.0
    for _ in range(60):
        middle = (low + high) / 2
        implied_over = 1 - _poisson_cdf(below, middle)
        if implied_over < over_probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _home_share_from_win_probability(total_goals: float, home_win_probability: float) -> float:
    """Fraction of the goals the home side is expected to score.

    Found by bisection on the share that reproduces the market's home-win
    probability under independent Poisson scorelines.
    """
    if not 0 < home_win_probability < 1:
        return 0.5

    low, high = MIN_HOME_SHARE, MAX_HOME_SHARE
    for _ in range(40):
        middle = (low + high) / 2
        implied = _home_win_probability(total_goals * middle, total_goals * (1 - middle))
        if implied < home_win_probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _home_win_probability(home_goals: float, away_goals: float) -> float:
    """P(home outscores away) for independent Poisson scorelines."""
    probability = 0.0
    for home in range(MAX_GOALS + 1):
        home_probability = _poisson_pmf(home, home_goals)
        if home_probability == 0:
            continue
        probability += home_probability * _poisson_cdf(home - 1, away_goals)
    return probability


def team_expectations(odds: pd.DataFrame) -> pd.DataFrame:
    """Per-team expected goals and clean-sheet probability, from the market.

    One row per team per match: ``team``, ``opponent``, ``is_home``,
    ``expected_goals_for``, ``expected_goals_against``, ``clean_sheet``,
    ``win``. Team-shaped rather than match-shaped, for the same reason
    :mod:`fpl.domain.fixtures` is.

    ``clean_sheet`` is the chance the opponent fails to score, which under the
    Poisson assumption is ``exp(-their expected goals)``.
    """
    columns = [
        "match_id",
        "team",
        "opponent",
        "is_home",
        "expected_goals_for",
        "expected_goals_against",
        "clean_sheet",
        "win",
    ]

    matches = match_probabilities(odds)
    if matches.empty:
        return pd.DataFrame(columns=columns)

    totals = total_goals_line(odds).set_index("match_id")["expected_total_goals"]

    rows = []
    for match in matches.itertuples():
        total = float(totals.get(match.match_id, DEFAULT_TOTAL_GOALS))
        if total != total:
            total = DEFAULT_TOTAL_GOALS

        share = _home_share_from_win_probability(total, match.home_win)
        home_goals = total * share
        away_goals = total * (1 - share)

        rows.append(
            {
                "match_id": match.match_id,
                "team": match.home_team,
                "opponent": match.away_team,
                "is_home": True,
                "expected_goals_for": home_goals,
                "expected_goals_against": away_goals,
                "clean_sheet": math.exp(-away_goals),
                "win": match.home_win,
            }
        )
        rows.append(
            {
                "match_id": match.match_id,
                "team": match.away_team,
                "opponent": match.home_team,
                "is_home": False,
                "expected_goals_for": away_goals,
                "expected_goals_against": home_goals,
                "clean_sheet": math.exp(-home_goals),
                "win": match.away_win,
            }
        )

    return pd.DataFrame(rows, columns=columns)
