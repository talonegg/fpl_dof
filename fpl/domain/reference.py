"""Lookup tables for translating FPL's raw numeric codes into readable names."""


def build_team_map(teams):
    """Map team id -> team name, e.g. {1: 'Arsenal', 2: 'Aston Villa', ...}"""
    return {team["id"]: team["name"] for team in teams}


def build_position_map(element_types):
    """Map position id -> position name, e.g. {1: 'Goalkeeper', 2: 'Defender', ...}"""
    return {et["id"]: et["singular_name"] for et in element_types}


def add_readable_columns(df, teams, element_types):
    """Return a copy of the player dataframe with readable team/position/price columns."""
    team_map = build_team_map(teams)
    position_map = build_position_map(element_types)

    df = df.copy()
    df["team_name"] = df["team"].map(team_map)
    df["position"] = df["element_type"].map(position_map)
    df["price"] = df["now_cost"] / 10
    return df
