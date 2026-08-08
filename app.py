import streamlit as st
import requests
import pandas as pd

from data.reference import add_readable_columns

st.title("FPL Data Explorer")

response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
data = response.json()

df = pd.DataFrame(data["elements"])
df = add_readable_columns(df, teams=data["teams"], element_types=data["element_types"])

st.write(f"Loaded {len(df)} players.")

st.sidebar.header("Filters")

positions = sorted(df["position"].unique())
selected_positions = st.sidebar.multiselect("Position", options=positions, default=positions)

teams = sorted(df["team_name"].unique())
selected_teams = st.sidebar.multiselect("Team", options=teams, default=teams)

position_team_df = df[df["position"].isin(selected_positions) & df["team_name"].isin(selected_teams)]
min_price = float(position_team_df["price"].min())
max_price = float(position_team_df["price"].max())

price_range = st.sidebar.slider(
    "Price range (£m)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=0.1,
)

filtered_df = position_team_df[position_team_df["price"].between(price_range[0], price_range[1])]

mandatory_columns = ["web_name", "team_name", "position", "price"]
available_columns = [
    "total_points", "form", "points_per_game", "selected_by_percent",
    "minutes", "goals_scored", "assists", "clean_sheets",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "bonus", "ict_index",
]
default_columns = ["total_points"]
selected_columns = st.sidebar.multiselect("Additional columns to display", options=available_columns, default=default_columns)

display_columns = mandatory_columns + selected_columns

st.sidebar.header("Sort")
sort_columns = st.sidebar.multiselect(
    "Sort by (in order of priority)", options=display_columns, default=["price"]
)

sort_ascending = []
for column in sort_columns:
    ascending = st.sidebar.toggle(f"Ascending: {column}", value=False, key=f"sort_asc_{column}")
    sort_ascending.append(ascending)

if sort_columns:
    sorted_df = filtered_df.sort_values(by=sort_columns, ascending=sort_ascending)
else:
    sorted_df = filtered_df

st.write(f"Showing {len(sorted_df)} players.")
st.dataframe(sorted_df[display_columns])
