"""
Colors and chart style for the dashboard.
The palette comes from the reference image: deep navy background,
electric blue as the main color, green/red for gains/losses, gold as a highlight.
"""
import plotly.graph_objects as go
import plotly.io as pio

BG = "#0A1628"          # page background (same as config.toml)
PANEL = "#122542"       # cards / sidebar
GRID = "#1F3A5F"        # chart gridlines and borders
TEXT = "#E6EEF8"
MUTED = "#8FA6C4"       # secondary text, single-stock dots

BLUE = "#3B9EFF"        # main accent: your portfolio, main series
CYAN = "#5CE1FF"        # efficient frontier line
GREEN = "#3DDC84"
RED = "#FF5C6C"
GOLD = "#F5B83D"        # highlights: min-variance point, risk contribution

# Order used when a chart has several series (e.g. backtest strategies)
SERIES = [BLUE, GREEN, GOLD, CYAN, "#B18CFF", RED]

# Correlation heatmap: blue (move apart) -> navy (unrelated) -> red (move together)
DIVERGING = [[0.0, BLUE], [0.5, PANEL], [1.0, RED]]


def register_plotly_template() -> None:
    """Make every Plotly chart match the app: transparent background, navy grid, light text."""
    template = go.layout.Template(pio.templates["plotly_dark"])
    template.layout.update(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Source Sans Pro, sans-serif"),
        colorway=SERIES,
        xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BLUE, font=dict(color=TEXT)),
    )
    pio.templates["navy"] = template
    pio.templates.default = "navy"


CSS = f"""
<style>
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to   {{ opacity: 1; transform: none; }}
}}
[role="tabpanel"], [data-baseweb="tab-panel"] {{ animation: fadeIn 0.45s ease-out; }}

/* soft blue glow at the top of the page, like the reference image */
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 500px at 70% -10%, rgba(59,158,255,0.16), transparent 60%), {BG};
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {PANEL} 0%, {BG} 100%); }}

/* metric cards */
[data-testid="stMetric"] {{
  background: linear-gradient(145deg, rgba(59,158,255,0.10), rgba(18,37,66,0.6));
  border: 1px solid {GRID};
  border-radius: 12px;
  padding: 14px 18px;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}}
[data-testid="stMetric"]:hover {{
  transform: translateY(-2px);
  border-color: {BLUE};
  box-shadow: 0 6px 22px rgba(59,158,255,0.18);
}}
[data-testid="stMetricValue"] {{ color: {TEXT}; }}

/* tabs: blue underline on the active tab */
[data-baseweb="tab-highlight"] {{ background-color: {BLUE}; }}

@media (prefers-reduced-motion: reduce) {{
  [role="tabpanel"], [data-baseweb="tab-panel"] {{ animation: none; }}
  [data-testid="stMetric"], [data-testid="stMetric"]:hover {{ transition: none; transform: none; }}
}}
</style>
"""
