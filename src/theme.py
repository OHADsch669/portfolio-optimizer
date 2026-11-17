"""
Colors and chart style for the dashboard.
Black and white, like a financial news site: black menus with white text,
a white page and white charts, green/red for gains/losses and gold as a highlight.
"""
import plotly.graph_objects as go
import plotly.io as pio

# Page
BG = "#FFFFFF"          # page and chart background (same as config.toml)
SURFACE = "#F5F6F8"     # light grey: inputs and cards on the white page
BORDER = "#E3E5E8"      # card borders and chart gridlines
TEXT = "#111111"
MUTED = "#8A9099"       # secondary text, single-stock dots, SPY line

# Menus (sidebar)
MENU_BG = "#0B0B0B"
MENU_SURFACE = "#1C1C1C"
MENU_TEXT = "#FFFFFF"

# Price strip at the top (on black): brighter green/red so they read well on a dark background
TAPE_UP = "#3DDC84"
TAPE_DOWN = "#FF5C5C"

# Chart colors
INK = "#111111"         # main series: your portfolio, allocation bars
FRONTIER = "#111111"    # efficient frontier line
GREEN = "#0E9F4F"
RED = "#D92D20"
GOLD = "#F2A900"        # highlights: min-variance point, risk contribution
PURPLE = "#7A5AF8"

# Order used when a chart has several series (e.g. backtest strategies)
SERIES = [INK, GREEN, GOLD, PURPLE, MUTED, RED]

# Correlation heatmap: green (move apart) -> white (unrelated) -> red (move together)
DIVERGING = [[0.0, GREEN], [0.5, BG], [1.0, RED]]


def register_plotly_template() -> None:
    """Make every Plotly chart match the app: white background, light grey grid, dark text."""
    template = go.layout.Template(pio.templates["plotly_white"])
    template.layout.update(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Source Sans Pro, sans-serif"),
        colorway=SERIES,
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor="#C9CDD2"),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor="#C9CDD2"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=INK, font=dict(color=TEXT)),
        dragmode=False,                   # dragging does nothing; zoom with the +/- buttons
    )
    pio.templates["clean"] = template
    pio.templates.default = "clean"


# Plotly toolbar: only one-click actions, so the mouse never gets stuck in a zoom/pan mode
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "autoScale2d"],
}

# In the full-screen window: the toolbar is always visible and includes zoom-to-area and pan
FULL_SCREEN_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
}


# Each bar starts 40 ms after the previous one, so the bars grow in a wave
BAR_STAGGER = "\n".join(
    f'[class*="st-key-chart_"] .bars .point:nth-child({i}) path {{ animation-delay: {(i - 1) * 0.04:.2f}s; }}'
    for i in range(2, 21)
)

CSS = f"""
<style>
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to   {{ opacity: 1; transform: none; }}
}}
[role="tabpanel"], [data-baseweb="tab-panel"] {{ animation: fadeIn 0.45s ease-out; }}

[data-testid="stAppViewContainer"] {{ background: {BG}; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* menus: black with white text */
[data-testid="stSidebar"] {{ background: {MENU_BG}; }}
/* selected stocks in the sidebar: white chips with black text */
[data-testid="stSidebar"] [data-baseweb="tag"],
[data-testid="stSidebar"] [data-tag] {{ background-color: {MENU_TEXT} !important; color: {MENU_BG} !important; }}
[data-testid="stSidebar"] [data-baseweb="tag"] *,
[data-testid="stSidebar"] [data-tag] * {{ color: {MENU_BG} !important; }}
[data-testid="stSidebar"] [data-tag] svg path {{ stroke: {MENU_BG} !important; }}

/* metric cards */
[data-testid="stMetric"] {{
  background: {BG};
  border: 1px solid {BORDER};
  border-radius: 12px;
  padding: 14px 18px;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}}
[data-testid="stMetric"]:hover {{
  transform: translateY(-2px);
  border-color: {INK};
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}}
[data-testid="stMetricValue"] {{ color: {TEXT}; }}

/* date picker (opens from the black sidebar): white numbers on dark, greyed-out days you can't pick */
[data-testid="stDateInputCalendar"] {{ color: {MENU_TEXT}; }}
[data-testid="stDateInputCalendar"] [role="button"]:not([data-selected]) {{ color: {MENU_TEXT} !important; }}
[data-testid="stDateInputCalendar"] [role="button"][data-disabled] {{ color: rgba(255,255,255,0.3) !important; }}
[data-testid="stDateInputCalendar"] [role="button"][data-outside-month]:not([data-disabled]) {{ color: rgba(255,255,255,0.55) !important; }}
[data-testid="stDateInputCalendar"] [role="button"]:not([data-selected]):not([data-disabled]):hover {{
  background-color: #333333 !important; }}

/* charts: one way to go full screen (our own button), so hide Streamlit's built-in one */
[data-testid="stPlotlyChart"] button[aria-label="Fullscreen"],
[data-testid="stPlotlyChart"] .modebar-btn[data-title="Fullscreen"] {{ display: none !important; }}
/* the full-screen chart window fills almost the whole screen */
div[role="dialog"]:has([data-testid="stPlotlyChart"]) {{ width: 96vw !important; max-width: 96vw !important; }}

/* moving price strip at the top of the page */
.tape {{
  display: flex; align-items: center; gap: 14px;
  background: {MENU_BG}; color: {MENU_TEXT};
  border-radius: 10px; padding: 9px 14px; margin-bottom: 6px;
  font-size: 0.92rem; white-space: nowrap; overflow: hidden;
}}
.tape-date {{ color: #9AA0A6; font-size: 0.8rem; flex-shrink: 0; padding-right: 12px; border-right: 1px solid #333; }}
.tape-window {{ overflow: hidden; flex: 1;
  mask-image: linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent); }}
.tape-track {{ display: inline-block; animation: tape-scroll linear infinite; }}
.tape:hover .tape-track {{ animation-play-state: paused; }}
.tape-item {{ margin-right: 34px; }}
.tape-up {{ color: {TAPE_UP}; }}
.tape-down {{ color: {TAPE_DOWN}; }}
@keyframes tape-scroll {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}

/* charts draw themselves in whenever they update (bars grow, points and lines fade in, growth lines
   draw from left to right). Streamlit redraws a chart from scratch on each change, so the animation
   replays every time the stocks or the strategy change. */
@keyframes grow-x {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
@keyframes grow-y {{ from {{ transform: scaleY(0); }} to {{ transform: scaleY(1); }} }}
@keyframes fade-in {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes reveal {{ from {{ clip-path: inset(0 100% 0 0); }} to {{ clip-path: inset(0 0 0 0); }} }}
[class*="st-key-chart_"] .bars .point path {{ transform-box: fill-box; }}
.st-key-chart_allocation .bars .point path,
.st-key-chart_sectors .bars .point path {{
  transform-origin: left center; animation: grow-x 0.7s cubic-bezier(.2,.8,.2,1) both; }}
.st-key-chart_risk_vs_money .bars .point path {{
  transform-origin: center bottom; animation: grow-y 0.7s cubic-bezier(.2,.8,.2,1) both; }}
{BAR_STAGGER}
[class*="st-key-chart_"] .scatterlayer .points {{ animation: fade-in 0.6s ease-out 0.15s both; }}
.st-key-chart_frontier .scatterlayer .js-line {{ animation: fade-in 0.8s ease-out both; }}
[class*="st-key-chart_growth"] .scatterlayer,
[class*="st-key-chart_drawdown"] .scatterlayer {{ animation: reveal 1.2s cubic-bezier(.3,.7,.3,1) both; }}
.st-key-chart_correlation .heatmaplayer {{ animation: fade-in 0.6s ease-out both; }}

/* tabs: black underline on the active tab */
[data-baseweb="tab-highlight"] {{ background-color: {INK}; }}

@media (prefers-reduced-motion: reduce) {{
  [role="tabpanel"], [data-baseweb="tab-panel"] {{ animation: none; }}
  [data-testid="stMetric"], [data-testid="stMetric"]:hover {{ transition: none; transform: none; }}
  .tape-track {{ animation: none; }}
  [class*="st-key-chart_"] * {{ animation: none !important; }}
  .tape-window {{ overflow-x: auto; }}
}}
</style>
"""
