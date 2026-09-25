"""Visual theme, matching the design canvas (Fraunces + Manrope, ivory/teal/orange).
Change hex values or the Google Fonts import below and it applies everywhere —
this is the one file to touch for a restyle.
"""

COLORS = {
    "bg": "#F7F5F1",
    "surface": "#FFFFFF",
    "ink": "#1C1B1A",
    "ink_soft": "#6B6560",
    "border": "#E4DFD7",
    "teal_dark": "#1D4E4E",
    "teal": "#2A6F6F",
    "teal_bg": "#EAF2F0",
    "orange": "#C7622D",
    "orange_bg": "#FDF3ED",
    "orange_border": "#EBC9AD",
    "green": "#3F7A5D",
    "red": "#B23A3A",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Manrope', sans-serif;
    color: {COLORS['ink']};
}}
.stApp {{
    background: {COLORS['bg']};
}}
h1, h2, h3 {{
    font-family: 'Fraunces', serif !important;
    font-weight: 700 !important;
}}
.sc-tier-badge {{
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 999px;
    background: {COLORS['teal_bg']};
    color: {COLORS['teal_dark']};
}}
.sc-card {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 14px;
    padding: 20px 24px;
}}
.sc-callout-orange {{
    background: {COLORS['orange_bg']};
    border: 1px solid {COLORS['orange_border']};
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 13px;
    color: #8A4A22;
    line-height: 1.6;
}}
.sc-result-meets {{ color: {COLORS['green']}; font-weight: 700; }}
.sc-result-not-met {{ color: {COLORS['red']}; font-weight: 700; }}
.sc-result-cannot {{ color: {COLORS['ink_soft']}; font-style: italic; }}
div.stButton > button {{
    background: {COLORS['teal_dark']};
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 700;
    padding: 10px 24px;
}}
div.stButton > button:hover {{
    background: {COLORS['teal']};
    color: white;
}}
</style>
"""
