
"""Colors and fonts shared by the desktop controls and plots."""

from tkinter import ttk

NAV = "#153634"
ACCENT = "#126B58"
ACCENT_DARK = "#104F43"
SELECTED = "#E7F1EB"
BACKGROUND = "#F2F5F3"
PANEL = "#F8FAF7"
PAPER = "#FFFFFF"
LINE = "#D7E3DD"
INK = "#152E31"
MUTED = "#536B70"
ERROR = "#993E39"

UI_FONT = ("Microsoft YaHei UI", 10)
SMALL_FONT = ("Microsoft YaHei UI", 9)
HEADING_FONT = ("Microsoft YaHei UI", 11, "bold")
LOG_FONT = ("Consolas", 10)


def apply_theme(root):
    root.configure(background=BACKGROUND)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=UI_FONT, background=BACKGROUND, foreground=INK)
    style.configure("TFrame", background=BACKGROUND)
    style.configure("Card.TFrame", background=PAPER, bordercolor=LINE, borderwidth=1, relief="solid")
    style.configure("Sidebar.TFrame", background=PANEL, bordercolor=LINE, borderwidth=1, relief="solid")
    style.configure("TLabel", background=BACKGROUND, foreground=INK)
    style.configure("Panel.TLabel", background=PANEL, foreground=INK)
    style.configure("PanelHeading.TLabel", background=PANEL, foreground=INK, font=HEADING_FONT)
    style.configure("File.TLabel", background=PAPER, foreground=MUTED, font=SMALL_FONT)
    style.configure("Status.TLabel", background=PAPER, foreground=INK, font=HEADING_FONT)
    style.configure("Log.TLabel", background=PAPER, foreground=INK, font=HEADING_FONT)

    style.configure("TNotebook", background=BACKGROUND, borderwidth=0, bordercolor=LINE,
                    lightcolor=BACKGROUND, darkcolor=BACKGROUND, tabmargins=(0, 0, 0, 0))
    style.configure("TNotebook.Tab", padding=(16, 8), background=BACKGROUND,
                    bordercolor=LINE, lightcolor=BACKGROUND, darkcolor=BACKGROUND, foreground=MUTED)
    style.map("TNotebook.Tab", background=[("selected", PAPER), ("active", SELECTED)],
              foreground=[("selected", ACCENT), ("active", INK)])

    style.configure("TButton", padding=(12, 6), background=PAPER, bordercolor=LINE,
                    lightcolor=PAPER, darkcolor=PAPER, foreground=INK)
    style.map("TButton", background=[("pressed", SELECTED), ("active", SELECTED)],
              bordercolor=[("focus", ACCENT), ("active", ACCENT)])
    style.configure("Accent.TButton", background=ACCENT, bordercolor=ACCENT, foreground=PAPER,
                    lightcolor=ACCENT, darkcolor=ACCENT, font=HEADING_FONT)
    style.map("Accent.TButton",
              background=[("disabled", LINE), ("pressed", ACCENT_DARK), ("active", ACCENT_DARK)],
              bordercolor=[("focus", ACCENT_DARK), ("active", ACCENT_DARK)],
              foreground=[("disabled", MUTED), ("!disabled", PAPER)])

    style.configure("TEntry", fieldbackground=PAPER, background=PAPER, bordercolor=LINE,
                    lightcolor=PAPER, darkcolor=PAPER, padding=(6, 4), foreground=INK)
    style.map("TEntry", bordercolor=[("focus", ACCENT)])
    for control in ("TCheckbutton", "TRadiobutton"):
        style.configure(control, background=PANEL, foreground=INK, padding=(0, 3),
                        indicatorbackground=PAPER, indicatorforeground=ACCENT,
                        upperbordercolor=PAPER, lowerbordercolor=LINE)
        style.map(control, background=[("active", SELECTED)],
                  indicatorbackground=[("selected", SELECTED)],
                  indicatorforeground=[("selected", ACCENT)])
    style.configure("TSeparator", background=LINE)
    style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=SELECTED,
                    bordercolor=SELECTED, lightcolor=ACCENT, darkcolor=ACCENT, thickness=9)
    style.configure("Vertical.TScrollbar", background=LINE, troughcolor=PANEL,
                    bordercolor=PANEL, lightcolor=LINE, darkcolor=LINE, arrowsize=11)
