"""Colours and the application stylesheet.

Dark by design rather than by preference: the window is mostly a video
player and a timeline, and footage reads best against a dark surround.
Chapter colours encode the backend's `kind` (its priority signal), so they
are chosen to stay distinguishable from one another and from the red used
for redaction, which must never be mistaken for an ordinary chapter.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

BG = "#151619"
PANEL = "#1b1d21"
RAISED = "#23262b"
HOVER = "#2b2f35"
BORDER = "#30343b"
TEXT = "#e7e8eb"
MUTED = "#9299a3"
FAINT = "#5d636c"
ACCENT = "#6ea8fe"
DANGER = "#ff6161"
WARN = "#f5c451"
OK = "#5ec48a"

KIND_COLORS = {
    "pivotal": "#f0a742",
    "active": "#4f8ff0",
    "routine": "#3fae97",
    "idle": "#6b7380",
}
# For chapters with no kind (indexes from the curve segmenter): two neutral
# tones alternated so adjacent chapters stay visually separate.
UNKINDED = ("#6f7fa0", "#86789f")

KIND_LABELS = {
    "pivotal": "Pivotal — something decided, fixed or discovered",
    "active": "Active — ordinary productive work",
    "routine": "Routine — repetitive or mechanical",
    "idle": "Idle — little or nothing happening",
}

STATUS_COLORS = {
    "new": FAINT,
    "queued": MUTED,
    "running": ACCENT,
    "ready": OK,
    "failed": DANGER,
    "cancelled": FAINT,
}
STATUS_LABELS = {
    "new": "Not analyzed",
    "queued": "Queued",
    "running": "Analyzing",
    "ready": "Ready",
    "failed": "Failed",
    "cancelled": "Cancelled",
}


def kind_color(kind: str, position: int = 0) -> QColor:
    return QColor(KIND_COLORS.get(kind) or UNKINDED[position % 2])


def mono_font(size: int = 11) -> QFont:
    f = QFont("Menlo")
    f.setStyleHint(QFont.Monospace)
    f.setPointSize(size)
    return f


_QSS = f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:horizontal {{ width: 1px; background: {BORDER}; }}
QSplitter::handle:vertical {{ height: 1px; background: {BORDER}; }}

#Panel {{ background: {PANEL}; }}
#Header {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
#Transport {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
#Caption {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
#Banner {{ background: {RAISED}; border-bottom: 1px solid {BORDER}; }}
#VideoArea {{ background: #000; }}

QLabel#Title {{ font-size: 15px; font-weight: 600; }}
QLabel#Subtle {{ color: {MUTED}; }}
QLabel#Faint {{ color: {FAINT}; }}
QLabel#SectionLabel {{ color: {MUTED}; font-size: 11px; font-weight: 600;
                       letter-spacing: 0.5px; }}
QLabel#Warning {{ color: {WARN}; }}
QLabel#Danger {{ color: {DANGER}; }}

QPushButton, QToolButton {{
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 12px;
}}
QPushButton:hover, QToolButton:hover {{ background: {HOVER}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {FAINT}; }}
QPushButton#Primary {{ background: {ACCENT}; color: #0b1730;
                       border-color: {ACCENT}; font-weight: 600; }}
QPushButton#Primary:hover {{ background: #86b7ff; }}
QPushButton#Primary:disabled {{ background: {RAISED}; color: {FAINT};
                                border-color: {BORDER}; }}
QPushButton#DangerButton {{ color: {DANGER}; }}
QToolButton#Flat {{ background: transparent; border: none; padding: 4px; }}
QToolButton#Flat:hover {{ background: {HOVER}; }}
QToolButton::menu-indicator {{ image: none; }}

QLineEdit, QPlainTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 4px 6px; selection-background-color: {ACCENT};
    selection-color: #0b1730;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox QAbstractItemView {{ background: {RAISED}; border: 1px solid {BORDER}; }}

QListWidget, QTreeWidget, QTableView {{
    background: {PANEL}; border: none; outline: none;
    alternate-background-color: #1f2126;
}}
QTreeWidget::item, QTableView::item, QListWidget::item {{ padding: 3px 2px; }}
QTreeWidget::item:selected, QTableView::item:selected,
QListWidget::item:selected {{ background: #2c3a52; color: {TEXT}; }}
QHeaderView::section {{
    background: {PANEL}; color: {MUTED}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 4px 6px; font-size: 11px;
}}

QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; }}
QTabBar::tab {{
    background: transparent; color: {MUTED}; padding: 7px 14px;
    border: none; border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QGroupBox {{ border: 1px solid {BORDER}; border-radius: 8px; margin-top: 14px;
             padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {MUTED}; }}
QCheckBox, QRadioButton {{ spacing: 7px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px;
                               min-height: 24px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 4px;
                                 min-width: 24px; margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {BORDER};
            padding: 6px; }}
QMenu {{ background: {RAISED}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {HOVER}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
QStatusBar {{ background: {PANEL}; color: {MUTED};
              border-top: 1px solid {BORDER}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER};
                              border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; margin: -5px 0; border-radius: 6px;
                              background: {TEXT}; }}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(PANEL))
    pal.setColor(QPalette.AlternateBase, QColor("#1f2126"))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(RAISED))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#0b1730"))
    pal.setColor(QPalette.ToolTipBase, QColor(RAISED))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.PlaceholderText, QColor(FAINT))
    pal.setColor(QPalette.Link, QColor(ACCENT))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(FAINT))
    app.setPalette(pal)
    app.setStyleSheet(_QSS)
