"""
CLI display helpers – coloured terminal output, banners, menus.
"""

import os
import sys
from datetime import datetime


# ── ANSI colour codes ──────────────────────────────────────────────────────────
class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[40m"


def _supports_color():
    """Return True if the terminal supports ANSI colours."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def cprint(text: str, color: str = Colors.WHITE, bold: bool = False):
    if _supports_color():
        prefix = Colors.BOLD if bold else ""
        print(f"{prefix}{color}{text}{Colors.RESET}")
    else:
        print(text)


# ── Public helpers ─────────────────────────────────────────────────────────────

def print_banner():
    """Print the startup ASCII banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
  ╔══════════════════════════════════════════════════════════════════════╗
  ║                                                                      ║
  ║        ███████╗██╗   ██╗███████╗    ██╗   ██╗███████╗               ║
  ║        ██╔════╝██║   ██║██╔════╝    ██║   ██║██╔════╝               ║
  ║        ███████╗██║   ██║███████╗    ██║   ██║███████╗               ║
  ║        ╚════██║╚██╗ ██╔╝╚════██║    ╚██╗ ██╔╝╚════██║               ║
  ║        ███████║ ╚████╔╝ ███████║     ╚████╔╝ ███████║               ║
  ║        ╚══════╝  ╚═══╝  ╚══════╝      ╚═══╝  ╚══════╝               ║
  ║                                                                      ║
  ║          SMART VISION SYSTEM  –  Final Year Project                  ║
  ║          Multi-Modal Object Detection & Tracking                     ║
  ║                                                                      ║
  ║          🚗  Traffic Management  |  👁  Smart Surveillance           ║
  ╚══════════════════════════════════════════════════════════════════════╝
{Colors.RESET}"""
    print(banner)
    print(f"  {Colors.YELLOW}Started at : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}{Colors.RESET}")
    print(f"  {Colors.YELLOW}Python     : {sys.version.split()[0]}{Colors.RESET}\n")


def print_menu():
    """Print the main mode selection menu header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}  ══════════════  SELECT MODE  ══════════════{Colors.RESET}")


def print_success(msg: str):
    cprint(f"  ✅  {msg}", Colors.GREEN, bold=True)


def print_error(msg: str):
    cprint(f"  ❌  {msg}", Colors.RED, bold=True)


def print_info(msg: str):
    cprint(f"  ℹ️   {msg}", Colors.CYAN)


def print_warning(msg: str):
    cprint(f"  ⚠️   {msg}", Colors.YELLOW, bold=True)


def print_alert(msg: str):
    cprint(f"  🚨  {msg}", Colors.RED, bold=True)


def print_stat(label: str, value):
    cprint(f"  │  {label:<28} {Colors.YELLOW}{value}{Colors.RESET}", Colors.WHITE)


def print_section(title: str):
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}  ── {title} {'─' * max(0, 45 - len(title))}{Colors.RESET}")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")
