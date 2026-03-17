"""
╔══════════════════════════════════════════════════════════════════╗
║         SMART VISION SYSTEM - Final Year Project                ║
║         Multi-Modal Object Detection & Tracking                 ║
║         Supports: Traffic Management | Surveillance             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import argparse
from utils.logger import setup_logger
from utils.display import print_banner, print_menu, print_success, print_error, print_info
from config.settings import Settings

logger = setup_logger("main")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Smart Vision System")
    parser.add_argument("--mode", type=str, choices=["traffic", "surveillance"],
                        help="Run directly in a specific mode (skip menu)")
    parser.add_argument("--source", type=str, default=None,
                        help="Video source: path to video file, or 0 for webcam")
    parser.add_argument("--save", action="store_true",
                        help="Save output video to output/recordings/")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without showing video window (headless)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Detection confidence threshold (default: 0.4)")
    return parser.parse_args()


def get_video_source():
    """Prompt user to select video source."""
    print("\n" + "─" * 50)
    print("  SELECT INPUT SOURCE")
    print("─" * 50)
    print("  [1] Webcam (default camera)")
    print("  [2] Video File")
    print("  [3] IP Camera (RTSP stream)")
    print("─" * 50)

    choice = input("\n  Enter choice (1/2/3): ").strip()

    if choice == "1":
        print_info("Using webcam as input source.")
        return 0
    elif choice == "2":
        path = input("  Enter full path to video file: ").strip().strip('"')
        if not os.path.exists(path):
            print_error(f"File not found: {path}")
            return get_video_source()
        print_success(f"Video file selected: {path}")
        return path
    elif choice == "3":
        rtsp = input("  Enter RTSP URL (e.g. rtsp://192.168.1.1/stream): ").strip()
        print_info(f"Using IP camera: {rtsp}")
        return rtsp
    else:
        print_error("Invalid choice. Please enter 1, 2, or 3.")
        return get_video_source()


def get_mode_selection():
    """Display main menu and get mode selection."""
    print_menu()
    print("\n  ┌─────────────────────────────────┐")
    print("  │  [1] 🚗  Traffic Management      │")
    print("  │  [2] 👁   Smart Surveillance      │")
    print("  │  [3] ⚙   Settings                │")
    print("  │  [4] ❌  Exit                     │")
    print("  └─────────────────────────────────┘")

    choice = input("\n  Enter your choice (1/2/3/4): ").strip()

    if choice == "1":
        return "traffic"
    elif choice == "2":
        return "surveillance"
    elif choice == "3":
        show_settings()
        return get_mode_selection()
    elif choice == "4":
        print_info("Exiting Smart Vision System. Goodbye!")
        sys.exit(0)
    else:
        print_error("Invalid choice. Please enter 1, 2, 3, or 4.")
        return get_mode_selection()


def show_settings():
    """Display current settings."""
    settings = Settings()
    print("\n" + "─" * 50)
    print("  CURRENT SETTINGS")
    print("─" * 50)
    print(f"  Model Size        : {settings.model_size}")
    print(f"  Confidence Thresh : {settings.confidence_threshold}")
    print(f"  IOU Threshold     : {settings.iou_threshold}")
    print(f"  Max Track Age     : {settings.max_track_age}")
    print(f"  Device            : {settings.device}")
    print(f"  Output Dir        : {settings.output_dir}")
    print("─" * 50)
    input("\n  Press Enter to go back...")


def run_traffic_mode(source, args):
    """Launch Traffic Management mode."""
    print_success("Launching TRAFFIC MANAGEMENT mode...")
    print_info("Features: Vehicle Counting | Speed Estimation | Lane Violation | Re-ID")
    print_info("Press 'Q' to quit | 'S' to save snapshot | 'R' to reset counters\n")

    from traffic.traffic_system import TrafficSystem
    system = TrafficSystem(
        source=source,
        save_output=args.save,
        show_display=not args.no_display,
        conf_threshold=args.conf
    )
    system.run()


def run_surveillance_mode(source, args):
    """Launch Surveillance mode."""
    print_success("Launching SMART SURVEILLANCE mode...")
    print_info("Features: Person Re-ID | Loitering Detection | Fall Detection | Heatmap")
    print_info("Press 'Q' to quit | 'S' to save snapshot | 'H' to toggle heatmap\n")

    from surveillance.surveillance_system import SurveillanceSystem
    system = SurveillanceSystem(
        source=source,
        save_output=args.save,
        show_display=not args.no_display,
        conf_threshold=args.conf
    )
    system.run()


def main():
    args = parse_arguments()
    print_banner()

    # Handle direct CLI mode launch
    if args.mode:
        source = args.source if args.source else get_video_source()
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if args.mode == "traffic":
            run_traffic_mode(source, args)
        else:
            run_surveillance_mode(source, args)
        return

    # Interactive menu loop
    while True:
        mode = get_mode_selection()
        source = get_video_source()

        try:
            if mode == "traffic":
                run_traffic_mode(source, args)
            elif mode == "surveillance":
                run_surveillance_mode(source, args)
        except KeyboardInterrupt:
            print_info("\nStopped by user.")
        except Exception as e:
            print_error(f"Error during execution: {e}")
            logger.exception("Runtime error")

        print("\n")
        again = input("  Return to main menu? (y/n): ").strip().lower()
        if again != "y":
            print_info("Exiting Smart Vision System. Goodbye!")
            break


if __name__ == "__main__":
    main()
