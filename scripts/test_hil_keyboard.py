#!/usr/bin/env python
"""
Test script for HIL Keyboard Handler

Run this to verify keyboard input works before integrating with evaluation.
"""

import sys
import time
from scripts.utils.hil_keyboard import HILKeyboardHandler


def main():
    print("="*60)
    print("HIL Keyboard Handler Test (Terminal/SSH)")
    print("="*60)
    print("\nThis will test keyboard input for HIL evaluation.")
    print("\nSupported keys (single press, no Enter needed):")
    print("  'i' - Toggle intervention mode (POLICY ↔ HUMAN)")
    print("  's' - Mark episode as SUCCESS")
    print("  'f' - Mark episode as FAILURE")
    print("  'q' - Quit early")
    print("\nTest will run for 30 seconds or until you press 'q'")
    print("="*60 + "\n")

    input("Press ENTER to start...")

    # Create and start handler
    handler = HILKeyboardHandler()
    handler.start()

    # Test loop
    start_time = time.time()
    test_duration = 30  # seconds

    print("✅ Test started! Try pressing keys...\n")

    while time.time() - start_time < test_duration:
        # Poll for keyboard input (IMPORTANT: call this regularly!)
        handler.poll()

        # Check for toggle
        if handler.is_intervention_toggled():
            mode = "HUMAN" if handler.is_intervention_active() else "POLICY"
            print(f"🔄 Mode toggled: {mode}")
            handler.reset_toggle()

        # Check for episode label
        label = handler.get_episode_label()
        if label:
            print(f"🏷️  Episode labeled: {label.upper()}")

        # Check for quit
        if handler.is_quit_requested():
            print("⚠️  Quit requested!")
            break

        time.sleep(0.05)

    # Cleanup
    handler.stop()

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    status = handler.get_status()
    print(f"  Intervention Mode: {'ACTIVE' if status['intervention_active'] else 'INACTIVE'}")
    print(f"  Test Duration: {int(time.time() - start_time)}s")
    print("\n✅ If you saw responses when pressing keys, it's working!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
