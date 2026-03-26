"""
HIL Keyboard Input Handler for Terminal/SSH

Provides terminal-based keyboard input handling for Human-in-the-Loop evaluation.
Works in SSH sessions without requiring X11 or display servers.

This uses stdin polling which works in remote terminal environments.

Usage:
    from scripts.utils.hil_keyboard import HILKeyboardHandler

    handler = HILKeyboardHandler()
    handler.start()

    while running:
        handler.poll()  # Call this regularly to check for input

        if handler.is_intervention_toggled():
            print("Intervention mode toggled!")
            handler.reset_toggle()

        if handler.get_episode_label():
            label = handler.get_episode_label()
            print(f"Episode labeled: {label}")

        time.sleep(0.05)

    handler.stop()
"""

import sys
import threading
import time
import select
import tty
import termios
from typing import Optional


class HILKeyboardHandler:
    """
    Terminal-based keyboard input handler for HIL evaluation.

    This implementation uses stdin polling and works in SSH/terminal environments
    without requiring X11 or display servers.

    Supported Keys (single keypresses, no Enter needed):
        'b' - Start episode (when in idle phase)
        'i' - Toggle intervention mode (policy ↔ human control)
        's' - Finish current episode and move to next
        'f' - Mark episode as failed
        'ESC'/'q' - Quit/abort

    Note: This puts the terminal in raw mode, which may affect echo.
    The handler restores normal terminal mode on stop().
    """

    def __init__(self):
        # State flags
        self._intervention_toggle_requested = False
        self._intervention_mode_active = False

        # Episode control
        self._episode_start_requested = False  # 'b' pressed
        self._episode_end_requested = False    # 's' pressed
        self._episode_label: Optional[str] = None  # "success" or "failure"
        self._quit_requested = False

        # Terminal settings
        self._old_settings = None
        self._running = False
        self._lock = threading.Lock()

        # Input buffer
        self._input_buffer = ""

    def _setup_terminal(self):
        """Put terminal in raw mode for non-blocking input."""
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        except Exception as e:
            print(f"Warning: Could not set terminal to raw mode: {e}")
            print("Keyboard input may not work properly in this environment.")

    def _restore_terminal(self):
        """Restore normal terminal mode."""
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

    def _handle_key(self, key: str):
        """Handle a single key press."""
        with self._lock:
            if key == 'b' or key == 'B':
                self._episode_start_requested = True
                sys.stdout.write("\r▶️  Episode start requested\n")
                sys.stdout.flush()

            elif key == 'i' or key == 'I':
                self._intervention_mode_active = not self._intervention_mode_active
                self._intervention_toggle_requested = True
                mode = "HUMAN" if self._intervention_mode_active else "POLICY"
                sys.stdout.write(f"\r🔄 Mode switched to: {mode}\n")
                sys.stdout.flush()

            elif key == 's' or key == 'S':
                self._episode_end_requested = True
                self._episode_label = "success"
                sys.stdout.write("\r⏹️  Episode finished (SUCCESS)\n")
                sys.stdout.flush()

            elif key == 'f' or key == 'F':
                self._episode_label = "failure"
                sys.stdout.write("\r❌ Episode marked as FAILURE\n")
                sys.stdout.flush()

            elif key == 'q' or key == 'Q' or key == '\x1b' or key == '\x03':  # 'q', ESC, or Ctrl+C
                self._quit_requested = True
                sys.stdout.write("\r⚠️  Quit requested\n")
                sys.stdout.flush()

    def poll(self):
        """
        Check for keyboard input (non-blocking).

        Call this regularly from your main loop.
        """
        if not self._running:
            return

        # Check if there's data available to read
        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                key = sys.stdin.read(1)
                if key:
                    self._handle_key(key)
            except Exception:
                pass

    def start(self):
        """Start the keyboard handler."""
        if self._running:
            return

        self._running = True
        self._setup_terminal()

        sys.stdout.write("🎮 HIL Keyboard Handler started\n")
        sys.stdout.write("   Press 'b' to start episode\n")
        sys.stdout.write("   Press 'i' to toggle intervention mode (POLICY ↔ HUMAN)\n")
        sys.stdout.write("   Press 's' to finish episode\n")
        sys.stdout.write("   Press 'f' to mark episode as failure\n")
        sys.stdout.write("   Press 'ESC' or 'q' to quit\n")
        sys.stdout.flush()

    def stop(self):
        """Stop the keyboard handler and restore terminal."""
        if not self._running:
            return

        self._running = False
        self._restore_terminal()
        sys.stdout.write("🎮 HIL Keyboard Handler stopped\n")
        sys.stdout.flush()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def is_intervention_active(self) -> bool:
        """Check if intervention mode is currently active."""
        return self._intervention_mode_active

    def is_intervention_toggled(self) -> bool:
        """Check if intervention mode was toggled since last check."""
        return self._intervention_toggle_requested

    def reset_toggle(self):
        """Reset the toggle flag after handling."""
        self._intervention_toggle_requested = False

    def is_episode_start_requested(self) -> bool:
        """Check if episode start was requested (b key pressed)."""
        return self._episode_start_requested

    def reset_episode_start(self):
        """Reset the episode start flag after handling."""
        self._episode_start_requested = False

    def is_episode_end_requested(self) -> bool:
        """Check if episode end was requested (s key pressed)."""
        return self._episode_end_requested

    def reset_episode_end(self):
        """Reset the episode end flag after handling."""
        self._episode_end_requested = False

    def get_episode_label(self) -> Optional[str]:
        """
        Get and clear the episode label.

        Returns:
            "success", "failure", or None
        """
        label = self._episode_label
        self._episode_label = None
        return label

    def is_quit_requested(self) -> bool:
        """Check if quit was requested."""
        return self._quit_requested

    def get_status(self) -> dict:
        """Get current handler status."""
        return {
            "running": self._running,
            "intervention_active": self._intervention_mode_active,
            "episode_start_requested": self._episode_start_requested,
            "episode_end_requested": self._episode_end_requested,
            "episode_label": self._episode_label,
            "quit_requested": self._quit_requested,
        }


# -------------------------------------------------------------------------
# Test / Demo
# -------------------------------------------------------------------------

def test_keyboard_handler():
    """Test the keyboard handler with a simple demo."""
    import time

    handler = HILKeyboardHandler()
    handler.start()

    print("\n" + "="*50)
    print("HIL Keyboard Handler Test")
    print("="*50)
    print("Running for 30 seconds...")
    print("Try pressing: b, i, s, f, ESC")
    print("="*50 + "\n")

    start_time = time.time()
    while time.time() - start_time < 30:
        # Check for episode start
        if handler.is_episode_start_requested():
            print("✓ Episode start requested (b pressed)")
            handler.reset_episode_start()

        # Check for toggle
        if handler.is_intervention_toggled():
            mode = "HUMAN" if handler.is_intervention_active() else "POLICY"
            print(f"✓ Detected toggle: now in {mode} mode")
            handler.reset_toggle()

        # Check for episode end
        if handler.is_episode_end_requested():
            print("✓ Episode end requested (s pressed)")
            handler.reset_episode_end()

        # Check for episode label
        label = handler.get_episode_label()
        if label:
            print(f"✓ Episode labeled: {label}")

        # Check for quit
        if handler.is_quit_requested():
            print("✓ Quit requested, exiting early...")
            break

        # Print status every 5 seconds
        elapsed = int(time.time() - start_time)
        if elapsed % 5 == 0 and elapsed > 0:
            status = handler.get_status()
            print(f"\n📊 Status (t={elapsed}s):")
            print(f"   Intervention: {'ACTIVE' if status['intervention_active'] else 'INACTIVE'}")
            print(f"   Running: {status['running']}\n")

        time.sleep(0.1)

    handler.stop()
    print("\n" + "="*50)
    print("Test completed!")
    print("="*50)


if __name__ == "__main__":
    test_keyboard_handler()
