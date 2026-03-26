#!/usr/bin/env python3
"""
Test MoE-SmolVLA Policy Inference

Tests the MoE policy with a single observation to verify:
1. Component loading (VLM, experts, router)
2. Router inference
3. Expert selection
4. Action generation
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_policy.moe_smolvla_policy import MoESmolVLAPolicy


def create_mock_observation(image_size=(480, 640)) -> dict:
    """Create a mock observation for testing."""
    obs = {}

    # RGB images (random data for testing)
    for cam in ["top", "left", "right"]:
        key = f"observation.images.{cam}_rgb"
        obs[key] = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)

    # Joint state (12 DOF for dual-arm)
    obs["observation.state"] = np.random.randn(12).astype(np.float32)

    return obs


def test_moe_policy():
    """Test MoE policy end-to-end."""
    print("=" * 60)
    print("MoE-SmolVLA Policy Test")
    print("=" * 60)

    # Test configuration
    test_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "router_checkpoint": "outputs/train/router/checkpoints/best/router.pt",
    }

    print(f"\n[Test Config]")
    print(f"  Device: {test_config['device']}")
    print(f"  Router: {test_config['router_checkpoint']}")

    # Initialize policy
    print(f"\n[1/4] Initializing MoE Policy...")
    try:
        policy = MoESmolVLAPolicy(**test_config)
        print("  ✅ Policy initialized successfully")
    except Exception as e:
        print(f"  ❌ Policy initialization failed: {e}")
        return False

    # Test router inference
    print(f"\n[2/4] Testing Router Inference...")
    try:
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        garment_type, confidence = policy._route_image(mock_image)
        print(f"  ✅ Router prediction: {garment_type} (confidence: {confidence:.3f})")
    except Exception as e:
        print(f"  ❌ Router inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test expert availability
    print(f"\n[3/4] Checking Expert Availability...")
    print(f"  Available experts: {list(policy.experts.keys())}")
    print(f"  Missing experts: {policy.missing_experts}")

    if len(policy.experts) == 0:
        print("  ❌ No experts available!")
        return False

    # Test action selection
    print(f"\n[4/4] Testing Action Selection...")
    try:
        observation = create_mock_observation()
        action = policy.select_action(observation)

        print(f"  ✅ Action generated successfully")
        print(f"  Action shape: {action.shape}")
        print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
        print(f"  Selected expert: {policy.selected_expert}")

    except Exception as e:
        print(f"  ❌ Action selection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 60)
    print("Test Results: ✅ ALL TESTS PASSED")
    print("=" * 60)
    print(f"\nMoE Policy is ready for inference!")
    print(f"  - Loaded {len(policy.experts)} experts")
    print(f"  - Router supports {len(policy.GARMENT_TYPES)} garment types")
    print(f"  - Device: {test_config['device']}")

    return True


def test_expert_switching():
    """Test switching between different experts."""
    print("\n" + "=" * 60)
    print("Expert Switching Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️ Skipping (CUDA not available)")
        return True

    policy = MoESmolVLAPolicy(device="cuda")

    print(f"\nAvailable experts: {list(policy.experts.keys())}")

    # Test with different garment types
    for target_expert in list(policy.experts.keys())[:3]:  # Test first 3
        print(f"\n--- Testing {target_expert} ---")

        # Create observation
        observation = create_mock_observation()

        # Force selection (bypass router for testing)
        policy.selected_expert = target_expert

        try:
            action = policy.select_action(observation)
            print(f"  ✅ Action shape: {action.shape}")
            print(f"  Action mean: {action.mean():.3f}, std: {action.std():.3f}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return False

    print("\n✅ Expert switching test passed!")
    return True


if __name__ == "__main__":
    print("Starting MoE Policy Tests...\n")

    # Run basic test
    success = test_moe_policy()

    if success:
        # Run expert switching test
        test_expert_switching()

    print("\nTest complete!")
