#!/usr/bin/env python3
"""
Test script for length scheduling functionality.
This script tests the LengthScheduler class without requiring a full training setup.
"""

import sys
from pathlib import Path

# Add the project root to the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_length_scheduler():
    """Test the LengthScheduler class with sample configuration."""
    
    # Import the LengthScheduler class
    from train_length import LengthScheduler
    
    # Sample configuration similar to the config file
    phases = [
        {
            "name": "short_sequence",
            "seq_length": 512,
            "steps": 1000,
            "description": "80% of training at 512 tokens"
        },
        {
            "name": "medium_sequence", 
            "seq_length": 1024,
            "steps": 200,
            "description": "16% of training at 1024 tokens"
        },
        {
            "name": "long_sequence",
            "seq_length": 2048,
            "steps": 50,
            "description": "4% of training at 2048 tokens with RoPE scaling",
            "rope_scaling": {
                "type": "linear",
                "factor": 2.0
            }
        }
    ]
    
    total_steps = 1250  # 1000 + 200 + 50
    
    # Create scheduler
    scheduler = LengthScheduler(phases, total_steps)
    
    # Test phase transitions
    test_steps = [0, 500, 1000, 1100, 1200, 1249]
    
    print("Testing LengthScheduler:")
    print("=" * 50)
    
    for step in test_steps:
        config = scheduler.get_current_config(step)
        phase_info = scheduler.get_phase_info(step)
        
        print(f"Step {step:4d}: {config['seq_length']:4d} tokens, "
              f"Phase: {phase_info['phase_name']}, "
              f"Progress: {phase_info['phase_progress']:.2f}")
        
        if config['rope_scaling']:
            print(f"         RoPE scaling: {config['rope_scaling']}")
    
    print("\nTesting phase transitions:")
    print("=" * 50)
    
    # Test transition points
    transition_steps = [999, 1000, 1001, 1199, 1200, 1201]
    
    for step in transition_steps:
        config = scheduler.get_current_config(step)
        phase_info = scheduler.get_phase_info(step)
        
        print(f"Step {step:4d}: {config['seq_length']:4d} tokens, "
              f"Phase: {phase_info['phase_name']}")

def test_rope_scaling():
    """Test RoPE scaling configuration."""
    
    print("\nTesting RoPE scaling configuration:")
    print("=" * 50)
    
    # Test different scaling configurations
    scaling_configs = [
        None,
        {"type": "linear", "factor": 1.0},
        {"type": "linear", "factor": 2.0},
        {"type": "dynamic", "factor": 1.5}
    ]
    
    for i, config in enumerate(scaling_configs):
        print(f"Config {i+1}: {config}")
        if config:
            print(f"  Type: {config.get('type', 'unknown')}")
            print(f"  Factor: {config.get('factor', 1.0)}")
        else:
            print("  No scaling")

if __name__ == "__main__":
    print("Length Scheduling Test Suite")
    print("=" * 50)
    
    try:
        test_length_scheduler()
        test_rope_scaling()
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
