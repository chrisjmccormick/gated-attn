# Length Scheduling Training

This document describes the length scheduling training approach implemented in `train_length.py`. This feature allows you to train models with progressively increasing sequence lengths, which can be more memory-efficient and help models learn to handle longer sequences.

## Overview

Length scheduling training divides the training process into multiple phases, each with a different sequence length:

1. **Short Sequence Phase (80%)**: Train at 512 tokens for most of the training
2. **Medium Sequence Phase (16%)**: Train at 1024 tokens (model's native max)
3. **Long Sequence Phase (4%)**: Train at 2048 tokens with RoPE scaling

## Configuration

Add the following to your training configuration JSON file:

```json
{
  "pre_train": {
    "length_scheduling": {
      "enabled": true,
      "phases": [
        {
          "name": "short_sequence",
          "seq_length": 512,
          "steps": 10000,
          "description": "80% of training at 512 tokens"
        },
        {
          "name": "medium_sequence", 
          "seq_length": 1024,
          "steps": 2000,
          "description": "16% of training at 1024 tokens"
        },
        {
          "name": "long_sequence",
          "seq_length": 2048,
          "steps": 500,
          "description": "4% of training at 2048 tokens with RoPE scaling",
          "rope_scaling": {
            "type": "linear",
            "factor": 2.0
          }
        }
      ]
    }
  }
}
```

### Configuration Parameters

- `enabled`: Boolean to enable/disable length scheduling
- `phases`: Array of training phases
  - `name`: Descriptive name for the phase
  - `seq_length`: Sequence length for this phase
  - `steps`: Number of training steps in this phase
  - `description`: Optional description
  - `rope_scaling`: Optional RoPE scaling configuration
    - `type`: Scaling type ("linear" or "dynamic")
    - `factor`: Scaling factor (e.g., 2.0 for 2x longer sequences)

## Usage

### Basic Usage

```bash
python train_length.py --config configs/gpt-2_mla_c4_length.json
```

### Disabling Length Scheduling

To disable length scheduling and use standard training:

```json
{
  "pre_train": {
    "length_scheduling": {
      "enabled": false
    }
  }
}
```

## How It Works

### LengthScheduler Class

The `LengthScheduler` class manages the transition between training phases:

- Tracks current training step and determines which phase is active
- Provides current sequence length and RoPE scaling configuration
- Handles phase transitions automatically

### LengthScheduledDataCollator

A custom data collator that:

- Dynamically truncates sequences to the current phase's length
- Communicates with the scheduler to get the current sequence length
- Ensures efficient memory usage during training

### LengthSchedulingCallback

A training callback that:

- Monitors training steps and triggers phase transitions
- Updates model's RoPE scaling configuration when needed
- Logs sequence length changes to Weights & Biases
- Recreates RoPE embeddings with new scaling factors

### RoPE Scaling

When transitioning to longer sequences (e.g., 2048 tokens), the model uses RoPE scaling:

- **Linear Scaling**: Divides frequencies by the scaling factor
- **Dynamic Scaling**: Adjusts frequencies based on sequence length
- Automatically recreates RoPE embeddings with new scaling

## Benefits

1. **Memory Efficiency**: Start with shorter sequences to reduce memory usage
2. **Progressive Learning**: Models learn to handle longer sequences gradually
3. **Budget-Friendly**: More efficient training for longer sequence models
4. **Flexible Configuration**: Easy to adjust phase lengths and sequence lengths

## Monitoring

The training script provides detailed logging:

- Phase transition notifications
- Sequence length changes
- RoPE scaling updates
- Weights & Biases integration for tracking metrics

## Example Output

```
=== Length Scheduling Configuration ===
Length scheduling is ENABLED
Length Scheduler initialized with 3 phases:
  Phase 1: short_sequence - 512 tokens for 10000 steps
  Phase 2: medium_sequence - 1024 tokens for 2000 steps
  Phase 3: long_sequence - 2048 tokens for 500 steps
Length scheduling initialized successfully

=== Length Scheduling Update at Step 10000 ===
Sequence length: 512 -> 1024
Transitioning to phase 2: medium_sequence
  Sequence length: 1024

=== Length Scheduling Update at Step 12000 ===
Sequence length: 1024 -> 2048
Transitioning to phase 3: long_sequence
  Sequence length: 2048
Updating RoPE scaling: {'type': 'linear', 'factor': 2.0}
Updated max_position_embeddings to 2048
Recreated RoPE embeddings with scaling factor 2.0
```

## Testing

Run the test script to verify the implementation:

```bash
python test_length_scheduling.py
```

This will test the LengthScheduler class and RoPE scaling configurations without requiring a full training setup.

## Troubleshooting

### Common Issues

1. **Sequence length exceeds max_seq_length**: The script will warn if any phase uses a sequence length longer than the configured maximum.

2. **Phase steps don't match total steps**: The scheduler will automatically adjust the final phase to cover remaining steps.

3. **RoPE scaling not working**: Ensure the scaling configuration is properly formatted and the model supports RoPE scaling.

### Memory Considerations

- Start with shorter sequences to reduce initial memory usage
- Monitor GPU memory usage during phase transitions
- Consider reducing batch size for longer sequence phases

## Future Improvements

- Support for more RoPE scaling types (e.g., NTK-aware scaling)
- Automatic phase length optimization
- Support for different learning rates per phase
- Integration with other training optimizations
