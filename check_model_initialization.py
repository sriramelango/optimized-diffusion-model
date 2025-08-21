#!/usr/bin/env python3
"""
Check model initialization during training to verify all optimized parameters.
"""

import sys
import os
sys.path.insert(0, 'Reflected-Diffusion')

import torch
import models.utils as mutils
from omegaconf import OmegaConf

def check_model_initialization():
    """Check that the model is initialized with correct optimized parameters."""
    
    print("=" * 80)
    print("CHECKING MODEL INITIALIZATION FOR TRAINING")
    print("=" * 80)
    
    # Load the configs properly
    model_config = OmegaConf.load('Reflected-Diffusion/configs/model/unet1d_gto.yaml')
    data_config = OmegaConf.load('Reflected-Diffusion/configs/data/gto_halo_1d.yaml')
    train_config = OmegaConf.load('Reflected-Diffusion/configs/train_gto_1d.yaml')
    
    print("📋 Loading training configuration...")
    print(f"  Model: {model_config.name}")
    print(f"  Data: {data_config.dataset}")
    print(f"  Training batch size: {train_config.training.batch_size}")
    
    # Create full config like training does
    config = OmegaConf.create({
        'model': model_config,
        'data': data_config,
        'training': train_config.training,
        'optim': train_config.optim,
        'sde': train_config.sde,
        'eval': train_config.eval,
        'ngpus': train_config.ngpus,
        'dataroot': train_config.dataroot
    })
    
    # Create the model exactly as training does
    print("\n🏗️  Creating model...")
    model = mutils.create_model(config)
    
    print(f"✅ Model created: {type(model).__name__}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Check critical architectural parameters
    print("\n🔍 VERIFYING CRITICAL OPTIMIZATIONS:")
    print("-" * 50)
    
    # Check attention configuration
    has_attention = False
    attention_heads = None
    attention_dim_head = None
    
    for name, module in model.named_modules():
        if 'attn' in name.lower() or 'attention' in name.lower():
            has_attention = True
            if hasattr(module, 'heads'):
                attention_heads = module.heads
            if hasattr(module, 'dim_head'):
                attention_dim_head = module.dim_head
            print(f"  🔍 Found attention module: {name}")
            print(f"     Type: {type(module).__name__}")
            if hasattr(module, 'heads'):
                print(f"     Heads: {module.heads}")
            if hasattr(module, 'dim_head'):
                print(f"     Dim per head: {module.dim_head}")
    
    # Check ResNet block groups
    resnet_groups = None
    for name, module in model.named_modules():
        if 'resnet' in name.lower() and hasattr(module, 'block1'):
            if hasattr(module.block1, 'norm') and hasattr(module.block1.norm, 'num_groups'):
                resnet_groups = module.block1.norm.num_groups
                print(f"  🔍 ResNet block groups: {resnet_groups}")
                break
    
    # Check sinusoidal embedding
    has_learned_sinusoidal = False
    for name, module in model.named_modules():
        if 'sinusoidal' in name.lower():
            print(f"  🔍 Found sinusoidal module: {name} -> {type(module).__name__}")
            if 'learned' in type(module).__name__.lower():
                has_learned_sinusoidal = True
    
    # Check class embedding structure
    class_embedding_dims = []
    for name, module in model.named_modules():
        if 'classes_mlp' in name and isinstance(module, torch.nn.Linear):
            class_embedding_dims.append((module.in_features, module.out_features))
    
    print(f"  🔍 Class embedding structure: {class_embedding_dims}")
    
    # Verification summary
    print("\n📊 OPTIMIZATION VERIFICATION SUMMARY:")
    print("-" * 50)
    
    checks = [
        ("🔥 Attention heads = 8", attention_heads == 8),
        ("🔥 Attention dim head = 32", attention_dim_head == 32),
        ("🔥 ResNet block groups = 8", resnet_groups == 8),
        ("🔥 Learned sinusoidal conditioning", has_learned_sinusoidal),
        ("🔥 Deep class embeddings", len(class_embedding_dims) >= 3),
    ]
    
    all_good = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name:<35} | {status}")
        if not passed:
            all_good = False
    
    # Test forward pass with realistic inputs
    print("\n🧪 TESTING FORWARD PASS:")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create test inputs matching training data
    batch_size = 4  # Small batch for testing
    channels = 3
    seq_length = 22
    
    x = torch.randn(batch_size, channels, seq_length, device=device)
    t = torch.rand(batch_size, device=device)  
    class_labels = torch.randn(batch_size, 1, device=device)
    
    print(f"  Input shapes:")
    print(f"    x (sequence): {x.shape}")
    print(f"    t (time): {t.shape}")
    print(f"    class_labels: {class_labels.shape}")
    
    try:
        with torch.no_grad():
            output = model(x, t, class_labels=class_labels)
        
        print(f"  ✅ Forward pass successful!")
        print(f"    Output shape: {output.shape}")
        print(f"    Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"    Output device: {output.device}")
        
        # Test with conditional dropout (training mode)
        model.train()
        output_train = model(x, t, class_labels=class_labels, cond_drop_prob=0.1)
        print(f"  ✅ Training mode forward pass successful!")
        print(f"    Output shape: {output_train.shape}")
        
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        all_good = False
    
    # Test data compatibility
    print("\n📊 TESTING DATA COMPATIBILITY:")
    print("-" * 50)
    
    try:
        import datasets
        train_ds, eval_ds = datasets.get_dataset(config)
        
        train_iter = iter(train_ds)
        batch = next(train_iter)
        
        batch_data = batch[0]
        batch_labels = batch[1]
        
        print(f"  ✅ Dataset loading successful!")
        print(f"    Batch data shape: {batch_data.shape}")
        print(f"    Batch labels shape: {batch_labels.shape}")
        print(f"    Data range: [{batch_data.min():.3f}, {batch_data.max():.3f}]")
        print(f"    Labels range: [{batch_labels.min():.3f}, {batch_labels.max():.3f}]")
        
        # Test model with real data
        if batch_data.shape[0] > 0:
            real_x = batch_data[:min(2, batch_data.shape[0])].to(device)
            real_labels = batch_labels[:min(2, batch_labels.shape[0])].to(device)
            real_t = torch.rand(real_x.shape[0], device=device)
            
            with torch.no_grad():
                real_output = model(real_x, real_t, class_labels=real_labels)
            
            print(f"  ✅ Real data forward pass successful!")
            print(f"    Real output shape: {real_output.shape}")
        
    except Exception as e:
        print(f"  ❌ Data compatibility test failed: {e}")
        all_good = False
    
    print("\n" + "=" * 80)
    
    if all_good:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Model initialization is PERFECT for training")
        print("✅ All optimized hyperparameters are correctly applied")
        print("✅ Forward pass works with training data")
        print("✅ Ready to start training with optimized configuration!")
    else:
        print("⚠️  Some issues found. Please check the details above.")
    
    print("=" * 80)
    
    return all_good

if __name__ == "__main__":
    success = check_model_initialization()
    print(f"\n🚀 Model initialization check: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
