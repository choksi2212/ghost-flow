use ghostflow_nn::prelude::*;
use ghostflow_core::Tensor;

fn main() {
    println!("=== GhostFlow v0.2.0 New Features Demo ===\n");

    // Test Conv1d
    println!("1. Testing Conv1d...");
    let conv1d = Conv1d::new(3, 16, 3, 1, 1);
    let input_1d = Tensor::randn(&[2, 3, 32]);
    let output_1d = conv1d.forward(&input_1d);
    println!("   Input shape: {:?}", input_1d.dims());
    println!("   Output shape: {:?}", output_1d.dims());
    assert_eq!(output_1d.dims(), &[2, 16, 32]);
    println!("   ✓ Conv1d works!\n");

    // Test Conv3d
    println!("2. Testing Conv3d...");
    let conv3d = Conv3d::new(3, 16, (3, 3, 3), (1, 1, 1), (1, 1, 1));
    let input_3d = Tensor::randn(&[2, 3, 8, 8, 8]);
    let output_3d = conv3d.forward(&input_3d);
    println!("   Input shape: {:?}", input_3d.dims());
    println!("   Output shape: {:?}", output_3d.dims());
    assert_eq!(output_3d.dims(), &[2, 16, 8, 8, 8]);
    println!("   ✓ Conv3d works!\n");

    // Test TransposeConv2d
    println!("3. Testing TransposeConv2d...");
    let tconv = TransposeConv2d::new(16, 3, (4, 4), (2, 2), (1, 1), (0, 0));
    let input_tconv = Tensor::randn(&[2, 16, 8, 8]);
    let output_tconv = tconv.forward(&input_tconv);
    println!("   Input shape: {:?}", input_tconv.dims());
    println!("   Output shape: {:?}", output_tconv.dims());
    println!("   ✓ TransposeConv2d works!\n");

    // Test GroupNorm
    println!("4. Testing GroupNorm...");
    let group_norm = GroupNorm::new(4, 16);
    let input_gn = Tensor::randn(&[2, 16, 8, 8]);
    let output_gn = group_norm.forward(&input_gn);
    println!("   Input shape: {:?}", input_gn.dims());
    println!("   Output shape: {:?}", output_gn.dims());
    assert_eq!(output_gn.dims(), input_gn.dims());
    println!("   ✓ GroupNorm works!\n");

    // Test InstanceNorm
    println!("5. Testing InstanceNorm...");
    let instance_norm = InstanceNorm::new(16);
    let input_in = Tensor::randn(&[2, 16, 8, 8]);
    let output_in = instance_norm.forward(&input_in);
    println!("   Input shape: {:?}", input_in.dims());
    println!("   Output shape: {:?}", output_in.dims());
    assert_eq!(output_in.dims(), input_in.dims());
    println!("   ✓ InstanceNorm works!\n");

    // Test new activation functions
    println!("6. Testing new activation functions...");
    
    let test_input = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
    
    let swish = Swish::new(1.0);
    let output_swish = swish.forward(&test_input);
    println!("   Swish output: {:?}", output_swish.data_f32());
    
    let mish = Mish::new();
    let output_mish = mish.forward(&test_input);
    println!("   Mish output: {:?}", output_mish.data_f32());
    
    let elu = ELU::new(1.0);
    let output_elu = elu.forward(&test_input);
    println!("   ELU output: {:?}", output_elu.data_f32());
    
    let selu = SELU::new();
    let output_selu = selu.forward(&test_input);
    println!("   SELU output: {:?}", output_selu.data_f32());
    
    let softplus = Softplus::default();
    let output_softplus = softplus.forward(&test_input);
    println!("   Softplus output: {:?}", output_softplus.data_f32());
    
    println!("   ✓ All new activations work!\n");

    // Test new loss functions
    println!("7. Testing new loss functions...");
    
    let logits = Tensor::from_slice(&[2.0f32, 1.0, 0.1, 0.5, 3.0, 0.2], &[2, 3]).unwrap();
    let targets = Tensor::from_slice(&[0.0f32, 2.0], &[2]).unwrap();
    
    let focal = ghostflow_nn::loss::focal_loss(&logits, &targets, 1.0, 2.0);
    println!("   Focal Loss: {:.4}", focal.data_f32()[0]);
    
    let x1 = Tensor::randn(&[4, 128]);
    let x2 = Tensor::randn(&[4, 128]);
    let labels = Tensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0], &[4]).unwrap();
    
    let contrastive = ghostflow_nn::loss::contrastive_loss(&x1, &x2, &labels, 1.0);
    println!("   Contrastive Loss: {:.4}", contrastive.data_f32()[0]);
    
    let anchor = Tensor::randn(&[4, 128]);
    let positive = Tensor::randn(&[4, 128]);
    let negative = Tensor::randn(&[4, 128]);
    
    let triplet = ghostflow_nn::loss::triplet_margin_loss(&anchor, &positive, &negative, 0.5);
    println!("   Triplet Loss: {:.4}", triplet.data_f32()[0]);
    
    let pred = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let target = Tensor::from_slice(&[1.1f32, 2.2, 2.8, 4.5], &[4]).unwrap();
    
    let huber = ghostflow_nn::loss::huber_loss(&pred, &target, 1.0);
    println!("   Huber Loss: {:.4}", huber.data_f32()[0]);
    
    println!("   ✓ All new loss functions work!\n");

    println!("=== All v0.2.0 features working correctly! ===");
}
