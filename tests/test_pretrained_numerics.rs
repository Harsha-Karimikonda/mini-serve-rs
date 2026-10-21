use candle_core::{DType, Device, Tensor};
use rlse::backends::qwen2::{Config, ModelForCausalLM};

// Absolute error catches outliers; RMS error prevents widespread drift from
// hiding behind that bound. Never accept NaN/Inf or a changed greedy decision.
fn compare(label: &str, actual: &Tensor, expected: &Tensor, max: f32, rms: f32) {
    assert_eq!(actual.dims(), expected.dims(), "{label}: shape");
    let values = |t: &Tensor| {
        t.to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let a = values(actual);
    let b = values(expected);
    assert!(!a.is_empty());
    assert!(
        a.iter().chain(&b).all(|x| x.is_finite()),
        "{label}: nonfinite logits"
    );
    let errors: Vec<f32> = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).collect();
    let max_error = errors.iter().copied().fold(0., f32::max);
    let rms_error = (errors.iter().map(|e| e * e).sum::<f32>() / a.len() as f32).sqrt();
    println!("{label}: max_abs={max_error:.6} rms={rms_error:.6}");
    assert!(
        max_error <= max && rms_error <= rms,
        "{label}: numeric drift exceeds max={max}, rms={rms}"
    );
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(argmax(&a), argmax(&b), "{label}: greedy decision");
}

#[test]
fn numeric_comparison_accepts_bounded_roundoff() {
    let a = Tensor::new(&[1f32, 2., 3.], &Device::Cpu).unwrap();
    let b = Tensor::new(&[1.001f32, 2., 3.], &Device::Cpu).unwrap();
    compare("roundoff", &a, &b, 0.002, 0.001);
}

#[test]
#[should_panic(expected = "nonfinite logits")]
fn numeric_comparison_rejects_nonfinite_logits() {
    let a = Tensor::new(&[f32::NAN, 2.], &Device::Cpu).unwrap();
    compare("nan", &a, &a, 1., 1.);
}

#[test]
#[should_panic(expected = "numeric drift")]
fn numeric_comparison_rejects_drift() {
    let a = Tensor::new(&[1f32, 2.], &Device::Cpu).unwrap();
    let b = Tensor::new(&[1f32, 3.], &Device::Cpu).unwrap();
    compare("drift", &a, &b, 0.1, 0.1);
}

#[test]
#[should_panic(expected = "greedy decision")]
fn numeric_comparison_rejects_changed_greedy_decision() {
    let a = Tensor::new(&[1.001f32, 1.], &Device::Cpu).unwrap();
    let b = Tensor::new(&[1f32, 1.001], &Device::Cpu).unwrap();
    compare("argmax", &a, &b, 0.01, 0.01);
}

#[test]
fn cpu_batched_decode_and_prefix_logits_match_single_execution() -> candle_core::Result<()> {
    let config: Config = serde_json::from_value(serde_json::json!({
        "hidden_size": 32, "intermediate_size": 64, "vocab_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "max_position_embeddings": 128,
        "head_dim": 8, "hidden_act": "silu", "attention_bias": false
    }))
    .unwrap();
    let vars = candle_nn::VarMap::new();
    let mut model = ModelForCausalLM::new(
        &config,
        candle_nn::VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
    )?;
    // Replace random initialization with stable, nonzero fixture weights.
    for (name, var) in vars.data().lock().unwrap().iter() {
        let offset: usize = name.bytes().map(usize::from).sum();
        let values: Vec<f32> = (0..var.elem_count())
            .map(|i| {
                if var.rank() == 1 {
                    1.
                } else {
                    ((i + offset) as f32 * 0.17).sin() * 0.1
                }
            })
            .collect();
        var.set(&Tensor::from_vec(values, var.shape(), &Device::Cpu)?)?;
    }
    model.set_kv_capacity(128)?;
    let fixture = serde_json::json!({"cases": [
        {"name": "short", "prompt_token_ids": [1, 2, 3]},
        {"name": "boundary", "prompt_token_ids": (1..=17).collect::<Vec<u32>>()},
        {"name": "long", "prompt_token_ids": (1..=33).collect::<Vec<u32>>()}
    ]});
    exercise_paths(&model, &model, &fixture, &Device::Cpu)
}

#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
#[ignore = "requires explicit GPU and pinned local Qwen3 weights; CPU/F32 versus GPU/F16 logits"]
fn pretrained_metal_logits_match_cpu_and_execution_paths() -> Result<(), Box<dyn std::error::Error>>
{
    use candle_nn::VarBuilder;
    use sha2::{Digest, Sha256};
    use std::{io::Read, path::PathBuf};

    let requested = std::env::var("RLSE_GPU_DEVICE").unwrap_or_default();
    assert!(
        matches!(requested.as_str(), "metal" | "cuda"),
        "set RLSE_GPU_DEVICE=metal or RLSE_GPU_DEVICE=cuda explicitly"
    );
    let dir = PathBuf::from(
        std::env::var("RLSE_GPU_MODEL_DIR").unwrap_or_else(|_| "models/Qwen3-0.6B".into()),
    );
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/generation/qwen3-reference.json"))?;
    // Stream hashes so checking the 1.4 GB artifact does not duplicate it in RAM.
    for (name, expected) in fixture["hashes"].as_object().unwrap() {
        let mut file = std::fs::File::open(dir.join(name))?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        assert_eq!(
            hex::encode(hash.finalize()),
            expected.as_str().unwrap(),
            "artifact {name}"
        );
    }
    let device = match requested.as_str() {
        "metal" => Device::new_metal(0)?,
        "cuda" => Device::new_cuda(0)?,
        _ => unreachable!(),
    };
    assert!(matches!(
        (requested.as_str(), &device),
        ("metal", Device::Metal(_)) | ("cuda", Device::Cuda(_))
    ));
    println!("ACTUAL_DEVICE={device:?}; model={}; revision={}; CPU/F32 reference, GPU/F16; greedy, 3 teacher-forced decode steps", dir.display(), fixture["revision"]);
    let config: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let load = |device: &Device, dtype| -> candle_core::Result<ModelForCausalLM> {
        // SAFETY: pinned local artifacts must remain unmodified during this test.
        #[allow(unsafe_code)]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, device)?
        };
        let mut model = ModelForCausalLM::new(&config, vb)?;
        model.set_kv_capacity(256)?;
        Ok(model)
    };
    let cpu = load(&Device::Cpu, DType::F32)?;
    let gpu = load(&device, DType::F16)?;
    exercise_paths(&cpu, &gpu, &fixture, &device)?;
    Ok(())
}

fn exercise_paths(
    cpu: &ModelForCausalLM,
    gpu: &ModelForCausalLM,
    fixture: &serde_json::Value,
    device: &Device,
) -> candle_core::Result<()> {
    let (cross_max, cross_rms, path_max, path_rms) = if matches!(device, Device::Cpu) {
        (1e-4, 1e-5, 1e-4, 1e-5)
    } else {
        (0.25, 0.04, 0.125, 0.03)
    };
    let mut singles = Vec::new();
    let mut batches = Vec::new();
    let mut references = Vec::new();
    let mut positions = Vec::new();
    let mut inputs = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let ids: Vec<u32> = serde_json::from_value(case["prompt_token_ids"].clone()).unwrap();
        let mut reference = cpu.clone();
        let expected =
            reference.forward(&Tensor::new(ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?, 0)?;
        let mut single = gpu.clone();
        let actual = single.forward(&Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?, 0)?;
        compare(
            &format!("{} prefill CPU/F32", case["name"]),
            &actual,
            &expected,
            cross_max,
            cross_rms,
        );

        // Reuse a real prefix at a block boundary, then prefill the suffix.
        let split = 16.min(ids.len() - 1);
        let mut prime = gpu.clone();
        prime.forward(&Tensor::new(&ids[..split], device)?.unsqueeze(0)?, 0)?;
        let snapshot: Vec<_> = prime
            .get_kv_snapshot()
            .into_iter()
            .map(Option::unwrap)
            .collect();
        let mut restored = gpu.clone();
        restored.set_kv_snapshot(&snapshot)?;
        let resumed =
            restored.forward(&Tensor::new(&ids[split..], device)?.unsqueeze(0)?, split)?;
        compare(
            &format!("{} restored prefix", case["name"]),
            &resumed,
            &actual,
            path_max,
            path_rms,
        );
        inputs.push(expected.flatten_all()?.argmax(0)?.to_scalar::<u32>()?);
        positions.push(ids.len());
        batches.push(single.clone());
        singles.push(single);
        references.push(reference);
    }
    for step in 0..3 {
        let mut models: Vec<_> = batches.iter_mut().collect();
        let batched =
            ModelForCausalLM::step_decode_batch(&mut models, &inputs, &positions, device)?;
        for i in 0..singles.len() {
            let actual = singles[i]
                .forward(&Tensor::new(&[[inputs[i]]], device)?, positions[i])?
                .flatten_all()?;
            let expected = references[i]
                .forward(&Tensor::new(&[[inputs[i]]], &Device::Cpu)?, positions[i])?
                .flatten_all()?;
            compare(
                &format!("case {i} decode {step} CPU/F32"),
                &actual,
                &expected,
                cross_max,
                cross_rms,
            );
            compare(
                &format!("case {i} decode {step} batched"),
                &batched.get(i)?,
                &actual,
                path_max,
                path_rms,
            );
            inputs[i] = expected.argmax(0)?.to_scalar::<u32>()?;
            positions[i] += 1;
        }
    }
    Ok(())
}
