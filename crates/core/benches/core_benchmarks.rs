//! Criterion micro-benchmarks for `index-tts-rs-core` DSP, Frontend, and GPT sampler.

#![forbid(unsafe_code)]
#![warn(rust_2024_compatibility, missing_docs, missing_debug_implementations)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use index_tts_rs_core::{
    config::{Language, SamplingParams},
    dsp::{campplus_fbank, mel_spectrogram, resample_sinc_hann, w2v_features},
    frontend::Frontend,
    models::{GPT_MEL_VOCAB, SamplerRng, SeenTokenBitSet, resolve_model_dir, sample_next_token},
};

#[allow(clippy::cast_precision_loss)]
fn bench_dsp(c: &mut Criterion) {
    let mut group = c.benchmark_group("dsp");
    let samples_24k: Vec<f32> = (0..48_000)
        .map(|i| ((i as f32) * 0.04).sin() * 0.25)
        .collect();
    let samples_22k: Vec<f32> = (0..44_100)
        .map(|i| ((i as f32) * 0.04).sin() * 0.25)
        .collect();
    let samples_16k: Vec<f32> = (0..32_000)
        .map(|i| ((i as f32) * 0.04).sin() * 0.25)
        .collect();

    group.bench_function("resample_24k_to_22k_2s", |b| {
        b.iter(|| resample_sinc_hann(black_box(&samples_24k), 24_000, 22_050).ok());
    });
    group.bench_function("mel_spectrogram_2s", |b| {
        b.iter(|| mel_spectrogram(black_box(&samples_22k)).ok());
    });
    group.bench_function("campplus_fbank_2s", |b| {
        b.iter(|| campplus_fbank(black_box(&samples_16k)).ok());
    });
    group.bench_function("w2v_features_2s", |b| {
        b.iter(|| w2v_features(black_box(&samples_16k)).ok());
    });
    group.finish();
}

#[allow(clippy::cast_precision_loss)]
fn bench_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpt_sampler");
    let base_logits: Vec<f32> = (0..GPT_MEL_VOCAB)
        .map(|i| ((i as f32) * 0.013).cos() * 2.5)
        .collect();
    let mut seen = SeenTokenBitSet::empty();
    for tok in 10..60 {
        seen.insert(tok);
    }
    let params = SamplingParams::default();

    group.bench_function("sample_next_token_8194", |b| {
        let mut rng = SamplerRng::new(42);
        let mut logits = base_logits.clone();
        b.iter(|| {
            logits.copy_from_slice(&base_logits);
            sample_next_token(
                black_box(&mut logits),
                black_box(&seen),
                black_box(&params),
                &mut rng,
            )
            .ok()
        });
    });
    group.finish();
}

fn bench_frontend(c: &mut Criterion) {
    let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") else {
        return;
    };
    let Ok(frontend) = Frontend::from_model_dir(&dir, 600, true, None) else {
        return;
    };

    let mut group = c.benchmark_group("frontend");
    let text = "大家好，欢迎使用 IndexTTS-2.5 \
                语音合成引擎！请把<行|hang2>业报告发给<行|xing2>长，我们在 Apple Silicon 上实现了 \
                100% 纯 Rust 高性能流式推理。";
    group.bench_function("prepare_mixed_zh_en", |b| {
        b.iter(|| frontend.prepare(black_box(text), Language::ZhEn, 120));
    });
    group.finish();
}

criterion_group!(benches, bench_dsp, bench_sampler, bench_frontend);
criterion_main!(benches);
