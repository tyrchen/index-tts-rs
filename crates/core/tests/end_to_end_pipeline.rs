//! End-to-end integration tests for `index-tts-rs-core`.

#![forbid(unsafe_code)]
#![warn(rust_2024_compatibility, missing_docs, missing_debug_implementations)]

use std::path::PathBuf;

use index_tts_rs_core::{
    EngineConfig, IndexTtsEngine, Language, SamplingParams, SpeakerContext, SpeakerSource,
    SynthesisRequest, SynthesisText, VoiceId, models::resolve_model_dir,
};

#[test]
fn test_should_register_voice_and_synthesize_wav_end_to_end() {
    if resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx").is_err() {
        return;
    }

    let Ok(temp_dir) = tempfile::tempdir() else {
        return;
    };
    let voices_dir = temp_dir.path().join("voices");
    let out_wav = temp_dir.path().join("output.wav");

    let cfg = EngineConfig::builder().voices_dir(Some(voices_dir)).build();
    let engine_res = IndexTtsEngine::start(cfg);
    assert!(engine_res.is_ok(), "engine start failed: {engine_res:?}");
    let Ok(engine) = engine_res else {
        return;
    };

    let golden_voice_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("golden_speaker.itts_voice");
    let Ok(golden_ctx) = SpeakerContext::load_from_path(&golden_voice_path) else {
        return;
    };
    let Ok(voice_id) = VoiceId::new("golden_demo") else {
        return;
    };

    let reg_res = engine.voice_store().register(voice_id.clone(), golden_ctx);
    assert!(reg_res.is_ok());

    let listed = engine.voice_store().list().unwrap_or_default();
    assert_eq!(listed.len(), 1);

    let Ok(text) = SynthesisText::new("欢迎使用 IndexTTS-2.5 Rust 原生语音引擎！")
    else {
        return;
    };
    let sampling = SamplingParams::builder()
        .greedy(true)
        .max_mel_tokens(24)
        .build();
    let req = SynthesisRequest::builder()
        .text(text)
        .speaker(SpeakerSource::VoiceId(voice_id))
        .language(Language::Zh)
        .sampling(sampling)
        .build();

    let result_res = engine.synthesize_blocking(req);
    assert!(result_res.is_ok(), "synthesis failed: {result_res:?}");
    if let Ok(res) = result_res {
        assert_eq!(res.sample_rate, 22_050);
        assert!(res.duration_secs > 0.0);
        assert!(res.save_wav(&out_wav).is_ok());
        assert!(out_wav.is_file());
    }

    assert!(engine.stop().is_ok());
}
