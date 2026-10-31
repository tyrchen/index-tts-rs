PYTHON_BIN ?= /Users/tchen/.cache/uv/archive-v0/PihKeqevoTnFDUja/bin/python
FIXTURE_DIR := crates/core/fixtures

build:
	@cargo build --workspace --all-targets

install: install-skill
	@cargo install --path apps/cli --locked --force

install-skill:
	@mkdir -p ~/.agents/skills ~/.claude/skills ~/.gemini/config/skills
	@rm -rf ~/.agents/skills/itts ~/.claude/skills/itts ~/.gemini/config/skills/itts
	@cp -R .agents/skills/itts ~/.agents/skills/itts
	@cp -R .claude/skills/itts ~/.claude/skills/itts
	@cp -R .agents/skills/itts ~/.gemini/config/skills/itts

serve:
	@cargo run --release -p index-tts-rs-server -- $(SERVER_ARGS)

test:
	@cargo nextest run --all-features

lint:
	@cargo +nightly fmt --all -- --check
	@cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic \
		-W clippy::unwrap_used -W clippy::expect_used -W clippy::indexing_slicing -W clippy::panic

fmt:
	@cargo +nightly fmt --all

audit:
	@cargo audit
	@cargo deny check

bench:
	@cargo bench -p index-tts-rs-core --bench core_benchmarks

generate-fixtures:
	@mkdir -p $(FIXTURE_DIR)
	@$(PYTHON_BIN) -c '\
import hashlib, json, os, numpy as np; \
import mlx.core as mx; \
from index_tts_2_5_mlx.download import ensure_model; \
from index_tts_2_5_mlx.engine import load_engines; \
from index_tts_2_5_mlx.core import cfm_solver, dsp, gpt_sampler; \
from index_tts_2_5_mlx.core.features import W2VFeatures; \
from index_tts_2_5_mlx.core.frontend import Frontend, lang_to_token; \
model_dir = ensure_model(); \
eng = load_engines(model_dir); \
fe = Frontend(model_dir, use_normalization=True); \
w2v = W2VFeatures(model_dir); \
proj = np.load(os.path.join(model_dir, "spk_proj.npz")); \
spk_proj_w = proj["weight"].astype(np.float32); \
spk_proj_b = proj["bias"].astype(np.float32); \
rng = np.random.default_rng(42); \
t = np.linspace(0.0, 2.0, 48000, endpoint=False, dtype=np.float32); \
raw_audio_24k = np.zeros_like(t); \
[raw_audio_24k.__iadd__((0.15 / k) * np.sin(2.0 * np.pi * 180.0 * k * t + 0.3 * k)) for k in range(1, 45)]; \
raw_audio_24k += 0.01 * rng.standard_normal(t.shape, dtype=np.float32); \
env = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(48000, dtype=np.float32) / 48000.0)).astype(np.float32); \
raw_audio_24k = (raw_audio_24k * env).astype(np.float32); \
audio_22k = dsp.resample_sinc_hann(raw_audio_24k, 24000, 22050); \
audio_16k = dsp.resample_sinc_hann(audio_22k, 22050, 16000); \
ref_mel = dsp.mel_spectrogram(audio_22k)[None].astype(np.float32); \
camp_fbank = dsp.campplus_fbank(audio_16k).astype(np.float32); \
w2v_feat, w2v_mask = w2v(audio_16k); \
spk_cond_emb = eng.semantic_model.run(w2v_feat, w2v_mask).astype(np.float32); \
ilens = np.array([spk_cond_emb.shape[1]], dtype=np.int64); \
emovec = eng.emo_vec.run(spk_cond_emb, ilens).astype(np.float32); \
style = eng.campplus.run(camp_fbank).astype(np.float32); \
spk_cond = (style @ spk_proj_w.T + spk_proj_b).astype(np.float32); \
conds = np.concatenate([(spk_cond + emovec)[:, None, :], np.zeros((1, 2, 1280), dtype=np.float32)], axis=1); \
prompt_condition = eng.length_regulator.run(spk_cond_emb, ref_mel.shape[-1]).astype(np.float32); \
samples = [ \
    ("zh", "大家好，欢迎使用 IndexTTS-2.5 语音合成引擎！"), \
    ("zh", "请把<行|hang2>业报告发给<行|xing2>长，并检查<居|ju1>民区的<曲|qu3>线。"), \
    ("en", "Hello world! We have 3 apples and 25% improvement, surprise!"), \
    ("ja", "<明日|あした>は<東京|トウキョウ>へ行きます。"), \
    ("yue", "今日天气真系好正，我哋一齐去饮茶啦！"), \
    ("zh/en", "我们在 Apple Silicon 上阅读 <read|R EH1 D> 文档并实现了高性能推理。"), \
]; \
tok_ids = [fe.prepare(txt, lang)[0] for lang, txt in samples]; \
tok_tensors = {f"token_ids_{idx}": mx.array(np.asarray(ids, dtype=np.int32)) for idx, ids in enumerate(tok_ids)}; \
tok_meta = {k: v for idx, (lang, txt) in enumerate(samples) for k, v in ((f"sample_{idx}_lang", lang), (f"sample_{idx}_text", txt))}; \
text_ids = np.asarray(fe.prepare(samples[0][1], samples[0][0])[0], dtype=np.int64)[None]; \
lang_id = lang_to_token(samples[0][0]); \
prefill_logits, _ = eng.gpt.prefill(text_ids, conds, lang_id); \
greedy_codes = gpt_sampler.generate_codes(eng.gpt, text_ids, conds, lang_id, greedy=True, max_mel_tokens=32); \
s_infer = eng.codec_decode.run(greedy_codes[None]).astype(np.float32); \
target_len = int(s_infer.shape[1] * 1.72 * 1.0); \
cond = eng.length_regulator.run(s_infer, target_len).astype(np.float32); \
cat_condition = np.concatenate([prompt_condition, cond], axis=1).astype(np.float32); \
cfm_rng = np.random.default_rng(12345); \
cfm_noise_z = cfm_rng.standard_normal((1, 80, cat_condition.shape[1]), dtype=np.float32); \
generated_mel = cfm_solver.solve(eng.cfm, cat_condition, ref_mel, style, n_timesteps=25, cfg_rate=0.7, z=cfm_noise_z).astype(np.float32); \
wav_f32 = eng.bigvgan.run(generated_mel).astype(np.float32); \
ref_tensors = { \
    "raw_audio_24k": mx.array(raw_audio_24k), \
    "audio_22k": mx.array(audio_22k), \
    "audio_16k": mx.array(audio_16k), \
    "ref_mel": mx.array(ref_mel), \
    "campplus_fbank": mx.array(camp_fbank), \
    "w2v_input_features": mx.array(w2v_feat.astype(np.float32)), \
    "w2v_attention_mask": mx.array(w2v_mask.astype(np.int32)), \
    "spk_cond_emb": mx.array(spk_cond_emb), \
    "emovec": mx.array(emovec), \
    "style": mx.array(style), \
    "conds": mx.array(conds), \
    "prompt_condition": mx.array(prompt_condition), \
    "prefill_logits": mx.array(prefill_logits.astype(np.float32)), \
    "greedy_codes": mx.array(greedy_codes.astype(np.int32)), \
    "s_infer": mx.array(s_infer), \
    "cat_condition": mx.array(cat_condition), \
    "cfm_noise_z": mx.array(cfm_noise_z), \
    "generated_mel": mx.array(generated_mel), \
    "wav_f32": mx.array(wav_f32), \
    **tok_tensors, \
}; \
mx.save_safetensors("$(FIXTURE_DIR)/golden_ref.safetensors", ref_tensors, metadata={"format_version": "1", **tok_meta}); \
voice_tensors = { \
    "conds": mx.array(conds), \
    "prompt_condition": mx.array(prompt_condition), \
    "ref_mel": mx.array(ref_mel), \
    "style": mx.array(style), \
}; \
voice_meta = { \
    "format_version": "1", \
    "model_id": "yunfengwang/IndexTTS-2.5-mlx", \
    "source_audio_blake3": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", \
    "emo_alpha": "1.0", \
}; \
mx.save_safetensors("$(FIXTURE_DIR)/golden_speaker.itts_voice", voice_tensors, metadata=voice_meta); \
os.path.exists("$(FIXTURE_DIR)/golden_speaker.itts_voice.safetensors") and os.replace("$(FIXTURE_DIR)/golden_speaker.itts_voice.safetensors", "$(FIXTURE_DIR)/golden_speaker.itts_voice"); \
print("Generated golden fixtures successfully in $(FIXTURE_DIR)"); \
'

check-agent-sync:
	@cmp -s CLAUDE.md AGENTS.md || { \
		echo "AGENTS.md must stay in sync with CLAUDE.md"; \
		echo "Update both files with the same shared project instructions."; \
		exit 1; \
	}
	@tmp_dir=$$(mktemp -d); \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	cp -R .claude/skills "$$tmp_dir/expected-skills"; \
	find "$$tmp_dir/expected-skills" -name SKILL.md -exec perl -0pi -e 's/CLAUDE\.md/AGENTS.md/g; s/Claude/Codex/g; s/claude/codex/g' {} +; \
	diff -ru --exclude agents "$$tmp_dir/expected-skills" .agents/skills || { \
		echo "Codex skills must stay in sync with Claude skills after Claude-to-Codex renaming."; \
		echo "Update .claude/skills first, then mirror the shared content into .agents/skills."; \
		exit 1; \
	}

release:
	@cargo release tag --execute
	@git cliff -o CHANGELOG.md
	@git commit -a -n -m "Update CHANGELOG.md" || true
	@git push origin master
	@cargo release push --execute

update-submodule:
	@git submodule update --init --recursive --remote

.PHONY: build install install-skill serve test lint fmt audit bench generate-fixtures check-agent-sync release update-submodule
