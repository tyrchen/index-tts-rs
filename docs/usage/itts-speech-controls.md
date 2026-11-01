# itts Speech Controls and AI Pacing

`itts` (CLI, daemon, and MCP server) and the browser studio compile speech controls with the
same request model before calling IndexTTS. Plain text
continues to work. No language-model request is made by ordinary synthesis.

## Manual controls

```bash
itts synth --voice tyr \
  --text '[[emo=低落:0.35,平静:0.35]][[temp=0.72]][[speed=0.88]]轻轻的我走了，[[pause=600ms]]正如我轻轻的来。' \
  --out poem.wav --play

# Or apply a default 8-class emotion mix across the entire synthesis request:
itts synth --voice tyr --emo "melancholic:0.45,calm:0.25" --text "沉默是今晚的康桥！" --play
```

| Control | Behavior |
| --- | --- |
| `[[emo=melancholic:0.45,calm:0.25]]` or `[[emo=低落:0.45,平静:0.25]]` | Blend up to 8 IndexTTS-2.5 emotion prototypes (`happy`/`高兴`, `angry`/`愤怒`, `sad`/`悲伤`, `afraid`/`恐惧`, `disgusted`/`反感`, `melancholic`/`低落`, `surprised`/`惊讶`, `calm`/`平静`); `[[emo=none]]` restores the speaker's base emotion |
| `[[temp=0.75]]` or `[[temperature=0.75]]` | Set GPT sampling temperature (`0.05..=2.0`, default `0.8`); lower (`0.68..0.76`) yields restrained/gentle inflection, higher (`0.84..0.95`) yields stronger prosodic expressiveness |
| `[[pause=600]]` or `[[pause=600ms]]` | Insert 600 ms of silence before the following speech; range 0–5000 ms |
| `[[speed=0.9]]` | Change speaking speed until the next speed control or reset, relative to the job speed (`--speed 1.1` with `[[speed=0.9]]` speaks at 0.99); range 0.25–4.0, larger is faster |
| `[[voice=tyr]]` | Switch to a registered voice ID |
| `[[lang=zh]]` | Switch synthesis language |
| `[[reset]]` | Restore the voice, language, emotion, temperature, and speed supplied to the synthesis command |
| `<荇|xing4>` | Existing model pronunciation override, passed through unchanged |

Controls apply to subsequent speech. A nonzero pause must be followed by text. Unknown controls
are rejected. Emotion vectors are matched against the speaker's `CAMPPlus` style embedding via
cosine similarity across the 73 upstream `IndexTTS-2.5` emotion prototypes (`emo_matrix.safetensors`),
normalized with upstream bias weights, and capped at a total weight sum of `0.8` so the speaker's
timbre is preserved. Rate, emotion, temperature, and voice changes start new inference units, so
prefer complete phrases or couplets over individual words.
All audible pauses, including playback and raw PCM output, are also present in the saved WAV.
CLI JSON Lines identify control-plan units with `utteranceIndex` / `totalUtterances`;
`segmentIndex` / `totalSegments` identify token-budget segments within that unit.

Blank lines separate paragraphs or poem stanzas. Lines within each paragraph retain their
punctuation; a line-ending semicolon becomes a full stop before model normalization, and an
unpunctuated line gets a comma before the next line. The model still determines phrasing within
an utterance. Use an explicit pause tag when an exact additional gap is needed. This does not
provide word-level duration, pitch contours, or text-driven emotional acting.

`--paragraph-pause 600` controls additional silence between explicit paragraphs in the CLI.
The browser exposes the same setting. Existing generated silence at the edges of speech is
preserved, so the total acoustic gap can exceed the additional silence value.

## Optional DeepSeek annotation

Set `DEEPSEEK_API_KEY` in the terminal that starts `itts` or the server, or send a key with
each request (HTTP header `X-DeepSeek-Api-Key`, daemon `apiKey` field). Keys are redacted in
Debug output, never logged, never sent to the browser, and never stored in YAML.

```bash
# Send the source text to DeepSeek once and save an editable control script.
itts annotate --stdin --poetry \
  --director-config config/director.yaml < poem.txt > poem.itts.txt

# Synthesis is local and can be repeated without another model request.
itts synth --voice tyr --stdin --out poem.wav --play < poem.itts.txt
```

The YAML settings in [director.yaml](../../config/director.yaml) select the model, request
timeout, and output-token budget. Without a file, the same defaults apply: `deepseek-flash`,
30 seconds, and 4096 output tokens. Thinking is disabled and JSON output is enabled.
`deepseek-flash` currently serves DeepSeek V4.1 Flash; see the
[official model documentation](https://api-docs.deepseek.com/quick_start/pricing/).

The model receives indexed source units (paragraphs in prose mode, or fine-grained couplets split
at `;` / `；` / `。` / `！` / `？` line endings in `--poetry` mode) and returns only unit indices,
8-class `emotions` weights (`0.0..=1.0`), `temperature` (`0.65..=0.95`), `speed` (`0.8..=1.2`),
and `pauseAfterMs` (`0..=2000` ms). Each unit must appear exactly once, in order. The local
renderer supplies all spoken text and validates the entire result before using it. No model
output is permitted to add, remove, or rewrite words. Empty, incomplete, excessive, or invalid
responses fail explicitly and leave the source intact. Requests use HTTPS with the Rustls
AWS-LC backend, disable redirects, and cap response bodies at 64 KiB.

In the browser, click **AI 朗读标注 · DeepSeek** to prepare a script, review or edit the controls,
and generate audio. **恢复原文** restores the untagged source. Annotation is optional and adds a
provider round trip; it is deliberately separate from speech generation so retries can reuse
an existing script. Local synthesis works without an API key.

### Per-line direction

For scripts that are synthesized line by line (one request per line), `itts annotate --units`
takes your lines with stable IDs and returns one delivery per line instead of a script:

```bash
cat > lines.json <<'JSON'
{ "style": "natural", "brief": "tech documentary narration, restrained and confident",
  "units": [ { "id": "s1.l1", "text": "三年前，我们的推理服务每天都在超时。" },
             { "id": "s1.l2", "text": "工程师们一次次熬夜排查，却始终找不到根因。" } ] }
JSON
itts annotate --units --stdin --output json < lines.json
```

```json
{ "status": "ok", "deliveries": [
  { "id": "s1.l1", "emo": "melancholic:0.2,calm:0.45", "temperature": 0.76, "speedScale": 0.9,
    "pauseAfterMs": 520, "controls": "[[emo=melancholic:0.2,calm:0.45]][[temp=0.76]][[speed=0.9]]" },
  ... ] }
```

Either prefix the line's text with `controls`, or copy the fields into its `SpeechJob`: `emo`,
`advanced.temperature`, and `speed` = your base speed × `speedScale` (the default base speed is
1.2, so `0.9` becomes `1.08`; `[[speed=0.9]]` applies the same scale). `pauseAfterMs` is a suggestion for
the gap you place after the line. The same request is served by the daemon (`annotate`), MCP
(`annotate_units`), and HTTP (`POST /v1/annotate`); `itts schema annotate-job` and
`itts schema annotate-result` publish the JSON Schemas. The provider sees all lines at once and is
told to change delivery gradually between neighbours unless the text turns, which keeps the
emotional arc coherent across lines.

## Sentence continuity

By default every sentence is generated as a fresh utterance, so each one starts like the
beginning of a new speech. `--continuity` (`"continuity": true`) continues each sentence after
the previous one: the decoder receives the previous sentence's text and speech codes as a prefix
and keeps going, so pace and intonation carry over. It applies between sentences and across
paragraphs spoken by the same voice in the same language; paragraph pauses and `[[pause]]` still
apply, while the fixed inter-sentence silence defaults to `0` because the model speaks its own
pauses.

When lines are synthesized in separate requests, pass the previous line as `--previous-text`
(`"previousText"`): the engine regenerates that line's speech codes with the same seed (about
0.4 s of extra GPT time) and continues after them. The previous text is not spoken, and the
result stays reproducible from `(input, previousText, seed, …)`.

```bash
itts synth --voice narrator --seed 7 --out line2.wav \
  --previous-text "三年前，我们的推理服务每天都在超时。" \
  --text "工程师们一次次熬夜排查，却始终找不到根因。"
```

Continuity is off by default because whether it sounds better is a listening decision; with the
previous sentence excluded from the repetition penalty its error rate is no higher than
independent synthesis. See the
[experiment notes](../research/sentence-continuity-research.md).

## Browser streaming and latency

Run `make serve`, then open <http://127.0.0.1:8080>. This target runs the optimized release
binary and keeps the model loaded across requests. The browser receives PCM chunks as they are
produced, can play them immediately, and assembles one complete WAV for download. The displayed
first-audio time, total generation time, and audio duration are separate measurements. Early
playback reduces waiting without reducing the CFM step count or changing sampling quality.

`POST /api/speech/stream` accepts the same fields as `/v1/audio/speech`, plus optional
`paragraphPauseMs` (default 600, range 0–5000). It returns newline-delimited JSON:

- `chunk`: `pcmBase64` (little-endian signed 16-bit mono PCM, including additional pauses),
  `sampleRate`, `utteranceIndex`, `totalUtterances`, and `elapsedSecs`.
- `done`: `wallSecs`, `durationSecs`, `samples`, `sampleRate`, and `paragraphs`.
- `error`: an explanatory message if generation fails after response headers were sent.

`POST /api/annotate` accepts `{"input":"...","poetry":true}` and returns `{"script":"..."}`.
The server supports `--director-config /path/to/director.yaml` for provider settings.

Generation has a 120-second timeout, a ten-minute audio cap, and at most 128 inference units.
Only one speech synthesis is admitted at a time; competing synthesis requests receive a busy
response instead of joining a long GPU queue. Cancelling the browser request disconnects the
stream; the worker finishes its current segment and skips remaining segments. Audio already
submitted to the GPU is not interrupted mid-segment.

Prosody also depends on the model and the reference recording. Controls provide pacing and
pronunciation, but cannot guarantee a particular poetic interpretation. Moderate rate changes
near 1.0 tend to avoid forcing large duration changes; compare the result by listening.
