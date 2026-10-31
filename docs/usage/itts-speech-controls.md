# itts Speech Controls and AI Pacing

`itts` and the browser studio compile speech controls before calling IndexTTS. Plain text
continues to work. No language-model request is made by ordinary synthesis.

## Manual controls

```bash
itts synth --voice tyr \
  --text '[[speed=0.9]]轻轻的我走了，[[pause=600ms]]正如我轻轻的来。' \
  --out poem.wav --play
```

| Control | Behavior |
| --- | --- |
| `[[pause=600]]` or `[[pause=600ms]]` | Insert 600 ms of silence before the following speech; range 0–5000 ms |
| `[[speed=0.9]]` | Change speaking speed until the next speed control or reset; range 0.25–4.0, larger is faster |
| `[[voice=tyr]]` | Switch to a registered voice ID |
| `[[lang=zh]]` | Switch synthesis language |
| `[[reset]]` | Restore the voice, language, and speed supplied to the synthesis command |
| `<荇|xing4>` | Existing model pronunciation override, passed through unchanged |

Controls apply to subsequent speech. A nonzero pause must be followed by text. Unknown controls
are rejected; `emotion`, `pitch`, and `emphasis` are not implemented acoustic controls. Rate and
voice changes start new inference units, so prefer complete phrases over individual words.
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

Set `DEEPSEEK_API_KEY` in the terminal that starts `itts` or the server. The key is read from
the environment, redacted in Debug output, and never sent to the browser or stored in YAML.

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

The model receives indexed source paragraphs and returns only paragraph indices, speeds
(0.8–1.2), and pauses (0–2000 ms). Each paragraph must appear exactly once, in order. The local
renderer supplies all spoken text and validates the entire result before using it. No model
output is permitted to add, remove, or rewrite words. Empty, incomplete, excessive, or invalid
responses fail explicitly and leave the source intact. Requests use HTTPS with the Rustls
AWS-LC backend, disable redirects, and cap response bodies at 64 KiB.

In the browser, click **AI 朗读标注 · DeepSeek** to prepare a script, review or edit the controls,
and generate audio. **恢复原文** restores the untagged source. Annotation is optional and adds a
provider round trip; it is deliberately separate from speech generation so retries can reuse
an existing script. Local synthesis works without an API key.

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
