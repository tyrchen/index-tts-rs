'use strict';

const element = (id) => document.getElementById(id);
const form = element('speech-form');
const input = element('input');
const voice = element('voice');
const language = element('lang');
const speed = element('speed');
const pause = element('paragraph-pause');
const generate = element('generate');
const refresh = element('refresh-voices');
const status = element('form-status');
const errorBox = element('error');
const voiceHint = element('voice-hint');
const emptyHint = element('empty-hint');
const counter = element('text-count');
const audio = element('audio');
const download = element('download');
const livePreview = element('live-preview');
const stopButton = element('stop-preview');
const cancelButton = element('cancel-generation');
const annotateButton = element('annotate');
const undoAnnotation = element('undo-annotation');
const encoder = new TextEncoder();
const maxBytes = 16384;
let voices = [];
let busy = false;
let loadingVoices = false;
let audioUrl = null;
let controller = null;
let preview = null;
let nextPlayTime = 0;
let playingSources = 0;
let originalText = null;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

function friendlyError(error) {
  if (error.name === 'TimeoutError') return '请求超时，请缩短文本后重试。';
  if (error instanceof TypeError) return '无法连接语音服务，请确认 server 已启动后重试。';
  return error.message;
}

function updateForm() {
  const bytes = encoder.encode(input.value).length;
  const tooLong = bytes > maxBytes;
  counter.textContent = `${bytes.toLocaleString('en-US')} / 16,384 字节`;
  counter.classList.toggle('over-limit', tooLong);
  input.classList.toggle('invalid', tooLong);
  input.setCustomValidity(tooLong ? '文本不能超过 16,384 字节，请缩短后重试。' : '');
  generate.disabled = busy || loadingVoices || !voice.value || !input.value.trim() || tooLong;
  annotateButton.disabled = busy || !input.value.trim() || tooLong || input.value.includes("[[");
  undoAnnotation.disabled = busy;
}

function updateVoiceHint() {
  const selected = voices.find((item) => item.id === voice.value);
  if (selected) voiceHint.textContent = `参考音频 ${selected.durationSecs.toFixed(1)} 秒 · 已注册音色`;
}

async function errorMessage(response) {
  try {
    const body = await response.json();
    if (typeof body.error === 'string') return body.error;
  } catch { /* Fall back to the HTTP status when a proxy returns a non-JSON error. */ }
  return `请求失败（HTTP ${response.status}），请稍后重试。`;
}

async function loadVoices() {
  const previous = voice.value;
  loadingVoices = true;
  refresh.disabled = true;
  voice.disabled = true;
  status.textContent = '正在读取音色库…';
  showError('');
  updateForm();
  try {
    const response = await fetch('/api/voices', { signal: AbortSignal.timeout(15000) });
    if (!response.ok) throw new Error(await errorMessage(response));
    const data = await response.json();
    if (!Array.isArray(data) || !data.every((item) => typeof item.id === 'string' && Number.isFinite(item.durationSecs))) {
      throw new Error('音色列表格式无效，请刷新页面后重试。');
    }
    voices = data;
    voice.replaceChildren();
    for (const item of voices) voice.add(new Option(item.id, item.id));
    if (voices.some((item) => item.id === previous)) voice.value = previous;
    emptyHint.hidden = voices.length > 0;
    voice.disabled = voices.length === 0;
    if (voices.length === 0) {
      voice.add(new Option('暂无音色', ''));
      voiceHint.textContent = '先注册一个音色，再开始生成。';
      status.textContent = '等待注册音色。';
    } else {
      updateVoiceHint();
      status.textContent = '准备就绪。保留空行可以保留诗节和段落。';
    }
  } catch (error) {
    voices = [];
    voice.replaceChildren(new Option('音色读取失败', ''));
    voiceHint.textContent = '点击「刷新列表」重试。';
    status.textContent = '暂时无法读取音色库。';
    showError(friendlyError(error));
  } finally {
    loadingVoices = false;
    refresh.disabled = false;
    updateForm();
  }
}

function stopPreview() {
  const context = preview;
  preview = null;
  playingSources = 0;
  stopButton.hidden = true;
  if (context && context.state !== 'closed') context.close().catch(() => {});
}

function scheduleAudio(bytes, sampleRate) {
  if (!preview) return;
  const context = preview;
  const pcm = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const buffer = context.createBuffer(1, bytes.byteLength / 2, sampleRate);
  const samples = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i++) samples[i] = pcm.getInt16(i * 2, true) / 32768;
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  playingSources++;
  source.onended = () => {
    if (preview !== context) return;
    playingSources--;
    if (!busy && playingSources === 0) stopPreview();
  };
  nextPlayTime = Math.max(nextPlayTime, context.currentTime + 0.04);
  source.start(nextPlayTime);
  nextPlayTime += buffer.duration;
}

function wavBlob(chunks, byteLength, sampleRate) {
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  const text = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
  text(0, 'RIFF'); view.setUint32(4, 36 + byteLength, true);
  text(8, 'WAVE'); text(12, 'fmt '); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, byteLength, true);
  return new Blob([header, ...chunks], { type: 'audio/wav' });
}

async function readEvents(response, receive) {
  if (!response.body) throw new Error('浏览器不支持流式响应。');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += done ? decoder.decode() : decoder.decode(value, { stream: true });
      if (pending.length > 8_000_000) throw new Error('服务返回的音频片段过大。');
      let boundary;
      while ((boundary = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, boundary);
        pending = pending.slice(boundary + 1);
        if (line.trim()) receive(JSON.parse(line));
      }
      if (done) break;
    }
    if (pending.trim()) receive(JSON.parse(pending));
  } finally {
    reader.releaseLock();
  }
}

function setBusy(value) {
  busy = value;
  form.setAttribute('aria-busy', String(value));
  generate.classList.toggle('busy', value);
  element('button-label').textContent = value ? '正在生成…' : '生成语音';
  cancelButton.hidden = !value;
  for (const control of [input, voice, language, speed, pause, refresh, livePreview]) control.disabled = value;
  updateForm();
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (generate.disabled || !form.reportValidity()) return;
  const payload = { input: input.value, voice: voice.value, lang: language.value, speed: Number(speed.value), paragraphPauseMs: Number(pause.value) };
  const selectedLanguage = language.selectedOptions.item(0)?.textContent ?? payload.lang;
  const chunks = [];
  let byteLength = 0;
  let sampleRate = null;
  let firstAudio = null;
  let summary = null;
  const started = performance.now();
  controller = new AbortController();
  stopPreview();
  audio.pause();
  setBusy(true);
  showError('');
  status.textContent = '正在生成第一段语音…';
  try {
    if (livePreview.checked) {
      preview = new AudioContext();
      await preview.resume();
      nextPlayTime = 0;
      stopButton.hidden = false;
    }
    const response = await fetch('/api/speech/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      signal: AbortSignal.any([controller.signal, AbortSignal.timeout(125000)]),
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    await readEvents(response, (packet) => {
      if (packet.event === 'error') throw new Error(packet.error);
      if (packet.event === 'done') { summary = packet; return; }
      if (packet.event !== 'chunk' || summary) throw new Error('音频流格式无效。');
      if (!Number.isInteger(packet.sampleRate) || packet.sampleRate < 8000 || packet.sampleRate > 48000) throw new Error('音频采样率无效。');
      if (sampleRate !== null && sampleRate !== packet.sampleRate) throw new Error('音频流采样率不一致。');
      sampleRate = packet.sampleRate;
      const binary = atob(packet.pcmBase64);
      const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
      if (!bytes.length || bytes.length % 2) throw new Error('音频数据无效。');
      byteLength += bytes.length;
      if (byteLength > 64_000_000) throw new Error('生成的音频过长，请分段提交。');
      chunks.push(bytes);
      if (firstAudio === null) firstAudio = (performance.now() - started) / 1000;
      scheduleAudio(bytes, sampleRate);
      status.textContent = `正在生成第 ${packet.utteranceIndex + 1} / ${packet.totalUtterances} 段${preview ? '，同步试听中' : ''}…`;
    });
    if (!summary || !byteLength || summary.samples !== byteLength / 2) throw new Error('音频流未完整生成，请重试。');
    const blob = wavBlob(chunks, byteLength, sampleRate);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = URL.createObjectURL(blob);
    audio.src = audioUrl;
    download.href = audioUrl;
    download.download = `${payload.voice}-${Date.now()}.wav`;
    element('result-empty').hidden = true;
    element('result').hidden = false;
    const shownVoice = payload.input.includes('[[voice=') ? '音色按标签' : payload.voice;
    const shownLanguage = payload.input.includes('[[lang=') ? '语言按标签' : selectedLanguage;
    const shownSpeed = payload.input.includes('[[speed=') ? '语速按标签' : `${payload.speed.toFixed(1)}×`;
    const shownEmo = payload.input.includes('[[emo=') ? ' · 情绪按标签' : '';
    element('result-meta').textContent = `${shownVoice} · ${shownLanguage} · ${shownSpeed}${shownEmo} · 首段 ${firstAudio.toFixed(1)} 秒 · 全部生成 ${summary.wallSecs.toFixed(1)} 秒 · 音频 ${summary.durationSecs.toFixed(1)} 秒`;
    status.textContent = '生成完成。下载文件包含完整语音和段落停顿。';
  } catch (error) {
    controller.abort();
    stopPreview();
    status.textContent = error.name === 'AbortError' ? '已停止生成。' : '生成未完成，请重试。';
    if (error.name !== 'AbortError') showError(friendlyError(error));
  } finally {
    controller = null;
    setBusy(false);
    if (preview && playingSources === 0) stopPreview();
  }
});

annotateButton.addEventListener('click', async () => {
  if (annotateButton.disabled) return;
  const source = input.value;
  controller = new AbortController();
  setBusy(true);
  element('button-label').textContent = '正在标注…';
  status.textContent = 'DeepSeek 正在标注朗读节奏，原文不会由模型改写…';
  showError('');
  try {
    const response = await fetch('/api/annotate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: source, poetry: /\n\s*\n/.test(source) }),
      signal: AbortSignal.any([controller.signal, AbortSignal.timeout(125000)]),
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    const result = await response.json();
    if (typeof result.script !== 'string' || encoder.encode(result.script).length > maxBytes) throw new Error('AI 标注结果无效。');
    originalText = source;
    input.value = result.script;
    undoAnnotation.hidden = false;
    status.textContent = '已添加情绪、语调、语速与停顿标签。可以编辑，再生成试听。';
  } catch (error) {
    controller.abort();
    status.textContent = '标注未完成，原文已保留。';
    if (error.name !== 'AbortError') showError(friendlyError(error));
  } finally {
    controller = null;
    setBusy(false);
  }
});
undoAnnotation.addEventListener('click', () => {
  if (originalText !== null) input.value = originalText;
  originalText = null;
  undoAnnotation.hidden = true;
  status.textContent = '已恢复原文。';
  showError('');
  updateForm();
});
input.addEventListener('input', updateForm);
voice.addEventListener('change', updateVoiceHint);
speed.addEventListener('input', () => { element('speed-value').value = `${Number(speed.value).toFixed(1)}×`; });
pause.addEventListener('input', () => { element('pause-value').value = `${(Number(pause.value) / 1000).toFixed(1)} 秒`; });
refresh.addEventListener('click', loadVoices);
stopButton.addEventListener('click', stopPreview);
cancelButton.addEventListener('click', () => { controller?.abort(); stopPreview(); });
audio.addEventListener('play', stopPreview);
window.addEventListener('pagehide', (event) => {
  controller?.abort();
  stopPreview();
  if (!event.persisted && audioUrl) URL.revokeObjectURL(audioUrl);
});
loadVoices();
