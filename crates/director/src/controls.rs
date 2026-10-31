//! Bounded inline controls compiled into executable synthesis requests.

use std::{iter::repeat_n, str::FromStr};

use index_tts_rs_core::config::{
    DurationFactor, EmotionVector, Language, SpeakerSource, SynthesisRequest, SynthesisText,
    Temperature, VoiceId,
};
use winnow::{Parser, error::ContextError, token::take_until};

use crate::{DirectorError, Result, reading::reading_units};

const MAX_UTTERANCES: usize = 128;

/// One speech request with an exact amount of silence preceding its first audio chunk.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PlannedUtterance {
    /// The text, voice, language, and sampling parameters used for this utterance.
    pub request: SynthesisRequest,
    /// Silence to insert before this utterance, in milliseconds (`0..=5000`).
    pub pause_before_ms: u32,
}

/// Compile `[[pause=600]]`, `[[speed=0.9]]`, `[[emo=melancholic:0.45,calm:0.25]]`,
/// `[[temp=0.75]]`, `[[voice=tyr]]`, `[[lang=zh]]`, and `[[reset]]`.
///
/// Controls apply until changed or reset. Existing `<行|hang2>` pronunciation annotations are
/// preserved. Blank lines create separate utterances; line-ending semicolons retain a strong
/// punctuation cue. Unknown controls, empty scripts, and more than 128 utterances are rejected.
///
/// # Errors
/// Returns [`DirectorError`] for malformed, out-of-range, or unsupported controls.
///
/// # Examples
/// ```
/// use index_tts_rs_core::config::{SpeakerSource, SynthesisRequest, SynthesisText, VoiceId};
/// use index_tts_rs_director::plan_utterances;
/// let request = SynthesisRequest::builder()
///     .text(SynthesisText::new(
///         "你好。[[pause=600]][[emo=melancholic:0.4,calm:0.3]][[temp=0.75]][[speed=0.9]]欢迎。",
///     )?)
///     .speaker(SpeakerSource::VoiceId(VoiceId::new("tyr")?))
///     .build();
/// let plan = plan_utterances(&request, 600)?;
/// assert_eq!(plan.len(), 2);
/// assert_eq!(plan.get(1).map(|part| part.pause_before_ms), Some(600));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn plan_utterances(
    base: &SynthesisRequest,
    paragraph_pause_ms: u32,
) -> Result<Vec<PlannedUtterance>> {
    if paragraph_pause_ms > 5000 {
        return Err(invalid("paragraph pause must be between 0 and 5000 ms"));
    }
    let mut plan = Vec::new();
    let mut state = base.clone();
    let mut pause = 0;
    let mut explicit_pause = false;
    let mut input = base.text.as_str();
    while let Some(position) = input.find("[[") {
        let before = input
            .get(..position)
            .ok_or_else(|| invalid("invalid control boundary"))?;
        let previous_len = plan.len();
        push_text(before, &state, paragraph_pause_ms, &mut pause, &mut plan)?;
        if plan.len() > previous_len {
            explicit_pause = false;
        }
        input = input
            .get(position.saturating_add(2)..)
            .ok_or_else(|| invalid("invalid control boundary"))?;
        let command: &str = take_until::<_, _, ContextError>(1..=256, "]]")
            .parse_next(&mut input)
            .map_err(|_| invalid("control must end with ]] and fit within 256 bytes"))?;
        input = input
            .strip_prefix("]]")
            .ok_or_else(|| invalid("unterminated control"))?;
        apply_control(command, base, &mut state, &mut pause)?;
        if command.starts_with("pause=") {
            explicit_pause = true;
        }
    }
    let previous_len = plan.len();
    push_text(input, &state, paragraph_pause_ms, &mut pause, &mut plan)?;
    if plan.len() > previous_len {
        explicit_pause = false;
    }
    if plan.is_empty() {
        return Err(invalid("script must contain spoken text"));
    }
    if explicit_pause && pause != 0 {
        return Err(invalid("a pause must be followed by spoken text"));
    }
    Ok(plan)
}

fn push_text(
    text: &str,
    state: &SynthesisRequest,
    paragraph_pause: u32,
    pause: &mut u32,
    plan: &mut Vec<PlannedUtterance>,
) -> Result<()> {
    for (index, paragraph) in reading_units(text)?.into_iter().enumerate() {
        if index > 0 {
            *pause = paragraph_pause;
        }
        if plan.len() >= MAX_UTTERANCES {
            return Err(invalid("at most 128 utterances are supported"));
        }
        let mut request = state.clone();
        request.text = SynthesisText::new(paragraph)?;
        plan.push(PlannedUtterance {
            request,
            pause_before_ms: *pause,
        });
    }
    if !text.trim().is_empty() {
        let trailing_newlines = text
            .chars()
            .rev()
            .take_while(|ch| ch.is_whitespace())
            .filter(|ch| *ch == '\n')
            .count();
        *pause = if trailing_newlines >= 2 {
            paragraph_pause
        } else {
            0
        };
    }
    Ok(())
}

fn apply_control(
    command: &str,
    base: &SynthesisRequest,
    state: &mut SynthesisRequest,
    pause: &mut u32,
) -> Result<()> {
    if command == "reset" {
        *state = base.clone();
        return Ok(());
    }
    let (name, value) = command
        .split_once('=')
        .ok_or_else(|| invalid("expected [[name=value]] or [[reset]]"))?;
    match name {
        "pause" => {
            let value = value.strip_suffix("ms").unwrap_or(value);
            let millis = value
                .parse::<u32>()
                .map_err(|_| invalid("pause must be an integer in milliseconds"))?;
            if millis > 5000 {
                return Err(invalid("pause must be between 0 and 5000 ms"));
            }
            *pause = millis;
        }
        "speed" => {
            let speed = value
                .parse::<f32>()
                .map_err(|_| invalid("speed must be a number"))?;
            if !speed.is_finite() || !(0.25..=4.0).contains(&speed) {
                return Err(invalid("speed must be between 0.25 and 4.0"));
            }
            state.cfm.duration_factor = DurationFactor::new(speed.recip())?;
        }
        "voice" => state.speaker = SpeakerSource::VoiceId(VoiceId::new(value)?),
        "lang" => state.language = Language::from_str(value)?,
        "emo" => state.emo_vector = EmotionVector::parse_spec(value)?,
        "temp" | "temperature" => {
            let temp = value
                .parse::<f32>()
                .map_err(|_| invalid("temperature must be a number"))?;
            state.sampling.temperature = Temperature::new(temp)?;
        }
        _ => {
            return Err(invalid(
                "unsupported control; supported controls: pause, speed, emo, temp, voice, lang, \
                 reset",
            ));
        }
    }
    Ok(())
}

/// Insert exact digital silence before PCM audio, shared by playback, streams, and WAV output.
///
/// # Errors
/// Returns [`DirectorError`] for invalid duration, sample rate, or length overflow.
///
/// # Examples
/// ```
/// let mut pcm = vec![123_i16; 10];
/// index_tts_rs_director::prepend_silence(&mut pcm, 22050, 200)?;
/// assert_eq!(pcm.len(), 4420);
/// # Ok::<(), index_tts_rs_director::DirectorError>(())
/// ```
pub fn prepend_silence(pcm: &mut Vec<i16>, sample_rate: u32, duration_ms: u32) -> Result<()> {
    if duration_ms > 5000 || !(8000..=48000).contains(&sample_rate) {
        return Err(invalid("invalid silence duration or sample rate"));
    }
    let samples =
        usize::try_from(u64::from(sample_rate).saturating_mul(u64::from(duration_ms)) / 1000)
            .map_err(|_| invalid("silence length overflow"))?;
    pcm.len()
        .checked_add(samples)
        .ok_or_else(|| invalid("audio length overflow"))?;
    pcm.splice(..0, repeat_n(0, samples));
    Ok(())
}

fn invalid(message: &str) -> DirectorError {
    DirectorError::Invalid(message.to_owned())
}

#[cfg(test)]
mod tests {
    use index_tts_rs_core::config::{
        Language, SpeakerSource, SynthesisRequest, SynthesisText, VoiceId,
    };
    use rstest::rstest;

    use super::plan_utterances;
    use crate::Result;

    fn base(text: &str) -> Result<SynthesisRequest> {
        Ok(SynthesisRequest::builder()
            .text(SynthesisText::new(text)?)
            .speaker(SpeakerSource::VoiceId(VoiceId::new("tyr")?))
            .build())
    }

    #[test]
    fn test_should_execute_controls_without_speaking_them() -> Result<()> {
        let plan = plan_utterances(
            &base(
                "一。[[pause=700ms]][[emo=低落:0.45,平静:0.25]][[temp=0.72]][[speed=0.\
                 9]][[lang=en]][[voice=other]]Two.[[reset]]三。",
            )?,
            600,
        )?;
        assert_eq!(plan.len(), 3);
        let middle = plan
            .get(1)
            .ok_or_else(|| super::invalid("missing test utterance"))?;
        assert_eq!(middle.pause_before_ms, 700);
        assert_eq!(middle.request.text.as_str(), "Two.");
        assert_eq!(middle.request.language, Language::En);
        assert!((middle.request.cfm.duration_factor.get() - 1.0 / 0.9).abs() < f32::EPSILON);
        assert!((middle.request.sampling.temperature.get() - 0.72).abs() < 1e-6);
        let emo = middle
            .request
            .emo_vector
            .ok_or_else(|| super::invalid("missing emotion vector"))?;
        assert!((emo.melancholic - 0.45).abs() < 1e-6);
        assert!((emo.calm - 0.25).abs() < 1e-6);
        let last = plan
            .get(2)
            .ok_or_else(|| super::invalid("missing third test utterance"))?;
        assert_eq!(last.request.language, Language::Zh);
        assert!(last.request.emo_vector.is_none());
        assert!((last.request.sampling.temperature.get() - 0.8).abs() < 1e-6);
        Ok(())
    }

    #[rstest]
    #[case("你好[[emotion=sad]]世界")]
    #[case("你好[[emo=unknown:0.5]]世界")]
    #[case("你好[[emo=sad:1.5]]世界")]
    #[case("你好[[temp=0.01]]世界")]
    #[case("你好[[temp=2.5]]世界")]
    #[case("你好[[pause=5001]]世界")]
    #[case("你好[[speed=NaN]]世界")]
    #[case("你好[[voice=../x]]世界")]
    #[case("你好[[pause=10")]
    #[case("[[pause=100]]")]
    fn test_should_reject_unsupported_or_invalid_controls(#[case] script: &str) -> Result<()> {
        assert!(plan_utterances(&base(script)?, 600).is_err());
        Ok(())
    }

    #[test]
    fn test_should_preserve_pronunciation_and_paragraph_boundaries() -> Result<()> {
        let plan = plan_utterances(&base("<行|hang2>业。\n\n第二段。")?, 800)?;
        assert_eq!(
            plan.first().map(|part| part.request.text.as_str()),
            Some("<行|hang2>业。")
        );
        assert_eq!(plan.get(1).map(|part| part.pause_before_ms), Some(800));
        Ok(())
    }
}
