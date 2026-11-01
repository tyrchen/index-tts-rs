//! Preserve paragraph boundaries and punctuation cues before model text normalization.

use std::{mem::take, ops::Range};

use crate::{DirectorError, Result};

const MAX_READING_PARAGRAPHS: usize = 128;

/// One paragraph to synthesize and the byte range of the input it was read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReadingUnit {
    pub(crate) text: String,
    pub(crate) source: Range<usize>,
}

/// Keep each explicit paragraph together so token packing cannot merge separate stanzas.
pub(crate) fn reading_units(text: &str) -> Result<Vec<ReadingUnit>> {
    let mut units = Vec::new();
    let mut paragraph = String::new();
    let mut source: Option<Range<usize>> = None;
    let mut offset = 0_usize;
    for raw in text.split_inclusive('\n') {
        let line_start = offset;
        offset += raw.len();
        let line = raw.trim();
        if line.is_empty() {
            push_paragraph(&mut units, &mut paragraph, &mut source)?;
            continue;
        }
        let start = line_start + (raw.len() - raw.trim_start().len());
        let range = start..start + line.len();
        source = Some(source.map_or(range.clone(), |open| open.start..range.end));
        if !paragraph.is_empty() {
            // The model normalizer maps semicolons to commas. At an explicit line ending,
            // retain the stronger sentence cue instead of silently losing it.
            if paragraph.ends_with([';', '；']) {
                paragraph.pop();
                paragraph.push('。');
            } else if !paragraph
                .ends_with([',', '，', '.', '。', '!', '！', '?', '？', '…', ':', '：'])
            {
                paragraph.push('，');
            }
            paragraph.push(' ');
        }
        paragraph.push_str(line);
    }
    push_paragraph(&mut units, &mut paragraph, &mut source)?;
    Ok(units)
}

fn push_paragraph(
    units: &mut Vec<ReadingUnit>,
    paragraph: &mut String,
    source: &mut Option<Range<usize>>,
) -> Result<()> {
    if let Some(range) = source.take()
        && !paragraph.is_empty()
    {
        if units.len() >= MAX_READING_PARAGRAPHS {
            return Err(DirectorError::Invalid(
                "at most 128 paragraphs are supported".to_owned(),
            ));
        }
        units.push(ReadingUnit {
            text: take(paragraph),
            source: range,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::reading_units;
    use crate::Result;

    #[test]
    fn test_should_preserve_stanzas_and_semicolon_line_endings() -> Result<()> {
        let units = reading_units(
            "轻轻的我走了，\n正如我轻轻的来；\n我轻轻的招手，\n作别西天的云彩。\n\n那河畔的金柳，\\
             \
             n是夕阳中的新娘；",
        )?;
        assert_eq!(units.len(), 2);
        assert_eq!(
            units.first().map(|unit| unit.text.as_str()),
            Some("轻轻的我走了， 正如我轻轻的来。 我轻轻的招手， 作别西天的云彩。")
        );
        Ok(())
    }

    #[test]
    fn test_should_handle_crlf_and_unpunctuated_lines() -> Result<()> {
        let script = "第一行\r\n第二行\r\n  \r\n第三行\r\n";
        let units = reading_units(script)?;
        let texts: Vec<&str> = units.iter().map(|unit| unit.text.as_str()).collect();
        assert_eq!(texts, ["第一行， 第二行", "第三行"]);
        let sources: Vec<&str> = units
            .iter()
            .filter_map(|unit| script.get(unit.source.clone()))
            .collect();
        assert_eq!(sources, ["第一行\r\n第二行", "第三行"]);
        Ok(())
    }

    #[test]
    fn test_should_keep_single_line_prose_unchanged() -> Result<()> {
        let units = reading_units("  Hello; world. 你好，世界！")?;
        let texts: Vec<&str> = units.iter().map(|unit| unit.text.as_str()).collect();
        assert_eq!(texts, ["Hello; world. 你好，世界！"]);
        assert_eq!(units.first().map(|unit| unit.source.clone()), Some(2..34));
        Ok(())
    }

    #[test]
    fn test_should_reject_excessive_paragraphs() {
        assert!(reading_units(&"x\n\n".repeat(129)).is_err());
    }
}
