//! Preserve paragraph boundaries and punctuation cues before model text normalization.

use std::mem::take;

use crate::{DirectorError, Result};

const MAX_READING_PARAGRAPHS: usize = 128;

/// Keep each explicit paragraph together so token packing cannot merge separate stanzas.
pub(crate) fn reading_units(text: &str) -> Result<Vec<String>> {
    let mut units = Vec::new();
    let mut paragraph = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            push_paragraph(&mut units, &mut paragraph)?;
            continue;
        }
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
    push_paragraph(&mut units, &mut paragraph)?;
    Ok(units)
}

fn push_paragraph(units: &mut Vec<String>, paragraph: &mut String) -> Result<()> {
    if !paragraph.is_empty() {
        if units.len() >= MAX_READING_PARAGRAPHS {
            return Err(DirectorError::Invalid(
                "at most 128 paragraphs are supported".to_owned(),
            ));
        }
        units.push(take(paragraph));
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
            units.first().map(String::as_str),
            Some("轻轻的我走了， 正如我轻轻的来。 我轻轻的招手， 作别西天的云彩。")
        );
        Ok(())
    }

    #[test]
    fn test_should_handle_crlf_and_unpunctuated_lines() -> Result<()> {
        let units = reading_units("第一行\r\n第二行\r\n  \r\n第三行\r\n")?;
        assert_eq!(units, ["第一行， 第二行", "第三行"]);
        Ok(())
    }

    #[test]
    fn test_should_keep_single_line_prose_unchanged() -> Result<()> {
        assert_eq!(
            reading_units("Hello; world. 你好，世界！")?,
            ["Hello; world. 你好，世界！"]
        );
        Ok(())
    }

    #[test]
    fn test_should_reject_excessive_paragraphs() {
        assert!(reading_units(&"x\n\n".repeat(129)).is_err());
    }
}
