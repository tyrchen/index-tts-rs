use rstest::rstest;

use super::{AlignEvidence, Finding, FindingKind, Severity, compare};
use crate::tokens::HeardToken;

/// Heard tokens for `text`: CJK characters one by one, Latin words and numbers whole, 0.2 s
/// apart.
#[allow(clippy::cast_precision_loss)]
fn heard(text: &str) -> Vec<HeardToken> {
    let mut pieces: Vec<String> = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '.' | '%') {
            word.push(c);
            continue;
        }
        if !word.is_empty() {
            pieces.push(std::mem::take(&mut word));
        }
        if !c.is_whitespace() {
            pieces.push(c.to_string());
        }
    }
    if !word.is_empty() {
        pieces.push(word);
    }
    pieces
        .into_iter()
        .enumerate()
        .map(|(i, text)| HeardToken {
            text,
            start: i as f32 * 0.2,
            end: (i + 1) as f32 * 0.2,
        })
        .collect()
}

fn check(script: &str, recognized: &str) -> Vec<Finding> {
    compare(script, &heard(recognized), AlignEvidence::default()).unwrap_or_default()
}

fn summary(findings: &[Finding]) -> Vec<(FindingKind, Severity, String, String)> {
    findings
        .iter()
        .map(|f| (f.kind, f.severity, f.expected.clone(), f.heard.clone()))
        .collect()
}

#[rstest]
#[case("从北边吹来的风越来越冷。", "从北北边吹来的风越来越冷", "北")]
#[case("我们今天下午去公园散步。", "我我们今天下午去公园散步", "我")]
#[case("编译器会检查所有的类型。", "编译编译器会检查所有的类型", "编译")]
#[case(
    "每一次笑，都是一把尺子在量人。",
    "每一次每一次笑都是一把尺子在量人",
    "每一次"
)]
#[case("鲲化而为鸟，其名为鹏。", "昆画画而为鸟其名为鹏", "画")]
#[case("The engine runs hot today.", "The the engine runs hot today.", "the")]
#[case(
    "We need to finish this by Friday.",
    "We need to to finish this by Friday.",
    "to"
)]
fn test_should_report_stutters_as_repeat_errors(
    #[case] script: &str,
    #[case] recognized: &str,
    #[case] extra: &str,
) {
    let found = check(script, recognized);
    let got: Vec<_> = summary(&found)
        .into_iter()
        .map(|(kind, severity, expected, heard)| (kind, severity, expected, heard.to_lowercase()))
        .collect();
    assert_eq!(
        got,
        [(
            FindingKind::Repeat,
            Severity::Error,
            String::new(),
            extra.to_owned()
        )],
        "{found:?}"
    );
    assert!(found.iter().all(|f| f.chars.is_empty()));
}

#[rstest]
#[case("2026年，我们发布了第三个版本。", "两千零二十六年我们发布了第三个版本")]
#[case("2026年，我们发布了第三个版本。", "二零二六年我们发布了第三个版本")]
#[case("11月3日上午开会。", "十一月三号上午开会")]
#[case("速度提升了1.5倍。", "速度提升了一点五倍")]
#[case("鹏往下看，看到的也是一片蓝。", "朋往下看看到的也是一片蓝")]
#[case("现在就用 Homebrew 安装 Axon 吧。", "现在就用 home brew 安装 axon 吧")]
#[case("哪儿都去不了。", "哪都去不了")]
#[case("出门前敲句 <axon|AE1 K S AA0 N> away。", "出门前敲句 axon away")]
#[case(
    "That recalls Heartbleed, in twenty fourteen.",
    "That recalls Heartbleed in 2014."
)]
fn test_should_accept_equivalent_readings(#[case] script: &str, #[case] recognized: &str) {
    let found = check(script, recognized);
    assert!(found.is_empty(), "{found:?}");
}

#[rstest]
#[case("第3章讲的是热机。", "第三十章讲的是热机", "3", "三十")]
#[case("温度降到了零下5度。", "温度降到了零下十五度", "5", "十五")]
#[case(
    "大约有85%的人同意。",
    "大约有百分之五十八的人同意",
    "85%",
    "百分之五十八"
)]
#[case("后天上午十点见。", "后天上午三点见", "十", "三")]
#[case(
    "In 2026, sales grew by 15 percent.",
    "In 2026, sales grew by 50%.",
    "15 percent",
    "50%"
)]
fn test_should_report_other_values_as_number_errors(
    #[case] script: &str,
    #[case] recognized: &str,
    #[case] expected: &str,
    #[case] got: &str,
) {
    let found = check(script, recognized);
    assert_eq!(
        summary(&found),
        [(
            FindingKind::Number,
            Severity::Error,
            expected.to_owned(),
            got.to_owned()
        )]
    );
}

#[test]
fn test_should_grade_omissions_by_size() {
    let found = check("他放下手里的书，抬头看了看窗外。", "他放下书抬头看了看窗外");
    assert_eq!(
        summary(&found),
        [(
            FindingKind::Omission,
            Severity::Warning,
            "手里的".to_owned(),
            String::new()
        )]
    );
    assert_eq!(found.first().map(|f| f.chars.clone()), Some(3..6));
    let found = check("风积得不厚，就托不起大翅膀。", "风积得不厚就托不起翅膀");
    assert_eq!(
        summary(&found),
        [(
            FindingKind::Omission,
            Severity::Info,
            "大".to_owned(),
            String::new()
        )]
    );
}

#[test]
fn test_should_raise_one_unit_differences_the_aligner_also_doubts() {
    let tokens = heard("这本书的作者是孔子");
    let quiet = compare("这本书的作者是庄子。", &tokens, AlignEvidence::default());
    let weak = [7];
    let doubted = compare(
        "这本书的作者是庄子。",
        &tokens,
        AlignEvidence {
            weak: &weak,
            ..AlignEvidence::default()
        },
    );
    let severity = |r: &super::Result<Vec<Finding>>| {
        r.as_ref()
            .ok()
            .and_then(|f| f.first().map(|f| (f.kind, f.severity, f.align_weak)))
    };
    assert_eq!(
        severity(&quiet),
        Some((FindingKind::Substitution, Severity::Info, false))
    );
    assert_eq!(
        severity(&doubted),
        Some((FindingKind::Substitution, Severity::Warning, true))
    );
}

#[test]
fn test_should_report_insertions_that_do_not_repeat() {
    let found = check(
        "这个问题没有标准答案，但值得想一想。",
        "这个问题其实没有标准答案但是值得认真想一想",
    );
    let kinds: Vec<_> = found.iter().map(|f| (f.kind, f.heard.as_str())).collect();
    assert_eq!(
        kinds,
        [
            (FindingKind::Insertion, "其实"),
            (FindingKind::Insertion, "是"),
            (FindingKind::Insertion, "认真")
        ]
    );
}

#[test]
fn test_should_report_skipped_words_as_an_omission() {
    let found = check(
        "Please send me the final quarterly report by tomorrow morning.",
        "Please send me the report by tomorrow morning.",
    );
    assert_eq!(
        summary(&found),
        [(
            FindingKind::Omission,
            Severity::Warning,
            "final quarterly".to_owned(),
            String::new()
        )]
    );
    assert_eq!(found.first().map(|f| f.sounds), Some(2));
}

#[test]
fn test_should_report_a_misread_word_once() {
    let found = check(
        "Carnot asked how much work heat can do.",
        "Carnot asked how much work steam can do.",
    );
    assert_eq!(
        summary(&found),
        [(
            FindingKind::Substitution,
            Severity::Info,
            "heat".to_owned(),
            "steam".to_owned()
        )]
    );
    let found = check("From vLLM to SGLang.", "from v l l m to s g line");
    assert_eq!(
        summary(&found),
        [(
            FindingKind::Substitution,
            Severity::Info,
            "SGLang".to_owned(),
            "s g line".to_owned()
        )]
    );
}

#[test]
fn test_should_time_findings_from_the_heard_tokens_and_the_alignment() {
    let found = check("从北边吹来的风。", "从北北边吹来的风");
    assert_eq!(found.first().map(|f| (f.start, f.end)), Some((0.2, 0.4)));
    let starts = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8];
    let ends = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1];
    let evidence = AlignEvidence {
        weak: &[],
        starts: &starts,
        ends: &ends,
    };
    let found = compare(
        "鲲化而为鸟，其名为鹏。",
        &heard("鲲而为鸟其名为鹏"),
        evidence,
    )
    .unwrap_or_default();
    assert_eq!(found.first().map(|f| (f.start, f.end)), Some((0.3, 0.6)));
    let found =
        compare("鲲化而为鸟。", &heard("鲲而为鸟"), AlignEvidence::default()).unwrap_or_default();
    assert_eq!(found.first().map(|f| (f.start, f.end)), Some((0.2, 0.2)));
}

#[test]
fn test_should_report_everything_heard_for_an_unrelated_recording() {
    let found = check("你好。", "");
    assert_eq!(
        summary(&found),
        [(
            FindingKind::Omission,
            Severity::Warning,
            "你好".to_owned(),
            String::new()
        )]
    );
    let found = check("你好。", "再见");
    assert_eq!(found.len(), 1);
}
