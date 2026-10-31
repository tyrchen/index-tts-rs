//! Length-bounded JSON Lines framing for the daemon socket.

use std::io::{Error as IoError, ErrorKind};

use serde::{Serialize, de::DeserializeOwned};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt};

use crate::daemon::protocol::MAX_LINE_BYTES;

/// Reads one `\n`-terminated line into `buf`, rejecting lines over [`MAX_LINE_BYTES`] without
/// buffering them. Returns `Ok(false)` at end of stream.
pub async fn read_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    buf: &mut Vec<u8>,
) -> Result<bool, IoError> {
    buf.clear();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(!buf.is_empty());
        }
        let (consumed, done) = match available.iter().position(|&b| b == b'\n') {
            Some(pos) => (pos + 1, true),
            None => (available.len(), false),
        };
        if buf.len().saturating_add(consumed) > MAX_LINE_BYTES {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                format!("line exceeds {MAX_LINE_BYTES} bytes"),
            ));
        }
        let take = if done { consumed - 1 } else { consumed };
        buf.extend_from_slice(available.get(..take).unwrap_or_default());
        reader.consume(consumed);
        if done {
            return Ok(true);
        }
    }
}

/// Reads and decodes the next non-empty JSON line. Returns `Ok(None)` at end of stream.
pub async fn read_json<R, T>(reader: &mut R, buf: &mut Vec<u8>) -> Result<Option<T>, IoError>
where
    R: AsyncBufRead + Unpin,
    T: DeserializeOwned,
{
    loop {
        if !read_line(reader, buf).await? {
            return Ok(None);
        }
        if buf.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        return serde_json::from_slice(buf)
            .map(Some)
            .map_err(|err| IoError::new(ErrorKind::InvalidData, err));
    }
}

/// Encodes `value` as one JSON line.
pub fn encode_line<T: Serialize>(value: &T) -> Result<Vec<u8>, IoError> {
    let mut line = serde_json::to_vec(value).map_err(IoError::other)?;
    line.push(b'\n');
    Ok(line)
}

/// Writes `value` as one JSON line and flushes.
pub async fn write_json<W, T>(writer: &mut W, value: &T) -> Result<(), IoError>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    writer.write_all(&encode_line(value)?).await?;
    writer.flush().await
}

#[cfg(test)]
mod tests {
    use std::io::Result as IoResult;

    use serde_json::{Value, json};
    use tokio::io::BufReader;

    use super::{read_json, read_line};
    use crate::daemon::protocol::MAX_LINE_BYTES;

    #[tokio::test]
    async fn test_should_split_lines_and_skip_blank_ones() -> IoResult<()> {
        let mut reader = BufReader::new(&b"\n{\"a\":1}\n{\"a\":2}"[..]);
        let mut buf = Vec::new();
        let first: Option<Value> = read_json(&mut reader, &mut buf).await?;
        let second: Option<Value> = read_json(&mut reader, &mut buf).await?;
        let third: Option<Value> = read_json(&mut reader, &mut buf).await?;
        assert_eq!(first, Some(json!({"a":1})));
        assert_eq!(second, Some(json!({"a":2})));
        assert_eq!(third, None);
        Ok(())
    }

    #[tokio::test]
    async fn test_should_reject_oversized_lines() {
        let data = vec![b'x'; MAX_LINE_BYTES + 1];
        let mut reader = BufReader::new(data.as_slice());
        let mut buf = Vec::new();
        assert!(read_line(&mut reader, &mut buf).await.is_err());
    }
}
