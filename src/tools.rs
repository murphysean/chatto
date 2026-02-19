use serde::Deserialize;
use serde_json::Value;
use std::{collections::HashMap, process::Command};
use strum::{Display, EnumString};

#[derive(Debug, Deserialize, Clone)]
pub struct OutputLimit {
    pub max_size: usize,
    pub method: TrimMethod,
}

#[derive(Debug, Deserialize, Clone, Display, EnumString)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum TrimMethod {
    Head,
    Tail,
    Bytes,
}

/// Default impl for Output Limit, 0 means no limit
impl Default for OutputLimit {
    fn default() -> Self {
        Self {
            max_size: 0,
            method: TrimMethod::Head,
        }
    }
}

impl From<OutputLimit> for config::Value {
    fn from(value: OutputLimit) -> Self {
        let mut ret: HashMap<String, String> = HashMap::new();
        ret.insert("max_size".to_string(), value.max_size.to_string());
        ret.insert("method".to_string(), value.method.to_string());
        ret.into()
    }
}

pub fn create_shell_tool() -> Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "execute_shell",
            "description": "Execute a shell command on the system",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "reason": {
                        "type": "string",
                        "description": "The reason the agent needs to use this command"
                    }
                },
                "required": ["command", "reason"]
            }
        }
    })
}

pub fn execute_command(command: &str, output_limit: &OutputLimit) -> String {
    println!("EXECUTING {}", command);
    let output = match Command::new("sh").arg("-c").arg(command).output() {
        Ok(output) => output,
        Err(e) => {
            let error_msg = format!("Failed to execute command: {}", e);
            println!("EXECUTION ERROR: {}", error_msg);
            return error_msg;
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let result = if !output.status.success() {
        let error_msg = format!(
            "Command failed with exit code {}: {}",
            output.status.code().unwrap_or(-1),
            stderr
        );
        println!("COMMAND FAILED\n{}", error_msg);
        error_msg
    } else {
        let result = trim_output(&stdout, output_limit);
        println!("OUTPUT:\n{}", result);
        result
    };

    result
}

fn trim_output(output: &str, limit: &OutputLimit) -> String {
    //Trim is disabled
    if limit.max_size == 0 {
        return output.to_string();
    }
    match limit.method {
        TrimMethod::Head => {
            let lines: Vec<&str> = output.lines().collect();
            let max_lines = limit.max_size / 80; // Rough estimate of chars per line
            if lines.len() > max_lines {
                let mut result = lines[..max_lines].join("\n");
                result.push_str(&format!(
                    "\n... [Output truncated: showing first {} lines of {}]",
                    max_lines,
                    lines.len()
                ));
                result
            } else {
                output.to_string()
            }
        }
        TrimMethod::Tail => {
            let lines: Vec<&str> = output.lines().collect();
            let max_lines = limit.max_size / 80;
            if lines.len() > max_lines {
                let mut result = format!(
                    "... [Output truncated: showing last {} lines of {}]\n",
                    max_lines,
                    lines.len()
                );
                result.push_str(&lines[lines.len() - max_lines..].join("\n"));
                result
            } else {
                output.to_string()
            }
        }
        TrimMethod::Bytes => {
            if output.len() > limit.max_size {
                let mut result = output[..limit.max_size].to_string();
                result.push_str(&format!(
                    "\n... [Output truncated at {} bytes]",
                    limit.max_size
                ));
                result
            } else {
                output.to_string()
            }
        }
    }
}

pub fn create_read_file_tool() -> Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read lines from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-based, optional)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (1-based, optional)"
                    }
                },
                "required": ["path"]
            }
        }
    })
}

pub fn create_write_file_tool() -> Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append", "insert", "replace"],
                        "description": "Write mode: overwrite (default), append, insert at line, or replace line range"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number for insert/replace operations (1-based)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line number for replace operations (1-based)"
                    }
                },
                "required": ["path", "content"]
            }
        }
    })
}

pub fn read_file_lines(path: &str, start_line: Option<usize>, end_line: Option<usize>) -> String {
    use std::fs;

    let content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => return format!("Error reading file: {}", e),
    };

    let lines: Vec<&str> = content.lines().collect();
    let start = start_line.unwrap_or(1).saturating_sub(1); // Convert 1-based to 0-based
    let end = end_line
        .unwrap_or(lines.len())
        .saturating_sub(1)
        .min(lines.len()); // Convert 1-based to 0-based

    if start >= lines.len() {
        return "Start line exceeds file length".to_string();
    }

    if start >= end {
        return "Start line must be less than or equal to end line".to_string();
    }

    let mut result = String::new();
    for (i, line) in lines[start..end].iter().enumerate() {
        let line_num = start + i + 1;
        result.push_str(&format!("{:4}: {}\n", line_num, line));
    }

    if result.ends_with('\n') {
        result.pop();
    }

    result
}

pub fn show_write_diff(
    path: &str,
    content: &str,
    mode: Option<&str>,
    start_line: Option<usize>,
    end_line: Option<usize>,
) {
    use std::fs;

    let existing = fs::read_to_string(path).unwrap_or_default();
    let existing_lines: Vec<&str> = existing.lines().collect();

    match mode.unwrap_or("overwrite") {
        "overwrite" => {
            let new_lines: Vec<&str> = content.lines().collect();
            for (i, line) in existing_lines.iter().enumerate() {
                println!("\x1b[41m- {:3}     : {}\x1b[0m", i + 1, line);
            }
            for (i, line) in new_lines.iter().enumerate() {
                println!("\x1b[42m+ {:3}     : {}\x1b[0m", i + 1, line);
            }
        }
        "append" => {
            let start_line_num = existing_lines.len() + 1;
            for line in content.lines() {
                println!("\x1b[42m+ {:3}     : {}\x1b[0m", start_line_num, line);
            }
        }
        "insert" => {
            let insert_at = start_line.unwrap_or(1);
            for line in content.lines() {
                println!("\x1b[42m+ {:3}     : {}\x1b[0m", insert_at, line);
            }
        }
        "replace" => {
            let start = start_line.unwrap_or(1);
            let end = end_line.unwrap_or(start);
            let start_idx = start.saturating_sub(1);
            let end_idx = end.min(existing_lines.len());

            // Show context before
            let context_start = start_idx.saturating_sub(3);
            for i in context_start..start_idx {
                if i < existing_lines.len() {
                    println!("  {:3}     : {}", i + 1, existing_lines[i]);
                }
            }

            // Show removed lines
            for i in start_idx..end_idx {
                if i < existing_lines.len() {
                    println!("\x1b[41m- {:3}     : {}\x1b[0m", i + 1, existing_lines[i]);
                }
            }

            // Show added lines
            for (i, line) in content.lines().enumerate() {
                println!("\x1b[42m+ {:3}     : {}\x1b[0m", start + i, line);
            }

            // Show context after
            let context_end = (end_idx + 3).min(existing_lines.len());
            for (i, line) in existing_lines
                .iter()
                .enumerate()
                .take(context_end)
                .skip(end_idx)
            {
                println!("  {:3}     : {}", i + 1, line);
            }
        }
        _ => {}
    }

    println!();
    println!(" ⋮ ");
}

pub fn write_file_content(
    path: &str,
    content: &str,
    mode: Option<&str>,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> String {
    use std::fs;

    match mode.unwrap_or("overwrite") {
        "overwrite" => match fs::write(path, content) {
            Ok(_) => "File written successfully".to_string(),
            Err(e) => format!("Error writing file: {}", e),
        },
        "append" => match fs::OpenOptions::new().create(true).append(true).open(path) {
            Ok(mut file) => {
                use std::io::Write;
                match writeln!(file, "{}", content) {
                    Ok(_) => "Content appended successfully".to_string(),
                    Err(e) => format!("Error appending to file: {}", e),
                }
            }
            Err(e) => format!("Error opening file for append: {}", e),
        },
        "insert" => {
            let existing = fs::read_to_string(path).unwrap_or_default();
            let mut lines: Vec<&str> = existing.lines().collect();
            let insert_at = start_line.unwrap_or(1).saturating_sub(1);

            if insert_at <= lines.len() {
                lines.insert(insert_at, content);
                let new_content = lines.join("\n");
                match fs::write(path, new_content) {
                    Ok(_) => "Content inserted successfully".to_string(),
                    Err(e) => format!("Error inserting content: {}", e),
                }
            } else {
                "Insert line exceeds file length".to_string()
            }
        }
        "replace" => {
            let existing = match fs::read_to_string(path) {
                Ok(content) => content,
                Err(e) => return format!("Error reading file: {}", e),
            };
            let mut lines: Vec<&str> = existing.lines().collect();
            let start = start_line.unwrap_or(1).saturating_sub(1); // Convert 1-based to 0-based
            let end = end_line
                .unwrap_or(lines.len())
                .saturating_sub(1)
                .min(lines.len()); // Convert 1-based to 0-based

            if start < lines.len() {
                lines.splice(start..end, content.lines());
                let new_content = lines.join("\n");
                match fs::write(path, new_content) {
                    Ok(_) => "Content replaced successfully".to_string(),
                    Err(e) => format!("Error replacing content: {}", e),
                }
            } else {
                "Replace line exceeds file length".to_string()
            }
        }
        _ => "Invalid mode".to_string(),
    }
}
