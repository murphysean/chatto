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

impl Default for OutputLimit {
    fn default() -> Self {
        Self {
            max_size: 2048,
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
