use std::{env, fs, io::Write, process::Command};

use reqwest::Client;
use rustyline::DefaultEditor;
use serde_json::json;
use tempfile::NamedTempFile;

use crate::{
    app::ApplicationState,
    ollama::{post_ollama_chat, OllamaChatMessage, OllamaChatRequest, ToolCall},
    tools::{
        create_read_file_tool, create_shell_tool, create_write_file_tool, execute_command,
        read_file_lines, show_write_diff, write_file_content,
    },
    ApplicationConfig,
};

pub async fn chat_mode(
    app_config: ApplicationConfig,
    mut session: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();

    let mut app_state = if let Some(ref session_name) = session {
        ApplicationState::load_session(session_name, &app_config)?
    } else {
        ApplicationState::new_from_config(&app_config)
    };
    app_state.tools = vec![
        create_shell_tool(),
        create_read_file_tool(),
        create_write_file_tool(),
    ];

    static DEFAULT_SYS_TOOLS_PROMPT: &str = r#"You are an AI assistant with access to specialized file tools and shell commands. ALWAYS prefer the dedicated file tools over shell commands for reading and writing files:

**PREFERRED FILE TOOLS:**
- `read_file`: Read file contents or specific line ranges. Use this instead of `cat`, `head`, `tail`, or `less`.
- `write_file`: Write, append, insert, or replace file content. Use this instead of `echo >`, `sed`, `awk`, or text editors.
- `execute_shell`: Use only for operations that cannot be done with file tools (compilation, git, directory operations, etc.).

**Shell commands for non-file operations:**
- `ls`: List directory contents.
- `git`: Version control operations.
- `make`, `cargo`, `gcc`: Build and compilation.
- `grep`: Search within command output (not files - use read_file + search).
- `find`: Locate files and directories.
- `chmod`, `chown`: File permissions.
- `curl`, `ssh`, `scp`: Network operations.

**IMPORTANT:** Always work within the current or project directory. Always use read_file for reading files and write_file for modifications. Only use shell commands when the file tools cannot accomplish the task.
"#;

    static DEFAULT_SYS_AGENT_PROMPT: &str = r#"You are working in the current directory, which is a codebase. Your role is to efficiently manage files and run commands. Here are your instructions:

1. **Use Dedicated File Tools**: ALWAYS use read_file for reading files and write_file for modifications. These tools are more reliable than shell commands for file operations.

2. **Focus on Specific Files and Lines**: Use read_file with line ranges to examine specific parts of files. Use write_file with insert/replace modes for targeted changes.

3. **Surgical Code Changes**: Make precise changes using write_file's replace mode for specific line ranges, or insert mode to add new code at exact locations.

4. **Shell Commands for Non-File Tasks**: Use execute_shell only for compilation, git operations, directory listing, and other non-file tasks.

5. **Compile and Test**: After file changes, use execute_shell for compilation (make, cargo, gcc, python) and testing.

6. **Loop Through Operations**: Read files to understand code, make targeted changes with write_file, then compile/test with execute_shell.

7. **Version Control**: Use execute_shell for git operations - commits, branches, status checks.

8. **Error Handling**: If operations fail, use read_file to examine error logs or configuration files, then use write_file to fix issues.

9. **Efficiency**: Minimize context by reading only necessary file sections and making targeted writes rather than rewriting entire files.

10. **File Tool Priority**: Remember - read_file and write_file are your primary tools. Use execute_shell as a secondary tool for everything else.

By following these instructions, you will efficiently manage the codebase with precise file operations and minimal context growth.
"#;
    let model_config = app_state.model_configuration(&app_config);

    let mut sys_content: String = if model_config.system_prompt.is_some() {
        model_config.system_prompt.unwrap()
    } else {
        DEFAULT_SYS_AGENT_PROMPT.to_string()
    };

    // Add AGENT.md context if available
    if let Some(agent_context) = load_agent_context() {
        sys_content += "\n\n## Project Context\n";
        sys_content += &agent_context;
    }

    if model_config.tools {
        sys_content += DEFAULT_SYS_TOOLS_PROMPT;
    }

    if app_state.messages.is_empty() {
        app_state.messages.push(OllamaChatMessage {
            role: "system".to_string(),
            content: sys_content,
            tool_calls: None,
            tool_name: None,
            tool_call_id: None,
        });
    } else {
        app_state.messages.insert(1, OllamaChatMessage {
            role: "system".to_string(),
            content: "REMINDER: You have access to execute_shell, read_file, and write_file tools. Use them to run commands and manage files.".to_string(),
            tool_calls: None,
                tool_name: None,
            tool_call_id: None,
        });
    }

    println!("Ollama URL: {}...", app_config.url);
    println!("Model: {}...", app_config.model);
    println!("Entering Chat mode with shell tools - type '/quit' to exit, '/compact' to force context compaction, '/editor' to open editor");

    let mut rl = DefaultEditor::new()?;

    //REPL Loop
    loop {
        //If there are tool calls that need to be executed, prompt user for those
        let mut temp_tool_calls: Vec<ToolCall> = Vec::new();
        if let Some(ref tool_calls) = app_state.messages.last().as_ref().unwrap().tool_calls {
            for tc in tool_calls {
                temp_tool_calls.push(tc.clone());
            }
        }
        if !temp_tool_calls.is_empty() {
            let tool_messages = process_tool_calls(&mut rl, &app_config, &temp_tool_calls);
            app_state.messages.extend(tool_messages);
        }

        //Prompt user for a message
        if app_state.should_prompt_user() {
            let max_context = app_config
                .models
                .get(app_state.model.as_str())
                .map(|m| m.num_ctx.unwrap_or(4096))
                .unwrap_or(4096);
            println!(
                "Waiting on your response... (Context {} tokens / Max {} tokens)",
                app_state.get_token_count_estimate(),
                max_context
            );

            let line = rl.readline("> ").unwrap();
            let input = {
                rl.add_history_entry(&line)?;
                line.trim().to_string()
            };
            let mut user_content: String = String::new();

            if input == "/quit" || input == "/exit" || input == "/done" {
                if let Some(ref session_name) = session {
                    app_state.save_session(session_name)?;
                    println!("Session saved: {}", session_name);
                }
                break;
            }

            if input.starts_with("/save") {
                let session_name = input.trim_start_matches("/save").trim();
                if session_name.is_empty() {
                    eprintln!("Error: session name cannot be empty");
                    continue;
                }
                if let Err(e) = app_state.save_session(session_name) {
                    eprintln!("Error saving session: {}", e);
                } else {
                    println!("Session saved: {}", session_name);
                }
                session = Some(session_name.to_string());
                continue;
            }

            if input == "/edit" || input == "/editor" {
                //This will open up the system defined or config defined editor like vim, emacs, or
                //nano. The whole session will be serialized as yaml and the user can edit any of it in
                //place. At the bottom is an open area where the user can type in a complex message.
                //When the file is closed, or the editor, then the whole file is read back in, the
                //messages array is updated and the new message is pushed as a user message.
                user_content = open_editor(&app_state.messages)?;
            }

            if input == "/tools" {
                //TODO Some models are not tool enabled. If we have a specific agentic tool enabled model
                //like function_gemma we can send the context to that model with specific sys prompt asking
                //it to review the agent response and see if any tool calls can be pulled out of it.
                //The response will then be appended to the last agent response and things will
                //continue from that point.
                todo!()
                //continue;
            }

            if input == "/reset" {
                app_state.messages.resize(1, OllamaChatMessage::default());
                continue;
            }

            if input == "/trim" {
                app_state.trim();
                continue;
            }

            if input == "/compact" {
                app_state.compact()?;
                continue;
            }

            if input == "/send" {
                //Skip creating a user message and just send the chat to the server as is
                println!("Forcing send without user message...");
            } else if user_content.is_empty() {
                app_state.add_user_message(&input);
            } else {
                app_state.add_user_message(&user_content);
            }
        }

        println!(
            "\nSending Request {} to {}...",
            app_state.model, app_config.url
        );

        //Call Ollamas chat endpoint
        let mut request: OllamaChatRequest = app_state.clone().into();
        request.stream = app_config.stream;
        request.think = false;
        if let Some(model_config) = app_config.models.get(app_state.model.as_str()) {
            request.think = model_config.think;
            if !model_config.tools {
                request.tools = None;
            }
            if model_config.num_ctx.is_some() {
                request.options = Some(json!({"num_ctx":model_config.num_ctx.unwrap()}));
            }
        }
        match post_ollama_chat(&client, app_config.url.as_str(), &request, Some(&mut app_state)).await {
            Ok(response) => {
                //Assume this is the last of the streaming messages, it's marked done
                println!(
                    "\nAssistant Finished... Prompt: {}, Eval: {}, Dur: {}",
                    response.prompt_eval_count.unwrap_or_default(),
                    response.eval_count.unwrap_or_default(),
                    response.total_duration.unwrap_or_default()
                );
            }
            Err(e) => {
                eprintln!("❌ API Error: {}", e);
                eprintln!("Please check:");
                eprintln!("  - Ollama is running (try: ollama serve)");
                eprintln!(
                    "  - Model '{}' is available (try: ollama list)",
                    app_config.model
                );
                eprintln!("  - URL '{}' is correct", app_config.url);
                return Err(e);
            }
        }
        println!();
    }

    Ok(())
}

/// Will take a vec of tool calls, prompt the user for approval and return a set of tool messages
fn process_tool_calls(
    rl: &mut DefaultEditor,
    app_config: &ApplicationConfig,
    tool_calls: &Vec<ToolCall>,
) -> Vec<OllamaChatMessage> {
    let mut ret: Vec<OllamaChatMessage> = Vec::new();
    for tc in tool_calls {
        match tc.function.name.as_str() {
            "execute_shell" => {
                println!(
                    "🛠️  Shell Command Requested!\n ● Command: {}\n ● Reason: {}",
                    tc.function.arguments.get("command").unwrap(),
                    tc.function.arguments.get("reason").unwrap()
                );
            }
            "read_file" => {
                let path = tc.function.arguments.get("path").unwrap().as_str().unwrap();
                let start_line = tc
                    .function
                    .arguments
                    .get("start_line")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                let end_line = tc
                    .function
                    .arguments
                    .get("end_line")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);

                println!("🛠️  Read File Requested!");
                match (start_line, end_line) {
                    (Some(start), Some(end)) => {
                        println!(" ● Reading file: {}, from line {} to {}", path, start, end)
                    }
                    (Some(start), None) => {
                        println!(" ● Reading file: {}, from line {} to end", path, start)
                    }
                    (None, Some(end)) => {
                        println!(" ● Reading file: {}, from start to line {}", path, end)
                    }
                    (None, None) => println!(" ● Reading file: {}", path),
                }
            }
            "write_file" => {
                let path = tc.function.arguments.get("path").unwrap().as_str().unwrap();
                let content = tc
                    .function
                    .arguments
                    .get("content")
                    .unwrap()
                    .as_str()
                    .unwrap();
                let mode = tc.function.arguments.get("mode").and_then(|v| v.as_str());
                let start_line = tc
                    .function
                    .arguments
                    .get("start_line")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                let end_line = tc
                    .function
                    .arguments
                    .get("end_line")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);

                println!("🛠️  Write File Requested!");
                match (mode.unwrap_or("overwrite"), start_line, end_line) {
                    ("replace", Some(start), Some(end)) => {
                        println!(" ● Replacing lines {}-{} in: {}", start, end, path)
                    }
                    ("replace", Some(start), None) => {
                        println!(" ● Replacing from line {} in: {}", start, path)
                    }
                    ("insert", Some(line), _) => {
                        println!(" ● Inserting at line {} in: {}", line, path)
                    }
                    ("append", _, _) => println!(" ● Appending to: {}", path),
                    _ => println!(" ● Writing to: {}", path),
                }
                println!();

                show_write_diff(path, content, mode, start_line, end_line);
            }
            _ => {
                println!("Tool Call {} Requested! Allow?", tc.function.name);
            }
        }
        let input = match rl.readline("y or no with reason/feedback > ") {
            Ok(line) => line.trim().to_string(),
            Err(_) => "ERROR".to_string(),
        };
        if input == "y" || input == "Y" {
            let tool_result = match tc.function.name.as_str() {
                "execute_shell" => execute_command(
                    tc.function
                        .arguments
                        .get("command")
                        .unwrap()
                        .as_str()
                        .unwrap(),
                    &app_config.output_limit,
                ),
                "read_file" => {
                    let path = tc.function.arguments.get("path").unwrap().as_str().unwrap();
                    let start_line = tc
                        .function
                        .arguments
                        .get("start_line")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize);
                    let end_line = tc
                        .function
                        .arguments
                        .get("end_line")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize);
                    let result = read_file_lines(path, start_line, end_line);
                    let byte_count = result.len();

                    // Count actual lines read
                    let line_count = result.lines().count();
                    let first_line = result
                        .lines()
                        .next()
                        .and_then(|line| line.split(':').next())
                        .and_then(|num| num.trim().parse::<usize>().ok())
                        .unwrap_or(1);
                    let last_line = first_line + line_count.saturating_sub(1);

                    if line_count > 0 {
                        println!(
                            " ✓ Successfully read {} bytes, {} lines ({}-{}) from {}",
                            byte_count, line_count, first_line, last_line, path
                        );
                    } else {
                        println!(" ✓ Successfully read {} bytes from {}", byte_count, path);
                    }

                    // Print what we're sending to the LLM
                    println!("OUTPUT:");
                    println!("{}", result);

                    result
                }
                "write_file" => {
                    let path = tc.function.arguments.get("path").unwrap().as_str().unwrap();
                    let content = tc
                        .function
                        .arguments
                        .get("content")
                        .unwrap()
                        .as_str()
                        .unwrap();
                    let mode = tc.function.arguments.get("mode").and_then(|v| v.as_str());
                    let start_line = tc
                        .function
                        .arguments
                        .get("start_line")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize);
                    let end_line = tc
                        .function
                        .arguments
                        .get("end_line")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize);
                    write_file_content(path, content, mode, start_line, end_line)
                }
                _ => format!("Unknown tool: {}", tc.function.name),
            };
            ret.push(OllamaChatMessage {
                role: "tool".to_string(),
                content: tool_result,
                tool_calls: None,
                tool_call_id: tc.id.clone(),
                tool_name: Some(tc.function.name.clone()),
            });
        } else {
            ret.push(OllamaChatMessage {
                role: "tool".to_string(),
                content: format!("TOOL CALL REJECTED. Feedback/Reason: {}", &input),
                tool_calls: None,
                tool_call_id: tc.id.clone(),
                tool_name: Some(tc.function.name.clone()),
            });
        }
    }
    ret
}

fn load_agent_context() -> Option<String> {
    let current_dir = env::current_dir().ok()?;
    let agent_file = current_dir.join("AGENT.md");

    if agent_file.exists() {
        fs::read_to_string(agent_file).ok()
    } else {
        None
    }
}

fn open_editor(messages: &Vec<OllamaChatMessage>) -> Result<String, Box<dyn std::error::Error>> {
    let yaml_content = serde_yaml::to_string(messages)?;

    let editor_content = format!("# Chatto API Request Editor\n# Edit the API request below, then add your message after the '---' separator\n\n{}\n\n---\n\n# Add your message below this line:\n", yaml_content);

    let mut temp_file = NamedTempFile::new()?;
    temp_file.write_all(editor_content.as_bytes())?;
    temp_file.flush()?;

    let editor = env::var("EDITOR").unwrap_or_else(|_| "nvim".to_string());

    let status = Command::new(&editor).arg(temp_file.path()).status()?;

    if !status.success() {
        return Err("Editor exited with error".into());
    }

    let edited_content = fs::read_to_string(temp_file.path())?;

    if let Some(separator_pos) = edited_content.rfind("\n---\n") {
        let message_part = &edited_content[separator_pos + 5..];

        let message = message_part
            .lines()
            .skip_while(|line| line.trim().is_empty() || line.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();

        if !message.is_empty() {
            return Ok(message);
        }
    }

    Err("No message found after separator".into())
}
