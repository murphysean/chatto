use reqwest::Client;
use rustyline::DefaultEditor;

use crate::{
    load_agent_context,
    ollama::{post_ollama_chat, OllamaChatMessage, ToolCall},
    tools::{
        create_read_file_tool, create_shell_tool, create_write_file_tool, execute_command,
        read_file_lines, show_write_diff, write_file_content,
    },
    ApplicationConfig, ApplicationState,
};

pub async fn chat_mode(
    app_config: ApplicationConfig,
    session: Option<String>,
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

            if input == "/quit" {
                if let Some(ref session_name) = session {
                    app_state.save_session(session_name)?;
                    println!("Session saved: {}", session_name);
                }
                break;
            }

            //TODO Add back in /compact
            //Will send all messages to model with a different system message asking the model to
            //read all messages and compact to a paragraph. We will send this out and then include
            //it as a single user message after the system message, resetting the app_state
            //messages to only two messages the system message, and the user message providing the
            //summary (Probably surrounded with some sort of contextual md like header footer to
            //let the model know this is a summary document/message)
            if input == "/compact" {
                app_state.messages.resize(1, OllamaChatMessage::default());
            }

            app_state.add_user_message(&input);
        }
        println!(
            "\nSending Request {} to {}...",
            app_state.model, app_config.url
        );

        //Call Ollamas chat endpoint
        if let Err(e) = post_ollama_chat(&client, &app_config, &mut app_state).await {
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
        println!();

        //Prompt user for tool calls
        let mut temp_tool_calls: Vec<ToolCall> = Vec::new();
        if let Some(ref tool_calls) = app_state.messages.last().as_ref().unwrap().tool_calls {
            for tc in tool_calls {
                temp_tool_calls.push(tc.clone());
            }
        }
        for tc in &temp_tool_calls {
            match tc.function.name.as_str() {
                "execute_shell" => {
                    println!(
                        "🛠️  Tool Call {} Requested! Allow?\nCommand: {}\nReason: {}",
                        tc.function.name,
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

                    println!("🛠️  Using tool: read_file");
                    println!(" ⋮ ");
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

                    println!("🛠️  Using tool: write_file");
                    println!(" ⋮ ");
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
                app_state.add_tool_result(
                    tc.id.as_ref().unwrap(),
                    tc.function.name.as_str(),
                    &tool_result,
                );
            } else {
                app_state.add_tool_result(
                    tc.id.as_ref().unwrap(),
                    tc.function.name.as_str(),
                    format!("TOOL CALL REJECTED. Feedback/Reason: {}", &input).as_str(),
                );
            }
        }
    }

    Ok(())
}
