use clap::{Parser, Subcommand};
use config::Config;
use reqwest::Client;
use rustyline::DefaultEditor;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use crate::ollama::{post_ollama_chat, OllamaChatResponseStreamingState};
use crate::ollama::{OllamaChatMessage, OllamaChatResponse, ToolCall};
use crate::tools::{create_shell_tool, execute_command, OutputLimit};

pub mod ollama;
pub mod session;
pub mod tools;

#[derive(Parser)]
#[command(name = "chatto")]
#[command(about = "A CLI chat interface for Ollama")]
struct Cli {
    ///The base url for the ollama instance eg. http://192.168.0.1:11434
    #[arg(short, long)]
    url: Option<String>,
    ///The model to be used on the ollama instance eg. gemma3:12b or llama2
    #[arg(short, long)]
    model: Option<String>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Chat {
        #[arg(short, long)]
        session: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
pub struct ApplicationConfig {
    url: String,
    model: String,
    stream: bool,
    output_limit: OutputLimit,
    models: HashMap<String, ModelConfig>,
}

#[derive(Default, Debug, Deserialize, Clone)]
pub struct ModelConfig {
    system_prompt: Option<String>,
    think: bool,
    tools: bool,
    num_ctx: Option<u64>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ApplicationState {
    session_id: String,
    model: String,
    tools: Vec<Value>,
    messages: Vec<OllamaChatMessage>,
}

impl ApplicationState {
    fn new_from_config(app_config: &ApplicationConfig) -> Self {
        Self {
            session_id: String::default(),
            model: app_config.model.clone(),
            messages: Vec::new(),
            tools: Vec::new(),
        }
    }

    fn load_session(
        session_name: &str,
        app_config: &ApplicationConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let filename = format!(".chatto-{}.session.yaml", session_name);
        if Path::new(&filename).exists() {
            let content = fs::read_to_string(&filename)?;
            let app_state: ApplicationState = serde_yaml::from_str(&content)?;
            println!("Loaded session: {}", session_name);
            Ok(app_state)
        } else {
            println!("Starting new session: {}", session_name);
            Ok(Self::new_from_config(app_config))
        }
    }

    fn model_configuration(&self, app_config: &ApplicationConfig) -> ModelConfig {
        match app_config.models.get(self.model.as_str()) {
            Some(model_config) => model_config.clone().to_owned(),
            None => ModelConfig::default(),
        }
    }

    fn save_session(&self, session_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let filename = format!(".chatto-{}.yaml", session_name);
        let yaml_content = serde_yaml::to_string(&self)?;
        fs::write(&filename, yaml_content)?;
        Ok(())
    }

    fn add_user_message(&mut self, content: &str) {
        let message = OllamaChatMessage {
            role: "user".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_name: None,
            tool_call_id: None,
        };
        self.messages.push(message);
    }

    fn add_assistant_message(&mut self, content: &str) {
        let message = OllamaChatMessage {
            role: "assistant".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_name: None,
            tool_call_id: None,
        };
        self.messages.push(message);
    }

    fn add_assistant_response(&mut self, resp: OllamaChatResponse) {
        if let Some(message) = resp.message {
            let new_message = OllamaChatMessage {
                role: message.role,
                content: message.content.clone(),
                tool_calls: message.tool_calls.clone(),
                tool_name: None,
                tool_call_id: None,
            };
            self.messages.push(new_message);
            if let Some(thinking) = message.thinking {
                if !thinking.is_empty() {
                    println!("Assistant Thought:\n\x1b[90m{}\x1b[0m", thinking);
                }
            }
            if !message.content.is_empty() {
                println!("Assistant:\n{}", message.content);
            }
            if message.tool_calls.is_some() {
                println!("Assistant Requests Tool Call(s)");
            }
        }
    }

    pub fn process_assistant_message_chunk(
        &mut self,
        prev_state: OllamaChatResponseStreamingState,
        resp: OllamaChatResponse,
    ) -> OllamaChatResponseStreamingState {
        let mut streaming_state = OllamaChatResponseStreamingState::Recieving;
        if let Some(message) = resp.message {
            if let Some(thinking) = message.thinking {
                //If it's a thought chunk, just send it to console
                if !thinking.is_empty() {
                    if let OllamaChatResponseStreamingState::Recieving = prev_state {
                        println!("Assistant Thinking...")
                    }
                    print!("\x1b[90m{}\x1b[0m", thinking);
                    io::stdout().flush().unwrap_or_default();
                    streaming_state = OllamaChatResponseStreamingState::Thinking;
                }
            }
            //Regular messages get saved into the last message
            if !message.content.is_empty() {
                if let OllamaChatResponseStreamingState::Thinking = prev_state {
                    println!("\nAssistant:")
                }
                if let Some(last_message) = self.messages.last_mut() {
                    last_message.content += &message.content;
                }
                print!("{}", message.content);
                io::stdout().flush().unwrap_or_default();
                streaming_state = OllamaChatResponseStreamingState::Responding;
            }
            //Tool calls will come through here as well
            if let Some(tool_calls) = message.tool_calls {
                if let Some(last_message) = self.messages.last_mut() {
                    if let Some(lm_tool_calls) = last_message.tool_calls.as_mut() {
                        lm_tool_calls.extend(tool_calls);
                    } else {
                        last_message.tool_calls = Some(tool_calls);
                    }
                }
                if let OllamaChatResponseStreamingState::Responding = prev_state {
                    println!("Tool Call...")
                }
                streaming_state = OllamaChatResponseStreamingState::CallingTools;
            }
        }
        streaming_state
    }

    pub fn finalize_assistant_message(&mut self, resp: &OllamaChatResponse) {
        //Assume this is the last of the streaming messages, it's marked done
        println!(
            "\nAssistant Finished... Prompt: {}, Eval: {}, Dur: {}",
            resp.prompt_eval_count.unwrap_or_default(),
            resp.eval_count.unwrap_or_default(),
            resp.total_duration.unwrap_or_default()
        );
    }

    fn add_tool_result(&mut self, tool_call_id: &str, tool_name: &str, result: &str) {
        let message = OllamaChatMessage {
            role: "tool".to_string(),
            content: result.to_string(),
            tool_calls: None,
            tool_name: Some(tool_name.to_string()),
            tool_call_id: Some(tool_call_id.to_string()),
        };
        self.messages.push(message);
    }
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

async fn chat_mode(
    app_config: ApplicationConfig,
    session: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();

    let mut app_state = if let Some(ref session_name) = session {
        ApplicationState::load_session(session_name, &app_config)?
    } else {
        ApplicationState::new_from_config(&app_config)
    };
    app_state.tools = vec![create_shell_tool()];

    static DEFAULT_SYS_TOOLS_PROMPT: &str = r#"You are an AI assistant with access to a shell prompt. Your primary tool is the shell prompt, which allows you to execute commands and interact with the file system. Here are some common commands and tools you might use:

- `ls`: List directory contents.
- `cd`: Change the current directory.
- `cat`: Concatenate and display file content.
- `grep`: Search text using patterns.
- `sed`: Stream editor for filtering and transforming text.
- `awk`: Pattern scanning and processing language.
- `git`: Version control system for tracking changes in source code.
- `make`: Build automation tool.
- `gcc`: GNU Compiler Collection for compiling C programs.
- `python`: Python interpreter for running Python scripts.
- `vim`: Text editor for editing files.
- `chmod`: Change file mode bits.
- `chown`: Change file owner and group.
- `find`: Search for files in a directory hierarchy.
- `diff`: Compare files line by line.
- `tar`: Archive files.
- `curl`: Transfer data with URLs.
- `ssh`: Secure Shell client for remote login.
- `scp`: Secure copy protocol for transferring files.
- `rsync`: Remote data synchronization tool.

Use these tools to write, review, compile, and set up codebases. Always ensure that your commands are precise and minimize the context growth by focusing on specific files and lines of code.
"#;

    static DEFAULT_SYS_AGENT_PROMPT: &str = r#"You are working in the current directory, which is a codebase. Your role is to run commands that limit the context growth. Here are some specific instructions:

1. **Focus on Specific Files and Lines**: Always look at specific lines of code, grep for specific functions, and identify the main starting point of the codebase. Avoid loading entire files into memory unless necessary.

2. **Surgical Code Changes**: Make surgical changes to the code using tools like `sed` for text manipulation. Ensure that your changes are minimal and targeted.

3. **Compile and Test**: After making changes, compile the code using appropriate tools (e.g., `make`, `gcc`, `python`) and run tests to ensure that the changes do not introduce new issues.

4. **Loop Through Commands**: Work in a loop where you run commands, look at the output, fix any issues, and repeat the process until the task is completed.

5. **Use Version Control**: Utilize `git` for version control. Commit changes frequently with meaningful commit messages. Use branches to isolate changes and merge them back into the main branch once tested.

6. **Error Handling**: If a command fails, analyze the error message and take appropriate action to fix the issue. Use tools like `grep` and `awk` to parse error messages and logs.

7. **Documentation**: Keep a log of the commands you run and the changes you make. This will help in tracking the progress and debugging any issues that arise.

8. **Security**: Be mindful of security best practices. Avoid running commands that could compromise the system or expose sensitive information.

9. **Efficiency**: Optimize your commands to be as efficient as possible. Avoid unnecessary operations and minimize the use of resources.

10. **Feedback Loop**: Continuously review the output of your commands and adjust your approach based on the feedback received.

By following these instructions, you will be able to effectively manage the codebase and ensure that your changes are accurate and efficient.

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
            content: "REMINDER: You have access to the execute_shell tool. Use it to run commands like ls, cat, grep, etc.".to_string(),
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
        if app_state.messages.last().unwrap().role == "assistant"
            || app_state.messages.last().unwrap().role == "system"
        {
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

            app_state.add_user_message(&input);
        }
        println!("\nSending Request...");

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
            println!(
                "Tool Call {} Requested! Allow?\nCommand: {}\nReason: {}",
                tc.function.name,
                tc.function.arguments.get("command").unwrap(),
                tc.function.arguments.get("reason").unwrap()
            );
            let input = match rl.readline("y or no with reason/feedback > ") {
                Ok(line) => line.trim().to_string(),
                Err(_) => "ERROR".to_string(),
            };
            if input == "y" || input == "Y" {
                let tool_result = execute_command(
                    tc.function
                        .arguments
                        .get("command")
                        .unwrap()
                        .as_str()
                        .unwrap(),
                    &app_config.output_limit,
                );
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    //Configs that we should pull in
    //1. General App Config
    //2. Model Config
    //3. Agent Config -> Might have model config in it
    //4. Prompt Config
    //5. MCP/Tools Config

    //Check for config
    let app_config: ApplicationConfig = Config::builder()
        .set_default("url", "https://localhost:11434")?
        .set_default("model", "llama2")?
        .set_default("streaming", "true")?
        .set_default("output_limit", OutputLimit::default())?
        .add_source(config::File::with_name(".chatto.yaml").required(false))
        .add_source(
            config::File::from(
                dir::home_dir()
                    .ok_or("Home Dir not found".to_string())?
                    .join(Path::new(".config/chatto/Config.yaml")),
            )
            .required(false),
        )
        .set_override_option("url", cli.url)?
        .set_override_option("model", cli.model)?
        .build()?
        .try_deserialize()?;

    match cli.command {
        Commands::Chat { session } => {
            chat_mode(app_config, session).await?;
        }
    }

    Ok(())
}
