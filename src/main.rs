//! # Chatto
//!
//! A CLI tool for contextual chat interactions with Ollama AI models.
//!
//! Chatto provides a command-line interface for interacting with Ollama instances,
//! supporting contextual conversations, system prompts, and tool execution.
//!
//! ## Features
//!
//! - **Contextual Chat**: Maintains conversation history for multi-turn interactions
//! - **Tool Support**: Execute shell commands, read/write files with AI assistance
//! - **Session Management**: Save and load conversation sessions
//! - **Streaming Responses**: Real-time streaming of AI responses
//! - **Model Management**: List available models and query model capabilities
//! - **Context Compaction**: Automatically summarize conversation history
//!
//! ## Commands
//!
//! - `chat`: Start an interactive chat session with an Ollama model
//! - `list`: List all available models on the Ollama instance
//!
//! ## Configuration
//!
//! Configuration is loaded from (in order of priority):
//! 1. Command-line arguments
//! 2. `.chatto.yaml` in the current directory
//! 3. `~/.config/chatto/Config.yaml`
//! 4. Default values
//!
//! ## Example
//!
//! ```bash
//! # Start a chat session
//! chatto chat --url http://localhost:11434 --model llama2
//!
//! # List available models
//! chatto list --url http://localhost:11434
//! ```

use clap::{Parser, Subcommand};
use config::Config;
use reqwest::Client;
use serde::Deserialize;
use std::path::Path;

use crate::chat::chat_mode;
use crate::ollama::{list_models, show_model, OllamaModel};
use crate::tools::OutputLimit;

pub mod app;
pub mod chat;
pub mod ollama;
pub mod tools;

/// CLI argument structure for the chatto command
#[derive(Parser)]
#[command(name = "chatto")]
#[command(about = "A CLI chat interface for Ollama")]
struct Cli {
    /// The base url for the ollama instance eg. http://192.168.0.1:11434
    #[arg(short, long)]
    url: Option<String>,
    /// The model to be used on the ollama instance eg. gemma3:12b or llama2
    #[arg(short, long)]
    model: Option<String>,
    /// API key for authentication (can also be set via OLLAMA_API_KEY env var)
    #[arg(short, long, env = "OLLAMA_API_KEY")]
    key: Option<String>,
    #[command(subcommand)]
    command: Commands,
}

/// Available CLI commands for chatto
#[derive(Subcommand)]
enum Commands {
    /// List all available models on the Ollama instance
    List,
    /// Start an interactive chat session with context management and tool support
    Chat {
        /// Disable streaming responses (receive complete response at once)
        #[arg(short, long)]
        disable_streaming: bool,
        /// Load or create a named session (saves to .chatto-{session}.session.yaml)
        #[arg(short, long)]
        session: Option<String>,
    },
}

/// Application configuration loaded from config files and CLI arguments
#[derive(Debug, Deserialize)]
pub struct ApplicationConfig {
    /// Base URL of the Ollama instance
    url: String,
    /// API key for authentication
    api_key: String,
    /// Default model to use for chat
    model: String,
    /// Whether to stream responses
    stream: bool,
    /// Output size limits for tool execution
    output_limit: OutputLimit,
    /// Cached model information
    models: Vec<OllamaModel>,
}

impl ApplicationConfig {
    /// Retrieves model configuration by name from the cached model list
    ///
    /// # Arguments
    /// * `model` - The model name to look up
    ///
    /// # Returns
    /// Reference to the model configuration if found
    pub fn get_model(&self, model: &str) -> Option<&OllamaModel> {
        self.models
            .iter()
            .find(|&m| m.name == model)
            .map(|v| v as _)
    }

    /// Merges model configuration into the cached list
    ///
    /// If the model already exists, updates its capabilities and info.
    /// If not, adds it to the list.
    ///
    /// # Arguments
    /// * `model` - Model configuration to merge
    pub fn merge_model(&mut self, model: OllamaModel) {
        let found = self.models.iter().any(|m| m.name == model.name);
        if !found {
            self.models.push(model);
            return;
        }
        for m in self.models.iter_mut() {
            if m.name == model.name {
                m.capabilities = model.capabilities;
                m.model_info = model.model_info;
                break;
            }
        }
    }
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
    let config_builder = Config::builder()
        .set_default("url", "https://localhost:11434")?
        .set_default("model", "llama2")?
        .set_default("stream", "true")?
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
        .set_override_option("api_key", cli.key)?
        .set_override_option("model", cli.model)?;

    let client = Client::new();
    let mut app_config: ApplicationConfig = config_builder.build()?.try_deserialize()?;
    //Pull down the model config from the server
    let model_info = show_model(
        &client,
        &app_config.url,
        &app_config.api_key,
        &app_config.model,
    )
    .await?;
    app_config.merge_model(model_info);

    match cli.command {
        Commands::List => {
            let models = list_models(&client, &app_config.url, &app_config.api_key).await?;
            println!("Available Models:");
            for model in models {
                let model_detail =
                    show_model(&client, &app_config.url, &app_config.api_key, &model.name).await?;
                println!(
                    "- {} ctx: {:?}",
                    model.name,
                    model_detail.get_context_length()
                );
                for c in model_detail.capabilities {
                    println!("  - {}", c);
                }
            }
        }
        Commands::Chat {
            disable_streaming,
            session,
        } => {
            if disable_streaming {
                app_config.stream = false;
            }
            chat_mode(&client, app_config, session).await?;
        }
    }

    Ok(())
}
