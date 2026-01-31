use clap::{Parser, Subcommand};
use config::Config;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::chat::chat_mode;
use crate::tools::OutputLimit;

pub mod app;
pub mod chat;
pub mod ollama;
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
