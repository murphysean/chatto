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
    #[arg(short, long, env = "OLLAMA_API_KEY")]
    key: Option<String>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    List,
    Chat {
        #[arg(short, long)]
        session: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
pub struct ApplicationConfig {
    url: String,
    api_key: String,
    model: String,
    stream: bool,
    output_limit: OutputLimit,
    models: Vec<OllamaModel>,
}

impl ApplicationConfig {
    pub fn get_model(&self, model: &str) -> Option<&OllamaModel> {
        self.models
            .iter()
            .find(|&m| m.name == model)
            .map(|v| v as _)
    }

    pub fn merge_model(&mut self, model: OllamaModel) {
        println!("MERGING {:?}", model);
        let found = self.models.iter().any(|m| m.name == model.name);
        if !found {
            println!("PUSHED");
            self.models.push(model);
            return;
        }
        for m in self.models.iter_mut() {
            if m.name == model.name {
                println!("FOUND");

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
    println!("Fetching Model Info...");
    let model_info = show_model(
        &client,
        &app_config.url,
        &app_config.api_key,
        &app_config.model,
    )
    .await?;
    println!("Model Info {:?}", &model_info);
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
        Commands::Chat { session } => {
            chat_mode(&client, app_config, session).await?;
        }
    }

    Ok(())
}
