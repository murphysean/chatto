use serde::{Deserialize, Serialize};
use std::fs;
use std::error::Error;
use std::path::Path;

use crate::ollama::OllamaChatMessage;

pub type ChatMessage = OllamaChatMessage;

#[derive(Serialize, Deserialize)]
pub struct ConversationContext {
    pub messages: Vec<ChatMessage>,
}

impl Default for ConversationContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationContext {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub fn from(messages: Vec<ChatMessage>) -> Self {
        Self { messages }
    }

    pub fn load(session_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session_path = format!("{}.json", session_name);
        if Path::new(&session_path).exists() {
            let content = fs::read_to_string(&session_path)?;
            let context: Self = serde_json::from_str(&content)?;
            Ok(context)
        } else {
            Ok(Self::new())
        }
    }

    pub fn save(&self, session_name: &str) -> Result<(), Box<dyn Error>> {
        let session_path = format!("{}.json", session_name);
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&session_path, content)?;
        Ok(())
    }

}
