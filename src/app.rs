use std::io::Write;
use std::{fs, io, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ollama::OllamaChatRequest;
use crate::{
    ollama::{OllamaChatMessage, OllamaChatResponse, OllamaChatResponseStreamingState},
    ApplicationConfig, ModelConfig,
};

#[derive(Serialize, Deserialize, Clone)]
pub struct ApplicationState {
    pub session_id: String,
    pub model: String,
    pub tools: Vec<Value>,
    pub messages: Vec<OllamaChatMessage>,
}

impl From<ApplicationState> for OllamaChatRequest {
    fn from(value: ApplicationState) -> Self {
        Self {
            model: value.model,
            messages: value.messages,
            tools: Some(value.tools),
            options: None,
            stream: false,
            think: false,
        }
    }
}

impl ApplicationState {
    pub fn new_from_config(app_config: &ApplicationConfig) -> Self {
        Self {
            session_id: String::default(),
            model: app_config.model.clone(),
            messages: Vec::new(),
            tools: Vec::new(),
        }
    }

    pub fn load_session(
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

    pub fn save_session(&self, session_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let filename = format!(".chatto-{}.yaml", session_name);
        let yaml_content = serde_yaml::to_string(&self)?;
        fs::write(&filename, yaml_content)?;
        Ok(())
    }

    /// A simple compact will keep the system message, and only the last assistant message. All
    /// tool calls and previous user messages will be deleted.
    pub fn simple_compact(&mut self) {
        todo!()
    }

    /// A summary compact will keep the system message. It will send all previous user,
    /// assistant, and tool messages to a model with instructions to summarize the messages into a
    /// single paragraph-ish length message.
    pub fn summary_compact(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        todo!()
    }

    pub fn model_configuration(&self, app_config: &ApplicationConfig) -> ModelConfig {
        match app_config.models.get(self.model.as_str()) {
            Some(model_config) => model_config.clone().to_owned(),
            None => ModelConfig::default(),
        }
    }

    pub fn add_user_message(&mut self, content: &str) {
        let message = OllamaChatMessage {
            role: "user".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_name: None,
            tool_call_id: None,
        };
        self.messages.push(message);
    }

    pub fn add_assistant_message(&mut self, content: &str) {
        let message = OllamaChatMessage {
            role: "assistant".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_name: None,
            tool_call_id: None,
        };
        self.messages.push(message);
    }

    pub fn add_assistant_response(&mut self, resp: OllamaChatResponse) {
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
                    println!("\nTool Call(s)...")
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

    pub fn add_tool_result(&mut self, tool_call_id: &str, tool_name: &str, result: &str) {
        let message = OllamaChatMessage {
            role: "tool".to_string(),
            content: result.to_string(),
            tool_calls: None,
            tool_name: Some(tool_name.to_string()),
            tool_call_id: Some(tool_call_id.to_string()),
        };
        self.messages.push(message);
    }

    pub fn should_prompt_user(&self) -> bool {
        // If the model hasn't been prompted yet, return true
        // Messages could be system->user->user (compact and rag might put documents/summaries as
        // user messages for context), still needs a user prompt to the model as it
        // hasn't been sent anything yet
        // If the assistant has left a message that needs a user response return true
        if self.messages.is_empty() {
            return true;
        }
        //Check if we are in an intitial state and the assistant hasn't responded yet
        if self
            .messages
            .iter()
            .filter(|m| m.role == "assistant")
            .count()
            == 0
        {
            return true;
        }
        // If last message is a tool (result) then we can just send those results to the assistant
        if self.messages.last().unwrap().role == "tool" {
            return false;
        }
        // Otherwise prompt user
        true
    }

    pub fn get_token_count_estimate(&self) -> usize {
        let messages_tokens = self
            .messages
            .iter()
            .map(|m| m.get_token_count_estimate())
            .sum::<usize>();

        let tools_tokens = self
            .tools
            .iter()
            .map(|tool| {
                let json_str = serde_json::to_string(tool).unwrap_or_default();
                (json_str.len() as f64 / 4.0) as usize // ~4 chars per token for JSON
            })
            .sum::<usize>();

        messages_tokens + tools_tokens
    }
}
