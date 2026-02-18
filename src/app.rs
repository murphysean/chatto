use std::io::Write;
use std::{fs, io, path::Path};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::ollama::{post_ollama_chat, OllamaChatRequest, StreamingChatHandler};
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

impl StreamingChatHandler for &mut ApplicationState {
    fn process_streaming_response(
        &mut self,
        previous_streaming_state: &OllamaChatResponseStreamingState,
        current_streaming_state: &OllamaChatResponseStreamingState,
        response: &OllamaChatResponse,
    ) {
        if let Some(message) = &response.message {
            if let Some(thinking) = &message.thinking {
                //If it's a thought chunk, just send it to console
                if !thinking.is_empty() {
                    if matches!(
                        previous_streaming_state,
                        OllamaChatResponseStreamingState::Receiving
                    ) && matches!(
                        current_streaming_state,
                        OllamaChatResponseStreamingState::Thinking
                    ) {
                        println!("Assistant Thinking...")
                    }
                    print!("\x1b[90m{}\x1b[0m", thinking);
                    io::stdout().flush().unwrap_or_default();
                    return;
                }
            }
            //Regular messages get saved into the last message
            if !message.content.is_empty() {
                if matches!(
                    previous_streaming_state,
                    OllamaChatResponseStreamingState::Receiving
                        | OllamaChatResponseStreamingState::Thinking
                ) && matches!(
                    current_streaming_state,
                    OllamaChatResponseStreamingState::Responding
                ) {
                    println!("\nAssistant:");
                }
                print!("{}", message.content);
                io::stdout().flush().unwrap_or_default();
                return;
            }
            //Tool calls will come through here as well
            if message.tool_calls.is_some()
                && matches!(
                    previous_streaming_state,
                    OllamaChatResponseStreamingState::Receiving
                        | OllamaChatResponseStreamingState::Thinking
                        | OllamaChatResponseStreamingState::Responding
                )
                && matches!(
                    current_streaming_state,
                    OllamaChatResponseStreamingState::CallingTools
                )
            {
                println!("\nTool Call(s)...");
            }
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
        let filename = format!(".chatto-{}.session.yaml", session_name);
        let yaml_content = serde_yaml::to_string(&self)?;
        fs::write(&filename, yaml_content)?;
        Ok(())
    }

    /// A simple compact will keep the system message, and only the last assistant message. All
    /// tool calls and previous user messages will be deleted.
    pub fn trim(&mut self) {
        let mut new_messages: Vec<OllamaChatMessage> = Vec::new();
        //All System messages
        new_messages.extend(self.messages.iter().filter(|m| m.role == "system").cloned());
        //The last user message
        if let Some(i) = self.messages.iter().rposition(|m| m.role == "user") {
            new_messages.push(self.messages.get(i).unwrap().clone());
        }
        //The last assistant message
        if let Some(i) = self.messages.iter().rposition(|m| m.role == "assistant") {
            new_messages.push(self.messages.get(i).unwrap().clone());
        }
        self.messages = new_messages;
    }

    /// A summary compact will keep the system message. It will send all previous user,
    /// assistant, and tool messages to a model with instructions to summarize the messages into a
    /// single paragraph-ish length message.
    pub async fn compact(
        &mut self,
        client: &Client,
        config: &ApplicationConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut messages: Vec<OllamaChatMessage> = Vec::new();
        messages.push(OllamaChatMessage {
            role: "system".to_string(),
            content: "Summarize the chat into a single paragraph that will give the agent sufficient context to continue the conversation but reduce the context. Please summarize tool calls and tool results as well as the user and assistant messages. Include a header for your messages that clarifies it is a summary.".to_string(),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
        });
        let mut content: String = String::new();
        self.messages.iter().skip(1).for_each(|m| {
            if m.role == "tool" {
                content.push_str(
                    format!(
                        "{}, id: {}, name: {}:\n--- {}\n",
                        m.role,
                        m.tool_call_id.clone().unwrap_or_default(),
                        m.tool_name.clone().unwrap_or_default(),
                        m.content
                    )
                    .as_str(),
                );
            } else {
                content.push_str(format!("{}:\n--- {}\n", m.role, m.content).as_str());
            }
            if let Some(tool_calls) = &m.tool_calls {
                tool_calls.iter().for_each(|tc| {
                    content.push_str(
                        format!("Tool Call {}:\n", tc.id.clone().unwrap_or_default()).as_str(),
                    );
                    content.push_str(
                        format!(
                            "\tname: {}\n\targuments: {}\n",
                            tc.function.name, tc.function.arguments
                        )
                        .as_str(),
                    );
                });
            }
            content.push_str("---\n");
        });
        messages.push(OllamaChatMessage {
            role: "user".to_string(),
            content,
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
        });
        let request: OllamaChatRequest = OllamaChatRequest {
            model: self.model.clone(),
            messages,
            tools: None,
            options: config.models.get(self.model.as_str()).as_ref().map(|c| {
                if let Some(num_ctx) = c.num_ctx {
                    json!({"num_ctx": num_ctx})
                } else {
                    Value::Null
                }
            }),
            stream: false,
            think: false,
        };
        let (response, _) = post_ollama_chat(
            client,
            &config.url,
            &config.api_key,
            &request,
            Option::<&mut ApplicationState>::None,
        )
        .await?;
        self.messages.truncate(1);
        self.messages.push(OllamaChatMessage {
            role: "user".to_string(),
            content: response.message.unwrap_or_default().content,
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
        });
        Ok(())
    }

    /// Will call a specialized model like functiongemma with the last user and assistant message
    /// to see if any tool calls can be extracted given the set of available tools.
    pub async fn get_tool_calls(
        &mut self,
        client: &Client,
        config: &ApplicationConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut messages: Vec<OllamaChatMessage> = Vec::new();
        messages.push(OllamaChatMessage {
            role: "system".to_string(),
            content: "Review the given conversation snippet and identify if there are any tool calls you can make based on the user message, or the assistant response. If the assistant gives a code snippet, try to write that to the file. If a shell command is included, try to run that. If the model needs to read contents of a file, do that. If a search is warranted, run grep or find.".to_string(),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
        });
        let mut content: String = String::new();
        /*
        //The last user message
        if let Some(i) = self.messages.iter().rposition(|m| m.role == "user") {
            content.push_str(format!("user: {}\n", self.messages.get(i).unwrap().content).as_str());
        }*/
        //The last assistant message
        if let Some(i) = self.messages.iter().rposition(|m| m.role == "assistant") {
            content.push_str(
                format!("assistant: {}\n", self.messages.get(i).unwrap().content).as_str(),
            );
        }
        messages.push(OllamaChatMessage {
            role: "user".to_string(),
            content,
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
        });
        let request: OllamaChatRequest = OllamaChatRequest {
            model: "functiongemma".to_string(),
            messages,
            tools: Some(self.tools.clone()),
            options: config.models.get("functiongemma").as_ref().map(|m| {
                if let Some(num_ctx) = m.num_ctx {
                    json!({"num_ctx": num_ctx})
                } else {
                    Value::Null
                }
            }),
            stream: false,
            think: false,
        };
        let (response, _) = post_ollama_chat(
            client,
            &config.url,
            &config.api_key,
            &request,
            Option::<&mut ApplicationState>::None,
        )
        .await?;

        if let Some(message) = response.message {
            self.messages.push(OllamaChatMessage {
                role: "user".to_string(),
                content: message.content.clone(),
                tool_calls: message.tool_calls.clone(),
                tool_call_id: None,
                tool_name: None,
            });
        }
        Ok(())
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

    /// Will Print out the response contents for the user
    pub fn print_assistant_response(&self, resp: &OllamaChatResponse) {
        if let Some(message) = &resp.message {
            if let Some(thinking) = &message.thinking {
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

    /// Will convert the response into a chat message and store it in the messages vec
    pub fn add_assistant_response(&mut self, resp: OllamaChatResponse) {
        if let Some(message) = resp.message {
            let new_message = OllamaChatMessage {
                role: "assistant".to_string(),
                content: message.content.clone(),
                tool_calls: message.tool_calls.clone(),
                tool_name: None,
                tool_call_id: None,
            };
            self.messages.push(new_message);
        }
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
