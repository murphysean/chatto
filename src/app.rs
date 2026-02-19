//! Application state management and chat message handling.
//!
//! This module contains the core application state for chat sessions,
//! including message history, tool definitions, and session persistence.
//! It implements the streaming chat handler for real-time response display.

use std::io::Write;
use std::{fs, io, path::Path};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ollama::{post_ollama_chat, OllamaChatRequest, StreamingChatHandler};
use crate::{
    ollama::{OllamaChatMessage, OllamaChatResponse, OllamaChatResponseStreamingState},
    ApplicationConfig,
};

/// Application state for a chat session
///
/// Maintains the conversation history, tool definitions, and session metadata.
/// Can be serialized/deserialized for session persistence.
#[derive(Serialize, Deserialize, Clone)]
pub struct ApplicationState {
    /// Unique identifier for this session
    pub session_id: String,
    /// Model name being used for this session
    pub model: String,
    /// Available tools for the model to use
    pub tools: Vec<Value>,
    /// Conversation history
    pub messages: Vec<OllamaChatMessage>,
}

/// Converts ApplicationState into an OllamaChatRequest for API submission.
///
/// Extracts the model, messages, and tools from the application state
/// to form a complete request. Note: stream and think are set to false
/// by default and should be overridden as needed.
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

/// Handles streaming response chunks for real-time display.
///
/// Implements the StreamingChatHandler trait to print thinking (in gray),
/// response content, and tool call indicators as they arrive during streaming.
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
    /// Creates a new application state from configuration
    ///
    /// # Arguments
    /// * `app_config` - Application configuration to initialize from
    ///
    /// # Returns
    /// New ApplicationState with empty message history
    pub fn new_from_config(app_config: &ApplicationConfig) -> Self {
        Self {
            session_id: String::default(),
            model: app_config.model.clone(),
            messages: Vec::new(),
            tools: Vec::new(),
        }
    }

    /// Loads a session from disk or creates a new one if it doesn't exist
    ///
    /// # Arguments
    /// * `session_name` - Name of the session to load
    /// * `app_config` - Application configuration for new session creation
    ///
    /// # Returns
    /// Loaded or newly created ApplicationState
    ///
    /// # Errors
    /// Returns error if session file exists but cannot be read or parsed
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

    /// Saves the current session to disk
    ///
    /// # Arguments
    /// * `session_name` - Name for the session file
    ///
    /// # Errors
    /// Returns error if file cannot be written
    pub fn save_session(&self, session_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let filename = format!(".chatto-{}.session.yaml", session_name);
        let yaml_content = serde_yaml::to_string(&self)?;
        fs::write(&filename, yaml_content)?;
        Ok(())
    }

    /// Trims message history to keep only essential messages
    ///
    /// Keeps system message, last user message, and last assistant message.
    /// All tool calls and previous messages are removed.
    /// Useful for reducing context size while maintaining conversation continuity.
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

    /// Compacts message history by summarizing previous messages
    ///
    /// Sends all messages (except system) to the model for summarization.
    /// The summary replaces the message history, keeping context small
    /// while preserving conversation continuity.
    ///
    /// # Arguments
    /// * `client` - HTTP client for API requests
    /// * `config` - Application configuration
    ///
    /// # Returns
    /// Result indicating success or failure of compaction
    ///
    /// # Errors
    /// Returns error if summarization request fails
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
            options: config
                .get_model(self.model.as_str())
                .and_then(|m| m.options.clone()),
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

    /// Extracts tool calls from conversation using a specialized model
    ///
    /// Uses 'functiongemma' model to analyze the last user/assistant messages
    /// and identify potential tool calls based on available tools.
    ///
    /// # Arguments
    /// * `client` - HTTP client for API requests
    /// * `config` - Application configuration
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Errors
    /// Returns error if tool extraction request fails
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
            options: config
                .get_model("functiongemma")
                .and_then(|m| m.options.clone()),
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

    /// Adds a user message to the conversation history
    ///
    /// # Arguments
    /// * `content` - The message content from the user
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

    /// Prints the assistant response to console with formatting
    ///
    /// Displays thinking (in gray), content, and tool call indicators
    ///
    /// # Arguments
    /// * `resp` - The response to print
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
    /// Adds an assistant response to the conversation history
    ///
    /// # Arguments
    /// * `resp` - The response to add to history
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

    /// Adds a tool result message to the conversation history
    ///
    /// # Arguments
    /// * `tool_call_id` - ID of the tool call this result is for
    /// * `tool_name` - Name of the tool that was executed
    /// * `result` - The result content from tool execution
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

    /// Determines if the user should be prompted for input
    ///
    /// Returns true if:
    /// - No messages exist yet
    /// - No assistant response has been received
    /// - Last message is not a tool result
    ///
    /// Returns false if last message is a tool result (should auto-send to assistant)
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

    /// Estimates the total token count for the current session
    ///
    /// Includes tokens from all messages and tool definitions
    ///
    /// # Returns
    /// Estimated token count
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
