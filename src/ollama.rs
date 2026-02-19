//! Ollama API client implementation.
//!
//! This module provides the core functionality for interacting with Ollama's
//! chat API, including request/response structures, streaming support, and
//! model information retrieval.
//!
//! ## Key Features
//!
//! - **Chat API**: Send messages and receive responses with streaming support
//! - **Model Management**: List available models and query model capabilities
//! - **Tool Calling**: Support for function/tool calling in chat sessions
//! - **Streaming**: Real-time response streaming with state tracking

use std::{collections::HashMap, error::Error};

use futures::AsyncBufReadExt;
use futures::TryStreamExt;
use futures_util::io::BufReader;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Request structure for the Ollama chat API.
///
/// This struct represents the data sent to Ollama when making a chat request,
/// including the model to use, messages, tools, options, and streaming preferences.
#[derive(Debug, Deserialize, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    pub options: Option<OllamaOptions>,
    pub stream: bool,
    pub think: bool,
}

/// Model generation options for Ollama requests.
///
/// Controls various aspects of text generation including randomness,
/// token limits, and stopping conditions.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OllamaOptions {
    /// Random seed for reproducible outputs
    pub seed: Option<u64>,
    /// Temperature for sampling (higher = more random)
    pub temperature: Option<f64>,
    /// Top-k sampling parameter
    pub top_k: Option<u64>,
    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f64>,
    /// Minimum probability threshold for sampling
    pub min_p: Option<f64>,
    /// Stop sequence that halts generation
    pub stop: Option<String>,
    /// Maximum context window size in tokens
    pub num_ctx: Option<u64>,
    /// Maximum number of tokens to predict
    pub num_predict: Option<u64>,
}

/// A single message in an Ollama chat conversation.
///
/// Represents messages from different roles (user, assistant, system, tool)
/// with support for tool calling functionality.
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct OllamaChatMessage {
    /// The role of this message sender (user, assistant, system, or tool)
    pub role: String,
    /// The text content of the message
    pub content: String,
    /// Tool calls requested by the assistant (when role is assistant)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Name of the tool that was executed (when role is tool)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// ID of the tool call this responds to (when role is tool)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl OllamaChatMessage {
    /// Estimates the token count for this message based on its content and tool calls.
    ///
    /// Uses a heuristic approach combining byte and word counting to estimate tokens.
    /// For tool calls, estimates based on JSON serialization length.
    ///
    /// Returns:
    ///     Estimated token count as usize
    pub fn get_token_count_estimate(&self) -> usize {
        let bytes = self.content.len() as f64;
        let words = self.content.split_whitespace().count() as f64;

        let byte_estimate = bytes / 5.0; // ~5 bytes per token
        let word_estimate = words * 1.3; // ~1.3 tokens per word

        let content_tokens = ((byte_estimate + word_estimate) / 2.0) as usize;

        // Add tokens for tool calls if present
        let tool_call_tokens = self
            .tool_calls
            .as_ref()
            .map(|calls| {
                calls
                    .iter()
                    .map(|call| {
                        let json_str = serde_json::to_string(call).unwrap_or_default();
                        (json_str.len() as f64 / 4.0) as usize
                    })
                    .sum::<usize>()
            })
            .unwrap_or(0);

        content_tokens + tool_call_tokens
    }
}

/// The message portion of an Ollama chat response.
///
/// Contains the assistant's reply including any thinking process
/// and tool calls.
#[derive(Debug, Default, Deserialize, Clone)]
pub struct OllamaChatResponseMessage {
    /// Always "assistant" for responses
    pub role: String,
    /// The main response content
    pub content: String,
    /// Internal reasoning/thinking (for models that support it)
    pub thinking: Option<String>,
    /// Tool calls requested by the assistant
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    #[serde(default)]
    pub message: Option<OllamaChatResponseMessage>,

    #[serde(default)]
    pub done: bool,
    pub done_reason: Option<String>,

    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,

    pub images: Option<Vec<String>>,
}

impl OllamaChatResponse {
    /// Merges an incoming streaming response chunk into this response.
    ///
    /// Accumulates content, thinking, and tool calls from streaming chunks
    /// into a single coherent response. Tracks the streaming state as chunks
    /// are merged.
    ///
    /// # Arguments
    /// * `incoming` - The new response chunk to merge into this response
    ///
    /// # Returns
    /// The current streaming state after merging (e.g., Thinking, Responding, CallingTools, Done)
    fn merge(&mut self, incoming: &OllamaChatResponse) -> OllamaChatResponseStreamingState {
        let mut ret: Option<OllamaChatResponseStreamingState> = None;
        self.model = incoming.model.clone();
        self.created_at = incoming.created_at.clone();
        if let Some(message) = &incoming.message {
            if self.message.is_none() {
                self.message = Some(OllamaChatResponseMessage::default());
            }

            if let Some(thinking) = &message.thinking {
                if self.message.as_ref().unwrap().thinking.is_none() {
                    self.message.as_mut().unwrap().thinking = Some(String::new());
                }
                self.message
                    .as_mut()
                    .unwrap()
                    .thinking
                    .as_mut()
                    .unwrap()
                    .push_str(thinking);
                ret = Some(OllamaChatResponseStreamingState::Thinking);
            }
            if !message.content.is_empty() {
                self.message
                    .as_mut()
                    .unwrap()
                    .content
                    .push_str(&message.content);
                ret = Some(OllamaChatResponseStreamingState::Responding);
            }
            if let Some(tool_calls) = &message.tool_calls {
                if self.message.as_ref().unwrap().tool_calls.is_none() {
                    self.message.as_mut().unwrap().tool_calls = Some(Vec::new());
                }
                self.message
                    .as_mut()
                    .unwrap()
                    .tool_calls
                    .as_mut()
                    .unwrap()
                    .extend(tool_calls.iter().cloned());
                ret = Some(OllamaChatResponseStreamingState::CallingTools);
            }
        }
        self.done = incoming.done;
        if self.done {
            ret = Some(OllamaChatResponseStreamingState::Done)
        }
        self.done_reason = incoming.done_reason.clone();

        self.total_duration = incoming.total_duration;
        self.load_duration = incoming.load_duration;
        self.prompt_eval_count = incoming.prompt_eval_count;
        self.prompt_eval_duration = incoming.prompt_eval_duration;
        self.eval_count = incoming.eval_count;
        self.eval_duration = incoming.eval_duration;

        if let Some(state) = ret {
            state
        } else {
            OllamaChatResponseStreamingState::Receiving
        }
    }
}

/// A tool call request from the assistant.
///
/// Represents a single function call that the assistant wants to execute,
/// including an ID for tracking and the function details.
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call (used in tool result messages)
    #[serde(default)]
    pub id: Option<String>,
    /// The function to call with its arguments
    pub function: ToolCallFunction,
}

/// Details of a tool function call.
///
/// Specifies which tool to call and with what arguments.
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct ToolCallFunction {
    /// Name of the tool function to call
    pub name: String,
    /// Optional description of why this tool is being called
    pub description: Option<String>,
    /// JSON object containing the function arguments
    pub arguments: Value,
}

/// States during streaming response processing.
///
/// Tracks the current phase of a streaming chat response, used to
/// provide user feedback and control message formatting.
#[derive(Debug, Clone, Copy)]
pub enum OllamaChatResponseStreamingState {
    /// Non-streaming response (single response)
    NoStream,
    /// Currently receiving data but no content yet
    Receiving,
    /// Model is outputting thinking/reasoning
    Thinking,
    /// Model is outputting response content
    Responding,
    /// Model is requesting tool execution
    CallingTools,
    /// Stream has completed
    Done,
}

/// Trait for handling streaming chat response chunks.
///
/// Implement this trait to process streaming responses in real-time,
/// such as displaying content to the user as it arrives.
///
/// # Example
/// ```ignore
/// struct MyHandler;
/// impl StreamingChatHandler for MyHandler {
///     fn process_streaming_response(&mut self, prev: &OllamaChatResponseStreamingState, curr: &OllamaChatResponseStreamingState, resp: &OllamaChatResponse) {
///         if let Some(msg) = &resp.message {
///             print!("{}", msg.content);
///         }
///     }
/// }
/// ```
pub trait StreamingChatHandler {
    /// Process a streaming response chunk.
    ///
    /// Called for each chunk received during streaming, providing both
    /// the previous and current streaming states for transition handling.
    ///
    /// # Arguments
    /// * `previous_streaming_state` - State before this chunk
    /// * `current_streaming_state` - State after processing this chunk
    /// * `ollama_response` - The response chunk to process
    fn process_streaming_response(
        &mut self,
        previous_streaming_state: &OllamaChatResponseStreamingState,
        current_streaming_state: &OllamaChatResponseStreamingState,
        ollama_response: &OllamaChatResponse,
    );
}

/// Sends a chat request to the Ollama API and handles the response.
///
/// Supports both streaming and non-streaming responses. For streaming responses,
/// optionally calls a handler for each chunk received.
///
/// # Arguments
/// * `client` - The HTTP client to use for the request
/// * `url` - The base URL of the Ollama API
/// * `key` - The API key for authentication
/// * `request` - The chat request to send
/// * `streaming_chat_handler` - Optional handler for processing streaming response chunks
///
/// # Returns
/// A tuple containing the complete response and the final streaming state,
/// or an error if the request fails.
///
/// # Errors
/// Returns an error if:
/// - Connection to Ollama fails
/// - The API returns a non-success status code
/// - JSON parsing fails
/// - The stream ends without a final response
pub async fn post_ollama_chat(
    client: &Client,
    url: &str,
    key: &str,
    request: &OllamaChatRequest,
    mut streaming_chat_handler: Option<impl StreamingChatHandler>,
) -> Result<(OllamaChatResponse, OllamaChatResponseStreamingState), Box<dyn Error>> {
    let response = client
        .post(format!("{}/api/chat", url))
        .header("Authorization", format!("Bearer {}", key))
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Failed to connect to Ollama at {}: {}", url, e))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("API Error: Status: {}: {}", status, error_text).into());
    }

    // Check content type
    if !request.stream
        && response
            .headers()
            .get("Content-Type")
            .map(|cthv| cthv.to_str().unwrap_or_default())
            == Some("application/json")
    {
        let body: OllamaChatResponse = response.json().await?;
        return Ok((body, OllamaChatResponseStreamingState::NoStream));
    }

    let reader = BufReader::new(
        response
            .bytes_stream()
            .map_err(futures::io::Error::other)
            .into_async_read(),
    );

    let mut streaming_state: OllamaChatResponseStreamingState =
        OllamaChatResponseStreamingState::Receiving;
    let mut ollama_response: OllamaChatResponse = OllamaChatResponse::default();

    let mut lines = reader.lines();
    while let Some(line) = lines.next().await {
        let line = line.map_err(|e| format!("Stream line error: {}", e))?;
        match serde_json::from_str::<OllamaChatResponse>(&line) {
            Ok(response_chunk) => {
                let prev_streaming_state = streaming_state;
                streaming_state = ollama_response.merge(&response_chunk);
                if let Some(streaming_chat_handler) = streaming_chat_handler.as_mut() {
                    streaming_chat_handler.process_streaming_response(
                        &prev_streaming_state,
                        &streaming_state,
                        &response_chunk,
                    );
                }
                if let OllamaChatResponseStreamingState::Done = &streaming_state {
                    return Ok((ollama_response, streaming_state));
                }
            }
            Err(e) => {
                println!("{line}");
                return Err(format!("JSON Error: {}", e).into());
            }
        }
    }

    Err(format!("No final response, {}", 0).into())
}

#[derive(Debug, Deserialize, Clone)]
pub struct OllamaModel {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub modified_at: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    pub details: Option<OllamaModelDetails>,
    #[serde(default)]
    pub model_info: HashMap<String, Value>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    pub options: Option<OllamaOptions>,
}

impl OllamaModel {
    /// Retrieves the context length for this model from its metadata.
    ///
    /// Searches the model info for a key ending with "context_length"
    /// and returns its numeric value if found.
    ///
    /// # Returns
    /// The context length as u64 if found, or None if not available
    pub fn get_context_length(&self) -> Option<u64> {
        for (k, v) in self.model_info.iter() {
            if k.ends_with("context_length") && v.is_number() {
                return v.as_u64();
            }
        }
        None
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct OllamaModelDetails {
    pub parent_model: String,
    pub format: String,
    pub family: String,
    pub parameter_size: String,
    //pub parameter_size: u64,
    pub quantization_level: String,
}

/// Lists all available models from the Ollama instance.
///
/// Retrieves a list of models available on the Ollama server, including
/// their metadata such as size, modification time, and capabilities.
///
/// # Arguments
/// * `client` - The HTTP client to use for the request
/// * `url` - The base URL of the Ollama API
/// * `key` - The API key for authentication
///
/// # Returns
/// A vector of OllamaModel objects representing available models,
/// or an error if the request fails.
///
/// # Errors
/// Returns an error if the connection fails or the response cannot be parsed.
pub async fn list_models(
    client: &Client,
    url: &str,
    key: &str,
) -> Result<Vec<OllamaModel>, Box<dyn Error>> {
    let response = client
        .get(format!("{}/api/tags", url))
        .header("Authorization", format!("Bearer {}", key))
        .send()
        .await
        .map_err(|e| format!("Failed to connect to Ollama at {}: {}", url, e))?;
    #[derive(Debug, Deserialize, Clone)]
    pub struct AnonWrapper {
        pub models: Vec<OllamaModel>,
    }
    let body: AnonWrapper = response.json().await?;
    Ok(body.models)
}

/// Retrieves detailed information about a specific model.
///
/// Fetches model metadata including configuration, parameters, and capabilities
/// for the specified model name.
///
/// # Arguments
/// * `client` - The HTTP client to use for the request
/// * `url` - The base URL of the Ollama API
/// * `key` - The API key for authentication
/// * `model` - The name of the model to retrieve information for
///
/// # Returns
/// An OllamaModel with detailed information, or an error if the request fails.
///
/// # Errors
/// Returns an error if:
/// - Connection to Ollama fails
/// - The model does not exist
/// - The API returns a non-success status code
pub async fn show_model(
    client: &Client,
    url: &str,
    key: &str,
    model: &str,
) -> Result<OllamaModel, Box<dyn Error>> {
    let response = client
        .post(format!("{}/api/show", url))
        .header("Authorization", format!("Bearer {}", key))
        .json(&json!({"model": model}))
        .send()
        .await
        .map_err(|e| format!("Failed to connect to Ollama at {}: {}", url, e))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("API Error: Status: {}: {}", status, error_text).into());
    }
    let mut body: OllamaModel = response.json().await?;
    body.name = model.to_string();
    Ok(body)
}
