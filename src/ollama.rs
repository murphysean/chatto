use std::error::Error;

use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::app::ApplicationState;
use crate::ApplicationConfig;

#[derive(Debug, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    pub options: Option<Value>,
    pub stream: bool,
    pub think: bool,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
    /// The agent fills this when it is requesting tool use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Used for role = tool
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Used for role = tool
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl OllamaChatMessage {
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

#[derive(Debug, Deserialize, Clone)]
pub struct OllamaChatResponseMessage {
    pub role: String,
    pub content: String,
    pub thinking: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    #[serde(default)]
    pub message: Option<OllamaChatResponseMessage>,

    #[serde(default)]
    pub done: bool,
    pub done_reason: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,

    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duariont: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,

    pub images: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: Option<String>,
    pub function: ToolCallFunction,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub description: Option<String>,
    pub arguments: Value,
}

pub enum OllamaChatResponseStreamingState {
    Recieving,
    Thinking,
    Responding,
    CallingTools,
    Done,
}

pub async fn post_ollama_chat(
    client: &Client,
    config: &ApplicationConfig,
    state: &mut ApplicationState,
) -> Result<OllamaChatResponse, Box<dyn Error>> {
    let mut request: OllamaChatRequest = state.clone().into();
    request.stream = config.stream;
    request.think = false;
    if let Some(model_config) = config.models.get(state.model.as_str()) {
        request.think = model_config.think;
        if !model_config.tools {
            request.tools = None;
        }
        if model_config.num_ctx.is_some() {
            request.options = Some(json!({"num_ctx":model_config.num_ctx.unwrap()}));
        }
    }

    let response = client
        .post(format!("{}/api/chat", config.url))
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Failed to connect to Ollama at {}: {}", config.url, e))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("API Error: Status: {}: {}", status, error_text).into());
    }

    // Check content type
    if response
        .headers()
        .get("Content-Type")
        .map(|cthv| cthv.to_str().unwrap_or_default())
        == Some("application/json")
    {
        let body: OllamaChatResponse = response.json().await?;
        state.add_assistant_response(body.clone());
        state.finalize_assistant_message(&body);
        return Ok(body);
    }

    state.add_assistant_message("");

    let mut stream = response.bytes_stream();

    let mut streaming_state: OllamaChatResponseStreamingState =
        OllamaChatResponseStreamingState::Recieving;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Stream chunk error: {}", e))?;
        let chunk_str = String::from_utf8_lossy(&chunk);

        for line in chunk_str.lines() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<OllamaChatResponse>(line) {
                Ok(ollama_response) => {
                    if ollama_response.done {
                        state.finalize_assistant_message(&ollama_response);
                        return Ok(ollama_response);
                    }
                    //update app state with incoming message
                    streaming_state =
                        state.process_assistant_message_chunk(streaming_state, ollama_response);
                }
                Err(e) => {
                    println!("{line}");
                    return Err(format!("JSON Error: {}", e).into());
                }
            }
        }
    }

    Err(format!("No final response, {}", 0).into())
}
