use std::error::Error;

use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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

#[derive(Debug, Default, Deserialize, Clone)]
pub struct OllamaChatResponseMessage {
    pub role: String,
    pub content: String,
    pub thinking: Option<String>,
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
    fn merge(&mut self, incoming: &OllamaChatResponse) -> OllamaChatResponseStreamingState {
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
                return OllamaChatResponseStreamingState::Thinking;
            }
            if !message.content.is_empty() {
                self.message
                    .as_mut()
                    .unwrap()
                    .content
                    .push_str(&message.content);
                return OllamaChatResponseStreamingState::Responding;
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
                return OllamaChatResponseStreamingState::CallingTools;
            }
        }
        self.done = incoming.done;
        self.done_reason = incoming.done_reason.clone();

        self.total_duration = incoming.total_duration;
        self.load_duration = incoming.load_duration;
        self.prompt_eval_count = incoming.prompt_eval_count;
        self.prompt_eval_duration = incoming.prompt_eval_duration;
        self.eval_count = incoming.eval_count;
        self.eval_duration = incoming.eval_duration;

        OllamaChatResponseStreamingState::Done
    }
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

#[derive(Copy, Clone)]
pub enum OllamaChatResponseStreamingState {
    Recieving,
    Thinking,
    Responding,
    CallingTools,
    Done,
}

pub trait StreamingChatHandler {
    fn process_streaming_response(
        &mut self,
        previous_streaming_state: &OllamaChatResponseStreamingState,
        current_streaming_state: &OllamaChatResponseStreamingState,
        ollama_response: &OllamaChatResponse,
    );
}

pub async fn post_ollama_chat(
    client: &Client,
    url: &str,
    request: &OllamaChatRequest,
    mut streaming_chat_handler: Option<impl StreamingChatHandler>,
) -> Result<OllamaChatResponse, Box<dyn Error>> {
    let response = client
        .post(format!("{}/api/chat", url))
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
    if response
        .headers()
        .get("Content-Type")
        .map(|cthv| cthv.to_str().unwrap_or_default())
        == Some("application/json")
    {
        let body: OllamaChatResponse = response.json().await?;
        return Ok(body);
    }

    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Stream chunk error: {}", e))?;
        let chunk_str = String::from_utf8_lossy(&chunk);
        let mut streaming_state: OllamaChatResponseStreamingState =
            OllamaChatResponseStreamingState::Recieving;
        let mut ollama_response: OllamaChatResponse = OllamaChatResponse::default();

        for line in chunk_str.lines() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<OllamaChatResponse>(line) {
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
                    if let OllamaChatResponseStreamingState::Done = streaming_state {
                        return Ok(ollama_response);
                    }
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
