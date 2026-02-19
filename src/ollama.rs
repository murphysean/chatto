use std::{collections::HashMap, error::Error};

use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OllamaOptions {
    pub seed: Option<u64>,
    pub temperature: Option<f64>,
    pub top_k: Option<u64>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub stop: Option<String>,
    pub num_ctx: Option<u64>,
    pub num_predict: Option<u64>,
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

#[derive(Debug, Clone, Copy)]
pub enum OllamaChatResponseStreamingState {
    NoStream,
    Receiving,
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

    let mut stream = response.bytes_stream();

    let mut streaming_state: OllamaChatResponseStreamingState =
        OllamaChatResponseStreamingState::Receiving;
    let mut ollama_response: OllamaChatResponse = OllamaChatResponse::default();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Stream chunk error: {}", e))?;
        let chunk_str = String::from_utf8_lossy(&chunk);

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
