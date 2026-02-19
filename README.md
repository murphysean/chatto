# Chatto

A powerful CLI tool for contextual chat interactions with Ollama AI models.

## Overview

Chatto provides a command-line interface for interacting with Ollama instances, supporting contextual conversations, system prompts, tool execution, and session management. It's designed for developers who want to work with AI models in a terminal environment with full control over context and tool usage.

## Features

- **Contextual Chat**: Maintains conversation history for multi-turn interactions
- **Tool Support**: Built-in tools for shell command execution, file reading, and file writing
- **Session Management**: Save and load conversation sessions to resume work later
- **Streaming Responses**: Real-time streaming of AI responses with progress indicators
- **Model Management**: List available models and query model capabilities
- **Context Compaction**: Automatically or manually summarize conversation history to stay within context limits
- **Thinking Support**: Display AI reasoning/thinking process when supported by the model
- **Editor Integration**: Open your preferred editor for composing complex messages

## Installation

### From Source

Ensure you have Rust installed on your machine, then:

```bash
git clone <repository-url>
cd chatto
cargo install --path .
```

### Using Cargo

```bash
cargo install chatto
```

## Quick Start

```bash
# List available models
chatto list

# Start a chat session
chatto chat

# Start with a specific model
chatto chat --model llama2

# Connect to a remote Ollama instance
chatto chat --url http://192.168.1.100:11434 --model llama2

# Start a named session (for persistence)
chatto chat --session my-project
```

## Commands

### `chatto list`

List all available models on the Ollama instance.

```bash
chatto list --url http://localhost:11434
```

Displays:
- Model names
- Context length for each model
- Model capabilities (e.g., "tools" for function calling support)

### `chatto chat`

Start an interactive chat session with an Ollama model.

```bash
chatto chat [OPTIONS]
```

**Options:**
- `--url <URL>`: Ollama instance URL (default: `https://localhost:11434`)
- `--model <MODEL>`: Model to use (default: `llama2`)
- `--session <NAME>`: Session name for persistence
- `--disable-streaming`: Receive complete responses instead of streaming

**Chat Commands:**
- `/quit`, `/exit`, `/done` - Exit the chat session (saves if session named)
- `/save <name>` - Save the current session with a name
- `/edit`, `/editor` - Open external editor for message composition
- `/tools` - Extract tool calls from last assistant message
- `/reset` - Clear message history (keeps system message)
- `/trim` - Trim history to essential messages (system + last user + last assistant)
- `/compact` - Summarize conversation history to reduce context
- `/send` - Force send to model without user message (useful after tool results)

## Configuration

Configuration is loaded in the following order (later sources override earlier ones):

1. **Default values**
   - URL: `https://localhost:11434`
   - Model: `llama2`
   - Streaming: `true`

2. **`~/.config/chatto/Config.yaml`** - User-level configuration

3. **`.chatto.yaml`** - Project-level configuration

4. **Environment variables**
   - `OLLAMA_API_KEY` - API key for authentication

5. **Command-line arguments** - Highest priority

### Configuration File Example

```yaml
url: http://localhost:11434
api_key: your-api-key-here
model: llama2
stream: true
output_limit:
  max_size: 0  # 0 = no limit
  method: head

models:
  - name: llama2
    options:
      temperature: 0.7
      num_ctx: 4096
```

## Session Management

Sessions are saved as YAML files (`.chatto-<name>.session.yaml`) in the current directory. They include:

- Conversation history
- Model configuration
- Tool definitions

```bash
# Start a new session
chatto chat --session my-project

# Resume the session later
chatto chat --session my-project

# Save current session manually during chat
/save my-project
```

## Tool Integration

When using models with tool-calling capabilities, Chatto provides three built-in tools:

### 1. `execute_shell`
Execute shell commands with output limiting support.
- The AI can run compilation, git operations, directory listings, etc.
- User approval required before execution
- Output can be limited to prevent context overflow

### 2. `read_file`
Read file contents with optional line ranges.
- More efficient than shell commands for file reading
- Supports reading specific line ranges
- Returns content with line numbers for easy reference

### 3. `write_file`
Write to files with multiple modes:
- **overwrite**: Replace entire file
- **append**: Add to end of file
- **insert**: Insert at specific line
- **replace**: Replace line range

All tool executions require user approval, with a diff preview for file writes.

## Project Context

Chatto automatically looks for an `AGENT.md` file in the current directory and includes its contents as project context for the AI. This allows you to provide project-specific instructions and documentation.

Example `AGENT.md`:
```markdown
# Project Context

This is a Rust web API using Actix-web. The project structure:
- src/handlers/ - HTTP request handlers
- src/models/ - Data models and DTOs
- src/db/ - Database operations

When modifying code:
1. Follow existing patterns
2. Add proper error handling
3. Update tests in tests/
```

## Development

### Building

```bash
cargo build --release
```

### Running Tests

```bash
cargo test
```

### Documentation

```bash
cargo doc --open
```

### Code Quality

```bash
cargo fmt
cargo clippy
```

## Architecture

- **`src/main.rs`** - CLI argument parsing and command routing
- **`src/ollama.rs`** - Ollama API client and data structures
- **`src/chat.rs`** - Chat mode REPL implementation
- **`src/app.rs`** - Application state and message management
- **`src/tools.rs`** - Built-in tool implementations

## API Compatibility

Chatto is designed to work with:
- Ollama (local or remote instances)
- OpenAI-compatible APIs
- Any service implementing the Ollama API specification

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Guidelines
- Follow Rust conventions (`cargo fmt`, `cargo clippy`)
- Add tests for new features
- Update documentation for API changes
- Keep PRs focused and well-described

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Built with:
- [clap](https://github.com/clap-rs/clap) - CLI argument parsing
- [reqwest](https://github.com/seanmonstar/reqwest) - HTTP client
- [tokio](https://github.com/tokio-rs/tokio) - Async runtime
- [serde](https://github.com/serde-rs/serde) - Serialization
- [rustyline](https://github.com/kkawakam/rustyline) - Readline implementation