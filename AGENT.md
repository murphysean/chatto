# Project Context for AI Assistant

## Project Overview
`chatto` is a Rust CLI tool that interfaces with Ollama instances using shell commands. It provides contextual chat interactions with AI models through the Ollama API, supporting system prompts and tool execution.

## Architecture
- `src/main.rs` - Entry point with CLI argument parsing
- `src/ollama.rs` - Core chat functionality and Ollama API integration
- `src/tools.rs` - Contains functionality for some of the built in agentic tools like shell command, read/write files
- `src/app.rs` - The application, stores app state and config. All methods for display and prompt are done here.
- `src/chat.rs` - Implementation of the chat cli command.
- Uses `.chatto.yaml` configuration file in user's project directory
- Built with async/await pattern using tokio

## Key Dependencies
- `clap` - Command-line argument parsing
- `tokio` - Async runtime
- `serde` - Serialization/deserialization
- `reqwest` - HTTP client for Ollama API calls
- `yaml` - Configuration file parsing

## Coding Standards
- Follow standard Rust conventions and idioms
- Use `cargo fmt` for formatting
- Run `cargo clippy` for linting
- Write tests for new features using `cargo test`
- Prefer explicit error handling over unwrap()

## Configuration
- Configuration stored in `.chatto.yaml` in project (or current) directory
- Supports URL and model specification via CLI or config
- Command structure: `chatto chat --url <URL> --model <MODEL>`

## Common Development Tasks
- New CLI commands should be added as subcommands in chat.rs
- Tool implementations should go in tools.rs
- Chat functionality extensions go in main.rs
- Configuration changes require updating the YAML schema
- API interactions should handle network errors gracefully

## Project Goals
- Simple, fast CLI interface to Ollama
- Contextual conversation support
- Extensible tool integration
- Reliable error handling and user feedback
