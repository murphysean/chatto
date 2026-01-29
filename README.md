# README

## Overview

This project provides a CLI tool named `chatto` designed to interface with an Ollama instance using shell commands. The tool allows users to execute various operations through the Ollama API while supporting contextual interactions, system prompts, and optional tools execution.

## Installation

To install `chatto`, ensure you have Rust installed on your machine. Then run:

```sh
cargo install chatto
```

## Usage

The application can be invoked with a command to start chatting or interacting with the Ollama model using shell commands. The main usage is through the `chat` subcommand.

#### Commands
- **chat**: Start a conversation session with the specified model and URL.
  ```sh
catto chat --url <URL> --model <MODEL>
```

## Configuration File

The application uses a configuration file named `.chatto.yaml` for settings. Ensure this file is correctly set up in your home directory or specify it via command-line arguments.

## Development

To contribute to the project, follow these steps:

1. **Clone the Repository**:
   ```sh
git clone <repository-url>
```

2. **Install Dependencies**: Follow any additional setup instructions provided in the repository for development dependencies.

3. **Run Tests**: Ensure all tests pass before submitting your changes.
  ```sh
  cargo test
  ```

4. **Build and Run**:
   Compile the project using:
   ```sh
   cargo build --release
   
   Then run it with:
   ```sh
target/release/chatto <command>
```

5. **Documentation**: Update any relevant documentation files or README sections as needed.

## Contributing

Contributions are welcome! Please follow our contribution guidelines, which include:
- **Code Quality**: Ensure your code follows the project's coding standards and conventions.
- **Testing**: Add comprehensive tests for new features.
- **Documentation**: Document all changes in the README or relevant files.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

