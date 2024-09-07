# Document Q&A Application

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Document Q&A Application is a Streamlit-based web application that allows users to upload documents, process them, and ask questions about their content. This application leverages advanced natural language processing techniques to understand and respond to user queries based on the uploaded document.

## Features

- Support for multiple document types (PDF, DOCX, TXT, MD)
- Document summarization
- Interactive Q&A based on document content
- Utilizes state-of-the-art language models for accurate responses
- Vector store for efficient document chunk retrieval
- Customizable LLM configuration

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rampal-punia/rag-app.git
   cd rag-app
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory and add the following:
   ```
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
   REDIS_URL=your_redis_url  # Optional, defaults to localhost
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the application:
   - Upload a document (PDF, DOCX, TXT, or MD)
   - View the document summary
   - Ask questions about the document content

## Project Structure

```
rag-app/
├── app.py                 # Main Streamlit application
├── configure_llm.py       # LLM configuration and chain building
├── loader.py              # Document processing and vector store management
├── settings.py            # Application settings and configurations
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (create this file)
├── logs/                  # Log files directory
└── README.md              # This file
```

## Configuration

The application can be configured by modifying the `settings.py` file. Key configurations include:

- `CHUNK_SIZE`: Size of document chunks for processing (default: 350)
- `CHUNK_OVERLAP`: Overlap between document chunks (default: 80)
- `REDIS_URL`: URL for Redis connection (optional)

LLM configurations can be adjusted in the `configure_llm.py` file, including model selection and parameters.

## Dependencies

Main dependencies include:

- streamlit
- langchain
- huggingface_hub
- chromadb
- pypdf
- docx2txt
- python-decouple

For a complete list of dependencies, refer to the `requirements.txt` file.

## Contributing

Contributions to the Document Q&A Application are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is intended as a learning exercise and demonstration of integrating these technologies:

- Sreamlit
- LangChain
- chromadb
- Hugging Face Endpoint APIs

Please note that this application is not designed or tested for production use. It serves as an educational resource and a showcase of technology integration rather than a production-ready web application.

Contributors and users are welcome to explore, learn from, and build upon this project for educational purposes.

---

For any questions or issues, please open an issue on the GitHub repository.