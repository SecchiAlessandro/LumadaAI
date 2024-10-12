# Lumada AI

## Overview

**Lumada AI** is an intelligent multi-agent framework designed to scan Hitachi Group's products and solutions websites. It utilizes a supervisor agent named **LumadaAI**, which dynamically selects the appropriate agent based on the user's query. This framework allows for efficient and effective retrieval of information from a wide range of resources within the Hitachi Group.

## Features

- **Multi-Agent Framework**: Incorporates several agents, each specialized in different areas of Hitachi's offerings.
- **Dynamic Query Handling**: The supervisor agent (LumadaAI) analyzes user queries and routes them to the appropriate agent.
- **Pre-Scraped Data**: The relevant information from the Hitachi Energy websites has already been scraped and is stored in a JSON file for quick access.

## Getting Started


### Installation

1. Clone the repository:
   ```bash
   git clone [repository_url]
   cd lumadaai
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Chatbot

To run the **Lumada AI** chatbot, execute the following command:

streamlit run chatbot.py

Ensure that the pre-scraped data from the Hitachi Energy websites is available in the JSON file before running the chatbot.

## Usage

After starting the chatbot, you can ask questions related to Hitachi Group products and solutions. The supervisor agent will determine the best response based on the question you provide.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.

## Contact

For questions or feedback, please reach out.
