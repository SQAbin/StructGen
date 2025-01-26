import json
import requests
from typing import Optional, Dict, Any, List, Union


class FeidaChat:
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize the Feida chat client

        Args:
            api_key (str): The API key for authentication
        """
        self.api_key = api_key
        self.base_url = base_url

    def generate(
            self,
            messages: Union[str, List[Dict[str, str]]],
            model: str = "gpt-3.5-turbo",
            temperature: float = 0.7,
            top_p: float = 1.0,
            max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate a response using the Feida API

        Args:
            messages: Either a string prompt or a list of message dictionaries
            model: The model to use for generation
            temperature: Controls randomness (0-1)
            top_p: Controls diversity of responses (0-1)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dict containing either the response text or error information
        """
        # Convert string input to proper message format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()

            return {
                "success": True,
                "text": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {})
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request error: {str(e)}"
            }
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return {
                "success": False,
                "error": f"Response parsing error: {str(e)}"
            }
