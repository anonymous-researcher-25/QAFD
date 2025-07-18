import base64
import json
import logging
import os
import re
import time
from http import HTTPStatus
from io import BytesIO

from openai import AzureOpenAI
from typing import Dict, List, Optional, Tuple, Any, TypedDict
import dashscope
from groq import Groq
import google.generativeai as genai
import openai
import requests
import tiktoken
import signal
import anthropic

logger = logging.getLogger("api-llms")


def call_llm(payload):
    model = payload["model"]
    stop = ["Observation:","\n\n\n\n","\n \n \n"]
    
    # Handle Claude models - Anthropic API
    if model.startswith("claude"):
        logger.info("Generating content with Claude model: %s", model)
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]
        
        anthropic_messages = []
        system_prompt = None
        
        for message in messages:
            # Extract system prompt if present
            if message["role"] == "system":
                for part in message["content"]:
                    if part['type'] == "text":
                        system_prompt = part['text']
                continue
                
            anthropic_message = {
                "role": message["role"],
                "content": []
            }
            
            for part in message["content"]:
                if part['type'] == "text":
                    anthropic_message['content'].append({"type": "text", "text": part['text']})
                elif part['type'] == "image_url":
                    # Handle base64 encoded images
                    if part['image_url']['url'].startswith("data:"):
                        image_type = part['image_url']['url'].split(';')[0].split(':')[1]
                        image_data = part['image_url']['url'].split(',')[1]
                        anthropic_message['content'].append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_type,
                                "data": image_data
                            }
                        })
                    else:
                        # Handle URL images
                        anthropic_message['content'].append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": part['image_url']['url']
                            }
                        })
            
            anthropic_messages.append(anthropic_message)
        
        # Initialize the Anthropic client
        try:
            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            return False, f"Failed to initialize Anthropic client: {e}"
        
        for i in range(3):
            try:
                logger.info(f"Calling Claude model: {model}, attempt {i+1}")
                
                # Get the specific model name from the full model string
                model_name = model.split("/")[-1] if "/" in model else model
                
                # Create a request to Anthropic
                # Filter out empty stop sequences as Anthropic API doesn't accept them
                valid_stop_sequences = [seq for seq in stop if seq and not seq.isspace()] if stop else None
                
                # Create the API request
                response = client.messages.create(
                    model=model_name,
                    messages=anthropic_messages,
                    system=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=valid_stop_sequences if valid_stop_sequences else None
                )
                
                # Extract the response content
                output_message = response.content[0].text
                logger.info(f"Successfully generated content with Claude model: {model}")
                return True, output_message
                
            except Exception as e:
                logger.error(f"Failed to call Claude LLM (attempt {i+1}): {str(e)}")
                
                if hasattr(e, 'status_code') and e.status_code == 400:
                    if "content_policy_violation" in str(e):
                        logger.warning("Content policy violation detected")
                        return False, "content_filter"
                    elif "context_length_exceeded" in str(e) or "too_many_tokens" in str(e):
                        logger.warning("Context length exceeded")
                        return False, "context_length_exceeded"
                
                logger.error(f"Retrying Claude API call in {4 * (2 ** (i + 1))} seconds...")
                time.sleep(4 * (2 ** (i + 1)))
                
        return False, "claude_api_error"
    
    # Handle GPT models - OpenAI API
    elif model.startswith("gpt"):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        logger.info("Generating content with GPT model: %s", model)
        

        for i in range(3):
            try:
                response = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=payload
                        )
                output_message = response.json()['choices'][0]['message']['content']
                # logger.info(f"Input: \n{payload['messages']}\nOutput:{response}")
                return True, output_message
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                if hasattr(e, 'response') and e.response is not None:
                    error_info = e.response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value        
                else:
                    code_value = 'unknown_error'
                logger.error("Retrying ...")
                time.sleep(4 * (2 ** (i + 1)))
        return False, code_value
    
    # Handle Local Llama model
    elif model.startswith("llama"):
        logger.info("Generating content with local Llama model: %s", model)
        api_url = "http://105.144.47.80:8001/v1/chat/completions"
        api_key = os.environ.get("LLAMA_API_KEY", "EMPTY")
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add authorization header only if API key is not EMPTY
        if api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Convert messages format if needed
        messages = payload["messages"]
        llama_messages = []
        
        for message in messages:
            llama_message = {
                "role": message["role"],
                "content": ""
            }
            
            # Extract text content
            for part in message["content"]:
                if part['type'] == "text":
                    llama_message['content'] = part['text']
            
            llama_messages.append(llama_message)
        
        # Prepare the request payload
        llama_payload = {
            "model": model,
            "messages": llama_messages,
            "max_tokens": payload.get("max_tokens", 2500),
            "temperature": payload.get("temperature", 0.5),
            "top_p": payload.get("top_p", 0.9),
            "stop": stop
        }
        
        for i in range(3):
            try:
                logger.info(f"Calling Llama model: {model}, attempt {i+1}")
                
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=llama_payload
                )
                
                if response.status_code == 200:
                    output_message = response.json()['choices'][0]['message']['content']
                    logger.info(f"Successfully generated content with Llama model: {model}")
                    return True, output_message
                else:
                    logger.error(f"Failed to call Llama with status code: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    
            except Exception as e:
                logger.error(f"Failed to call Llama LLM (attempt {i+1}): {str(e)}")
                
            logger.error(f"Retrying Llama API call in {4 * (2 ** (i + 1))} seconds...")
            time.sleep(4 * (2 ** (i + 1)))
        
        return False, "llama_api_error"
    
    elif model.startswith("o1"):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        logger.info("Generating content with GPT model: %s", model)
        
        messages = payload["messages"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]
            
            
        o1_messages = []

        for i, message in enumerate(messages):
            o1_message = {
                "role": message["role"] if message["role"] != "system" else "user",
                "content": ""
            }
            for part in message["content"]:
                o1_message['content'] = part['text'] if part['type'] == "text" else ""
                
                o1_messages.append(o1_message)

        payload["messages"] = o1_messages
        payload["max_completion_tokens"] = 10000
        del payload['max_tokens']
        del payload["temperature"]
        del payload["top_p"]

        for i in range(3):
            try:
                response = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=payload
                        )
                output_message = response.json()['choices'][0]['message']['content']
                # logger.info(f"Input: \n{payload['messages']}\nOutput:{response}")
                return True, output_message
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                logger.error("Retrying ...")
                time.sleep(10 * (2 ** (i + 1)))
        return False, code_value

    elif model.startswith("azure"):
        client = AzureOpenAI(
            api_key = os.environ['AZURE_API_KEY'],  
            api_version = "2024-02-15-preview",
            azure_endpoint = os.environ['AZURE_ENDPOINT']
            )
        model_name = model.split("/")[-1]
        for i in range(3):
            try:
                response = client.chat.completions.create(model=model_name,messages=payload['messages'], max_tokens=payload['max_tokens'], top_p=payload['top_p'], temperature=payload['temperature'], stop=stop)
                response = response.choices[0].message.content
                # logger.info(f"Input: \n{payload['messages']}\nOutput:{response}")
                return True, response
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                error_info = e.response.json()  
                code_value = error_info['error']['code']
                if code_value == "content_filter":
                    if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                        payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                if code_value == "context_length_exceeded":
                    return False, code_value        
                logger.error("Retrying ...")
                time.sleep(10 * (2 ** (i + 1)))
        return False, code_value
    
    elif model.startswith("anthropic"):
        logger.info("Generating content with Anthropic model: %s", model)
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]
        
        anthropic_messages = []
        system_prompt = None
        
        for message in messages:
            # Extract system prompt if present
            if message["role"] == "system":
                for part in message["content"]:
                    if part['type'] == "text":
                        system_prompt = part['text']
                continue
                
            anthropic_message = {
                "role": message["role"],
                "content": []
            }
            
            for part in message["content"]:
                if part['type'] == "text":
                    anthropic_message['content'].append({"type": "text", "text": part['text']})
                elif part['type'] == "image_url":
                    # Handle base64 encoded images
                    if part['image_url']['url'].startswith("data:"):
                        image_type = part['image_url']['url'].split(';')[0].split(':')[1]
                        image_data = part['image_url']['url'].split(',')[1]
                        anthropic_message['content'].append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_type,
                                "data": image_data
                            }
                        })
                    else:
                        # Handle URL images
                        anthropic_message['content'].append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": part['image_url']['url']
                            }
                        })
            
            anthropic_messages.append(anthropic_message)
        
        # Initialize the Anthropic client
        try:
            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            return False, f"Failed to initialize Anthropic client: {e}"
        
        for i in range(3):
            try:
                logger.info(f"Calling Anthropic model: {model}, attempt {i+1}")
                
                # Get the specific model name from the full model string
                model_name = model.split("/")[-1] if "/" in model else model
                
                # Create a request to Anthropic
                response = client.messages.create(
                    model=model_name,
                    messages=anthropic_messages,
                    system=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop if stop else None
                )
                
                # Extract the response content
                output_message = response.content[0].text
                logger.info(f"Successfully generated content with Anthropic model: {model}")
                return True, output_message
                
            except Exception as e:
                logger.error(f"Failed to call Anthropic LLM (attempt {i+1}): {str(e)}")
                
                if hasattr(e, 'status_code') and e.status_code == 400:
                    if "content_policy_violation" in str(e):
                        logger.warning("Content policy violation detected")
                        return False, "content_filter"
                    elif "context_length_exceeded" in str(e) or "too_many_tokens" in str(e):
                        logger.warning("Context length exceeded")
                        return False, "context_length_exceeded"
                
                logger.error(f"Retrying Anthropic API call in {4 * (2 ** (i + 1))} seconds...")
                time.sleep(4 * (2 ** (i + 1)))
                
        return False, "anthropic_api_error"

    elif model.startswith("mixtral"):
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        mistral_messages = []

        for i, message in enumerate(messages):
            mistral_message = {
                "role": message["role"],
                "content": ""
            }

            for part in message["content"]:
                mistral_message['content'] = part['text'] if part['type'] == "text" else ""

            mistral_messages.append(mistral_message)

        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        for i in range(2):
            try:
                logger.info("Generating content with model: %s", model)
                response = client.chat.completions.create(
                    messages=mistral_messages,
                    model=model,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    stop = stop
                )
                return True, response.choices[0].message.content
                
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(10 * (2 ** (i + 1)))
                if hasattr(e, 'response'):
                    error_info = e.response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value        
                else:
                    code_value = ""
                logger.error("Retrying ...") 

        return False, code_value
        
    # elif model.startswith("llama"):
    #     messages = payload["messages"]
    #     max_tokens = payload["max_tokens"]
    #     top_p = payload["top_p"]
    #     temperature = payload["temperature"]

    #     llama_messages = []

    #     for i, message in enumerate(messages):
    #         llama_message = {
    #             "role": message["role"],
    #             "content": ""
    #         }
    #         for part in message["content"]:
    #             llama_message['content'] = part['text'] if part['type'] == "text" else ""
    #             llama_messages.append(llama_message)   
    #         llama_messages.append(llama_message)

    #     for i in range(3):
    #         try:
    #             logger.info("Generating content with model: %s", model)
    #             response = dashscope.Generation.call(
    #                 model=model,
    #                 messages=llama_messages,
    #                 result_format="message",
    #                 max_length=max_tokens,
    #                 top_p=top_p,
    #                 temperature=temperature,
    #                 stop = stop
    #             )
    #             import pdb; pdb.set_trace()
    #             return True, response['output']['choices'][0]['message']['content']

    #         except Exception as e:
    #             logger.error("Failed to call LLM: " + str(e))
    #             time.sleep(4 * (2 ** (i + 1)))
    #             if hasattr(e, 'response'):
    #                 error_info = e.response.json()  
    #                 code_value = error_info['error']['code']
    #                 if code_value == "content_filter":
    #                     if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
    #                         payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
    #             else:
    #                 code_value = "context_length_exceeded"
    #             logger.error("Retrying ...")
    #     return False, code_value
    
    
    
    elif model.startswith("deepseek"):
        
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]
        from openai import OpenAI
        
        deepseek_messages = []

        for i, message in enumerate(messages):
            deepseek_message = {
                "role": message["role"],
                "content": ""
            }
            for part in message["content"]:
                deepseek_message['content'] = part['text'] if part['type'] == "text" else ""
                deepseek_messages.append(deepseek_message)
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        for i in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=deepseek_messages,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature
                )
                output_message = json.loads(response.json())['choices'][0]['message']['content']
                return True, output_message
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                if hasattr(e, 'response') and e.response is not None:
                    error_info = e.response.json()  
                    code_value = error_info['error']['code']

                    if code_value == "content_filter":
                        last_message = messages[-1]
                        if 'content' in last_message and isinstance(last_message['content'], str):
                            if not last_message['content'].endswith("They do not represent any real events or entities. ]"):
                                last_message['content'] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                        else:
                            logger.error("Unexpected message structure in 'messages'. Skipping content modification.")
                    elif code_value == "context_length_exceeded":
                        return False, code_value        
                else:
                    code_value = 'unknown_error'
                
                logger.error("Retrying ...")
                time.sleep(10 * (2 ** (i + 1)))
        return False, code_value


        
    elif model.startswith("qwen") or model.startswith("Qwen") or model.startswith("llama3.1"):
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]
        
        if model.startswith("llama3.1"):
            max_tokens = 2000

        qwen_messages = []

        for i, message in enumerate(messages):
            qwen_message = {
                "role": message["role"],
                "content": []
            }
            assert len(message["content"]) in [1, 2], "One text, or one text with one image"
            for part in message["content"]:
                qwen_message['content'].append({"text": part['text']}) if part['type'] == "text" else None

            qwen_messages.append(qwen_message)

        for i in range(3):
            try:
                logger.info("Generating content with model: %s", model)
                response = dashscope.Generation.call(
                    model=model,
                    messages=qwen_messages,
                    result_format="message",
                    max_length=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    stop = stop
                )
                return True, response['output']['choices'][0]['message']['content']

            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(3 * (2 ** (i + 1)))
                if hasattr(e, 'response'):
                    error_info = e.response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                else:
                    code_value = "context_length_exceeded"
                logger.error("Retrying ...")
        return False, code_value


        
    elif model.startswith("codellama") or model.startswith("mistralai"):
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        if model == "codellama/CodeLlama-70b-Instruct-hf":
            max_tokens = 800
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        mistral_messages = []

        for i, message in enumerate(messages):
            mistral_message = {
                "role": message["role"],
                "content": ""
            }

            for part in message["content"]:
                mistral_message['content'] = part['text'] if part['type'] == "text" else ""

            mistral_messages.append(mistral_message)

        from openai import OpenAI

        client = OpenAI(api_key=os.environ["TOGETHER_API_KEY"],
                        base_url='https://api.together.xyz',
                        )

        for i in range(3):
            try:
                logger.info("Generating content with model: %s", model)
                response = client.chat.completions.create(
                    messages=mistral_messages,
                    model=model,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature
                )
                return True, response.choices[0].message.content
                
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(10 * (2 ** (i + 1)))
                # if hasattr(e, 'response'):
                #     error_info = e.response  
                #     code_value = error_info['error']['param']
                #     if "content" in code_value:
                #         if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                #             payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                #     if code_value == "max_tokens":
                #         return False, code_value        
                # else:
                code_value = "context_length_exceeded"
                logger.error("Retrying ...")

        return False, code_value



    elif model == "gemini-1.5-pro-latest":
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        gemini_messages = []

        for i, message in enumerate(messages):
            gemini_message = {
                "role": message["role"],
                "content": []
            }
            assert len(message["content"]) in [1, 2], "One text, or one text with one image"
            for part in message["content"]:

                if part['type'] == "image_url":
                    image_source = {}
                    image_source["type"] = "base64"
                    image_source["media_type"] = "image/png"
                    image_source["data"] = part['image_url']['url'].replace("data:image/png;base64,", "")
                    gemini_message['content'].append({"type": "image", "source": image_source})

                if part['type'] == "text":
                    gemini_message['content'].append({"type": "text", "text": part['text']})

            gemini_messages.append(gemini_message)

        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {os.environ["GEMINI_API_KEY"]}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }  
        
        payload = json.dumps({"model": model,"messages": gemini_messages,"max_tokens": max_tokens,"temperature": temperature,"top_p": top_p})


        
        for i in range(3):
            try:
                response = requests.request("POST", "https://api2.aigcbest.top/v1/chat/completions", headers=headers, data=payload)
                logger.info(f"response_code {response.status_code}")
                if response.status_code == 200:
                    return True, response.json()['choices'][0]['message']['content']
                else:
                    error_info = response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value
                    logger.error("Retrying ...")
                    time.sleep(10 * (2 ** (i + 1)))
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(10 * (2 ** (i + 1)))
                code_value = "context_length_exceeded"
        return False, code_value