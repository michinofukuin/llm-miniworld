import openai
class gpt:
    def __init__(self,):
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        # Your Azure OpenAI resource's endpoint value.
        openai.api_base = "https://midivi-main-scu1.openai.azure.com"
        openai.api_key = "f825f61246354ec090c5703ca4f76418"