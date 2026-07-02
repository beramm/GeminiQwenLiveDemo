import base64
import os
import dashscope


# The following URL is for the Singapore region. When you call the API, replace {WorkspaceId} with your actual workspace ID. The URL varies by region.
# dashscope.base_http_api_url = 'https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api/v1'
dashscope.base_http_api_url = "https://ws-th0sustwxmii65s1.ap-southeast-1.maas.aliyuncs.com/api/v1"

# Encoding function: Converts a local file to a Base64-encoded string
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

# Replace xxxx/test.mp4 with the absolute path of your local video
base64_video = encode_video("/Users/chandra/Work/ai-assessment/gemini-live-api-examples/gemini-live-genai-python-sdk/testVideo.mov")

messages = [{'role':'user',
                # The fps parameter controls the number of frames extracted from the video. It indicates that one frame is extracted every 1/fps seconds.
             'content': [{'video': f"data:video/mov;base64,{base64_video}","fps":2}]}]

print("Base64 length:", len(base64_video))

response = dashscope.MultiModalConversation.call(
    # API keys vary by region. To obtain an API key, see https://www.alibabacloud.com/help/zh/model-studio/get-api-key
    # If you have not configured the environment variable, replace the following line with your Model Studio API key: api_key="sk-xxx"
    api_key="sk-ws-H.IPYIEI.sTvN.MEUCIAmWbITCkg7u2xhaOs_JlasEVtLArd6VSDAqbqxPkDQ9AiEA0KyTAD0irC7HpGb170ikSAvsgJFfqya7wJ_b4GB2lWQ",
    model='qwen3.7-plus',
    messages=messages)

print("Response:", response)
print(response.output.choices[0].message.content[0]["text"])