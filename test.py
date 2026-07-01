import os

path = "/Users/chandra/Work/ai-assessment/gemini-live-api-examples/gemini-live-genai-python-sdk/testVideo.mov"

print("exists:", os.path.exists(path))
print("size:", os.path.getsize(path))