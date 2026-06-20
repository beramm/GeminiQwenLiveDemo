from google import genai

client = genai.Client(
    vertexai=True,
    project="tsel-ai-translation-project",
    location="us-central1"
)

models = client.models.list()

for m in models:
    if "3.1" in m.name.lower():
        print(m.name)