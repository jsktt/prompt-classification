import gradio as gr
import os
import time
import torch
import pandas as pd
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nemoguardrails import RailsConfig, LLMRails

# global current prompt variable
current_prompt = ""

# ====== ENVIRONMENT SETUP ======
os.environ['OPENAI_API_KEY'] = 'API_KEY_HERE' # your open ai api key here


#====== DEVICE CHECK ======
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


#Get the directory where this script is located
project_dir = os.path.dirname(os.path.abspath(__file__))

#Construct the relative model path
model_path = os.path.join(project_dir, "models", "roBERTa_Large_Prompt_Classification_Model")

#====== MODEL LOADING ======
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ====== CONFIG FILE SETUP ======
textconfig = RailsConfig.from_path("./textconfig")
textrails = LLMRails(textconfig)

imageconfig = RailsConfig.from_path("./imageconfig")
imagerails = LLMRails(imageconfig)

openai.api_key = os.getenv("OPENAI_API_KEY")


# ====== CLASSIFIER FUNCTION ======
def classify_prompt(prompt: str) -> str:
    global current_prompt
    current_prompt = prompt.strip()
    """Classify the input prompt as text or image."""
    inputs = tokenizer(prompt, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    prediction = probabilities.argmax()  # binary classification: 0=image, 1=text
    category = "text" if prediction == 1 else "image"
    score = f"score: {probabilities}"
    return category, score


# ======= hangle misclassified prompts ========
FAILED_PROMPTS_FILE = os.path.join(project_dir, "datasets", "failed_prompts_dataset.csv")

def save_failed_prompt(prompt: str, correct_label: str):
    new_entry = pd.DataFrame([[prompt, correct_label]], columns=['prompt', 'class'])
    if os.path.exists(FAILED_PROMPTS_FILE):
        new_entry.to_csv(FAILED_PROMPTS_FILE, mode='a', header=False, index=False)
    else:
        new_entry.to_csv(FAILED_PROMPTS_FILE, mode='w', header=True, index=False)
    return f"Saved misclassified prompt as '{correct_label}' to failed dataset."


# ====== GRADIO INTERFACE ======
def gradio_interface(prompt):
    # Classify the prompt
    start_time = time.perf_counter() * 1000
    classification, score = classify_prompt(prompt)
    end_time = time.perf_counter() * 1000
    class_time = end_time - start_time

    info_text = f"Classified as a **{classification.upper()}** prompt in {class_time:.2f} ms.\nWith score of {score}"

    # Store state (prompt and classification)
    state_data = (prompt, classification)

    if classification == "text":
        response = handle_text_prompt(prompt)
        full_response = f"{info_text}\n\nResponse:\n{response}"
        return full_response, None, state_data
    else:
        image_url = handle_image_prompt(prompt)
        full_response = f"{info_text}\n\nImage Model Respponse: {image_url}."
        return full_response, image_url, state_data

def feedback_flip(state):
    global current_prompt
    if not current_prompt:
        return "No prompt to flip classification for."
    
    if state is None:
        return "No classification to correct yet.", None, None, None  # Add more outputs as needed

    prompt, predicted_label = state
    correct_label = 'image' if predicted_label == 'text' else 'text'

    # Save to dataset
    msg = save_failed_prompt(prompt, correct_label)

    # Generate new output based on the correct label
    if correct_label == "text":
        response = handle_text_prompt(prompt)
        return msg, response, None, (prompt, correct_label)
    else:
        try:
            image_url = handle_image_prompt(prompt)
            return msg, None, image_url, (prompt, correct_label)
        except Exception as e:
            return f"{msg} (but failed to generate image: {str(e)})", None, None, (prompt, correct_label)



# ====== IMAGE GENERATION ======
def generate_image_with_dalle(prompt):
    print(f"Sending prompt to DALL·E: {repr(prompt)}")  # Use repr to catch invisible characters
    if not prompt or len(prompt.strip()) < 5:
        raise ValueError("Prompt too short or empty for DALL·E.")
    
    print("Prompt allowed. Sending to DALL·E...")
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
        quality="standard"
    )
    image_url = response.data[0].url
    print(f"Generated image URL: {image_url}")
    return image_url  # for GUI use


# ====== IMAGE PROMPT HANDLER ======
def handle_image_prompt(user_prompt):
    response = imagerails.generate(prompt=user_prompt)
    info = imagerails.explain()

    # Only the initial check passed
    if len(info.llm_calls) == 1:
        return response
    return generate_image_with_dalle(user_prompt)


# ====== TEXT PROMPT HANDLER ======
def handle_text_prompt(user_prompt):
    response = textrails.generate(prompt=user_prompt)
    return response


# Gradio UI setup
with gr.Blocks() as demo:
    state = gr.State()

    with gr.Row():
        prompt_input = gr.Textbox(label="Enter your prompt", lines=2)
        submit_btn = gr.Button("Submit")

    with gr.Row():
        response_text = gr.Textbox(label="Response", lines=10)
        image_output = gr.Image(label="Generated Image", visible=True)

    with gr.Row():
        feedback_btn = gr.Button("Flip Classification (Incorrect?)")
        feedback_msg = gr.Textbox(label="Feedback Result", interactive=False)

    # === CALLBACK WIRING (place this here) ===
    prompt_input.submit(
        fn=gradio_interface,
        inputs=prompt_input,
        outputs=[response_text, image_output, state]
    )

    submit_btn.click(
        fn=gradio_interface,
        inputs=prompt_input,
        outputs=[response_text, image_output, state]
    )

    feedback_btn.click(
        fn=feedback_flip,
        inputs=state,
        outputs=[feedback_msg, response_text, image_output, state]
    )




demo.launch()
