import streamlit as st
from transformers import pipeline
from PIL import Image
from huggingface_hub import InferenceClient
import os
import openai
from openai.error import OpenAIError
from gradio_client import Client

# Set page configuration
st.set_page_config(
    page_title="Plate Mate - Your Culinary Assistant",
    page_icon="🍽️",
    layout="centered",  # center content for better mobile experience
    initial_sidebar_state="collapsed",
)

def local_css():
    st.markdown(
        """
        <style>
        /* General resets */
        body, html {
            margin: 0;
            padding: 0;
            font-family: "Helvetica Neue", Arial, sans-serif;
            background-color: #f9f9f9;
        }

        /* Container and spacing */
        .css-1aumxhk, .css-keje6w, .css-18e3th9, .css-12oz5g7 {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        /* Title styling */
        .title h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 0.5em;
            color: #333;
        }

        /* Subheader styling */
        h2, h3, h4, h5, h6 {
            color: #555;
            margin-bottom: 0.5em;
        }
        
        /* Adjust image styling */
        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }

        /* On mobile, reduce font sizes and margins */
        @media (max-width: 600px) {
            .title h1 {
                font-size: 1.8em;
            }

            h2, h3, h4 {
                font-size: 1em;
            }

            .stButton button {
                width: 100%;
            }
        }

        /* Sidebar adjustments */
        [data-testid="stSidebar"] {
            width: 250px;
            background: #fff;
        }

        /* Preset images container */
        .preset-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin: 1em 0;
        }
        .preset-container img {
            width: 80px;
            height: 80px;
            object-fit: cover;
            cursor: pointer;
            border: 2px solid transparent;
        }
        .preset-container img:hover {
            border: 2px solid #007BFF;
        }

        </style>
        """, unsafe_allow_html=True
    )

local_css()  # Apply the CSS

# Hugging Face API key
API_KEY = st.secrets["HF_API_KEY"]
client = InferenceClient(api_key=API_KEY)

@st.cache_resource
def load_image_classification_pipeline():
    return pipeline("image-classification", model="Shresthadev403/food-image-classification")

pipe_classification = load_image_classification_pipeline()

def get_ingredients_qwen(food_name):
    messages = [
        {
            "role": "user",
            "content": f"List only the main ingredients for {food_name}. "
                       f"Respond in a concise, comma-separated list without any extra text or explanations."
        }
    ]
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct", messages=messages, max_tokens=50
        )
        generated_text = completion.choices[0]['message']['content'].strip()
        return generated_text
    except Exception as e:
        return f"Error generating ingredients: {e}"

openai.api_key = st.secrets["openai"]

st.markdown('<div class="title"><h1>PlateMate - Your Culinary Assistant</h1></div>', unsafe_allow_html=True)

# Banner Image (Smaller or optional)
banner_image_path = "IR_IMAGE.png"
if os.path.exists(banner_image_path):
    # Display a smaller version of the banner
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.image(banner_image_path, use_container_width=True)
else:
    st.warning(f"Banner image '{banner_image_path}' not found.")




# Preset Images
preset_images = {
    "Pizza": "sample_pizza.png",
    "Salad": "sample_salad.png",
    "Sushi": "sample_sushi.png"
}

selected_preset = st.selectbox("Or choose a preset sample image:", ["None"] + list(preset_images.keys()))
if selected_preset != "None":
    uploaded_file = preset_images[selected_preset]
else:
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if isinstance(uploaded_file, str):
        # Use the preset image
        if os.path.exists(uploaded_file):
            image = Image.open(uploaded_file)
        else:
            st.error(f"Sample image '{uploaded_file}' not found.")
            image = None
    else:
        image = Image.open(uploaded_file)

    if image:
        st.image(image, caption="Selected Image", use_container_width=True)

        if st.button("Classify"):
            with st.spinner("Classifying..."):
                try:
                    predictions = pipe_classification(image)
                    if predictions:
                        top_food = predictions[0]['label']
                        confidence = predictions[0]['score']
                        st.header(f"🍽️ Food: {top_food} ({confidence*100:.2f}% confidence)")

                        # Generate ingredients
                        st.subheader("📝 Ingredients")
                        try:
                            ingredients = get_ingredients_qwen(top_food)
                            st.write(ingredients)
                        except Exception as e:
                            st.error(f"Error generating ingredients: {e}")

                        # Healthier Alternatives
                        st.subheader("💡 Healthier Alternatives")
                        try:
                            # ONLY THIS PART CHANGED:
                            # Use the RAG calling method instead of the OpenAI function
                            client_rag = Client("https://9a8ccf4a3d4ad96ccf.gradio.live/")
                            result = client_rag.predict(query=f"What's a healthy {top_food} recipe, and why is it healthy?", api_name="/get_response")
                            st.write(result)
                        except OpenAIError as e:
                            st.error(f"OpenAI API error: {e}")
                        except Exception as e:
                            st.error(f"Unable to generate healthier alternatives: {e}")
                    else:
                        st.error("No predictions returned from the classification model.")
                except Exception as e:
                    st.error(f"Error during classification: {e}")

else:
    st.info("Please select or upload an image to get started.")


