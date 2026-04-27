import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# =========================
# 🧠 Load Model
# =========================
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# =========================
# 🎨 Page Config
# =========================
st.set_page_config(page_title="Smart Image Caption Generator", layout="centered")

# =========================
# 🎨 Advanced CSS Styling
# =========================
st.markdown("""
<style>

/* 🌈 Full Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    color: white;
}

/* 🎯 Title Styling */
h1 {
    text-align: center;
    font-size: 42px !important;
    font-weight: bold;
    background: linear-gradient(90deg, #a855f7, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* 📄 Subtitle */
p {
    text-align: center;
    color: #cbd5e1;
    font-size: 16px;
}

/* 📦 Upload Box */
[data-testid="stFileUploader"] {
    background-color: #111827;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #374151;
}

/* 🎭 Radio Buttons Center */
.stRadio > div {
    justify-content: center;
}

/* 🔘 Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #a855f7, #06b6d4);
    color: white;
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
    border: none;
    transition: 0.3s;
}

/* ✨ Button Hover */
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #06b6d4;
}

/* 📊 Result Box */
.stAlert {
    background-color: #020617 !important;
    color: #22d3ee !important;
    border-radius: 12px;
    border: 1px solid #334155;
    font-size: 16px;
}

/* 🖼 Image Styling */
img {
    border-radius: 12px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.6);
}

</style>
""", unsafe_allow_html=True)

# =========================
# 🏷 Title
# =========================
st.title("✨ Smart Image Caption Generator")
st.markdown(
    "<h4 style='text-align:center;color:#94a3b8;'>AI-powered captions with style & creativity</h4>",
    unsafe_allow_html=True
)

# =========================
# 📤 Upload Image
# =========================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# =========================
# 🎭 Style Selection
# =========================
style = st.radio(
    "Pick a style",
    ["Descriptive", "Social", "Poetic", "SEO Alt-text", "Funny"],
    horizontal=True
)

# =========================
# 🖼 Show Image
# =========================
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # =========================
    # 🚀 Generate Caption
    # =========================
    if st.button("✨ Generate Caption"):
        with st.spinner("Generating caption..."):
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # 🎯 Styles
            if style == "Funny":
                caption = "😂 " + caption + " (This is hilarious!)"

            elif style == "Poetic":
                caption = "✨ " + caption + ", a story told through light and emotion."

            elif style == "Social":
                caption = caption + " 📸 #vibes #insta #trending"

            elif style == "SEO Alt-text":
                caption = caption + " | AI-generated descriptive alt text for accessibility and SEO."

            elif style == "Descriptive":
                caption = f"""
A detailed view of {caption} is presented in this image. The subject appears prominently in the frame, with clear textures and natural colors that enhance its visual appeal. The lighting adds depth and highlights important features, while the background remains softly blurred to maintain focus on the main subject. Overall, the composition creates a balanced and visually pleasing scene.
"""

        # =========================
        # 📊 Result
        # =========================
        st.subheader("📊 Result")
        st.success(caption)