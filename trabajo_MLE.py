import streamlit as st
from diffusers import StableDiffusionPipeline
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image
import torch

st.set_page_config(layout='wide')

st.markdown(
    """
    <h1 style='text-align: center; color: #0174DF ;'>
        TRABAJO DE MODELOS PREENTRENADOS DE HUGGINGFACE
    </h1> 
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align: right; font-size: 16px; color: #00008B;'>
        Desarrollado por:  <b>Jorge Jara Arenas</b>
    </div>
    """,
    unsafe_allow_html=True,
)
if "generated_image" not in st.session_state:
    st.session_state["generated_image"] = None

if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

if "predicted_class" not in st.session_state:
    st.session_state["predicted_class"] = None

@st.cache_resource
def load_stable_diffusion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.to("cpu")
    return pipe

# **Función para cargar el modelo de clasificación de imágenes**
@st.cache_resource
def load_resnet_model():
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    return processor, model

pipe = load_stable_diffusion()
processor, model = load_resnet_model()

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.title("Generador de Imágenes con el modelo Stable Diffusion v1-4")
        prompt = st.text_input("Escribe tu solicitud para la generación de imágenes:", "")
        if st.button("Generar Imagen"):
            if prompt.strip() == "":
                st.error("Por favor, ingresa un texto válido.")
            else:
                with st.spinner("Generando imagen..."):
                    try:
                        st.session_state["generated_image"] = pipe(prompt).images[0]
                    except Exception as e:
                        st.error(f"Error al generar la imagen: {str(e)}")
        if st.session_state.generated_image is not None:
            st.image(st.session_state.generated_image, caption="Imagen Generada", use_column_width=True)
        
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.title("Clasificador de Imágenes con el modelo ResNet-50")
        uploaded_file = st.file_uploader("Presiona el botón para cargar una imagen", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state["uploaded_image"] = Image.open(uploaded_file)

        if st.session_state["uploaded_image"] is not None:
            st.image(st.session_state["uploaded_image"], caption="Imagen cargada", use_column_width=True)
            st.write("Clasificando la imagen...")
            inputs = processor(images=st.session_state["uploaded_image"], return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                st.session_state["predicted_class"] = model.config.id2label[predicted_class_idx]
        if st.session_state.predicted_class is not None:
            st.success(f"¡Clasificación completada! La clase predicha es: **{st.session_state.predicted_class}**")




