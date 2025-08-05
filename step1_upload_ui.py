import gradio as gr
import shutil
import os

UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_video(file):
    if file is None:
        return "No file uploaded."
    dest_path = os.path.join(UPLOAD_DIR, os.path.basename(file.name))
    shutil.copy(file.name, dest_path)
    return f"✅ File saved to: {dest_path}"

iface = gr.Interface(
    fn=save_video,
    inputs=gr.Video(label="Upload a video"),
    outputs="text",
    title="🎬 Magma: Upload Video",
    description="Upload a Hindi video. We'll process it and translate it to Tamil with different speaker voices!"
)

if __name__ == "__main__":
    iface.launch()
