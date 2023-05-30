import gradio as gr

def process_image(image, name):
    # 在这里添加您的图片处理代码
    features = "特征值"
    match = "匹配度"
    return image, features, match

ui = gr.Interface(
    fn=process_image,
    inputs=["image", "text"],
    outputs=["image", "text", "text"],
    layout="vertical",
    live=True,
    interpretation="default",
)

ui.launch()
