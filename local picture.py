import imgui
import numpy as np
from PIL import Image
from imgui.integrations.pyglet import PygletRenderer
import pyglet
from OpenGL.GL import *


# 加载本地JPEG图像
def load_jpeg_image(file_path):
    image = Image.open(file_path)
    return np.array(image)


def main():
    # 创建Pyglet窗口
    window = pyglet.window.Window(width=800, height=600)
    imgui.create_context()
    impl = PygletRenderer(window)

    # 本地JPEG图片路径
    jpeg_image_path = "path_to_your_image.jpg"  # 替换为本地JPEG图片路径

    # 初始化变量
    jpeg_image_data = None
    texture_id = None

    @window.event
    def on_draw():
        nonlocal jpeg_image_data, texture_id  # 保持变量在闭包中的状态

        window.clear()
        impl.process_inputs()
        imgui.new_frame()

        # 在ImGui中显示JPEG图像
        imgui.begin("JPEG Image Viewer")

        if jpeg_image_data is not None:
            # 尝试使用create_texture方法，如果不可用，则手动创建OpenGL纹理
            if hasattr(impl, 'create_texture'):
                texture_id = impl.create_texture(jpeg_image_data)
            else:
                # 手动创建OpenGL纹理
                texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, jpeg_image_data.shape[1], jpeg_image_data.shape[0], 0, GL_RGB,
                             GL_UNSIGNED_BYTE, jpeg_image_data)

            imgui.image(texture_id, width=jpeg_image_data.shape[1], height=jpeg_image_data.shape[0])
        imgui.end()

        imgui.render()
        impl.render(imgui.get_draw_data())

    # 加载本地JPEG图像数据
    jpeg_image_data = load_jpeg_image(r"C:\Users\project\stylegan3\try3\interpolated_image_000.png")

    pyglet.app.run()


if __name__ == "__main__":
    main()
