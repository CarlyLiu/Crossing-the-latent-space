'''
chatgpt辅助理解源码
'''
import re
import time
import threading
from typing import List, Optional, Tuple, Union
import glob
import click
import math
import os
import cv2
import PIL.Image
import torch

import legacy

import multiprocessing
import numpy as np
import mediapipe as mp
import imgui
import dnnlib
from gui_utils import imgui_window
from gui_utils import imgui_utils
from gui_utils import gl_utils
from gui_utils import text_utils
from viz import renderer

#----------------------------------------------------------------------------

class Visualizer(imgui_window.ImguiWindow):
    def __init__(self, capture_dir=None):
        super().__init__(title='GAN Visualizer',  window_width=3840, window_height=2160)

        # Internals.
        self._last_error_print  = None
        self._async_renderer    = AsyncRenderer()
        self._defer_rendering   = 0
        self._tex_img           = None
        self._tex_obj           = None

        # Widget interface.
        self.args               = dnnlib.EasyDict()
        self.result             = dnnlib.EasyDict()
        self.pane_w             = 0
        self.label_w            = 0
        self.button_w           = 0

        # Widgets.
        self.pickle_widget      = PickleWidget(self)
        self.latent_widget      = LatentWidget(self)

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame() # Layout may change after first frame.

    def close(self):
        super().close()
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print('\n' + error + '\n')
            self._last_error_print = error

    def defer_rendering(self, num_frames=1):
        self._defer_rendering = max(self._defer_rendering, num_frames)

    def clear_result(self):
        self._async_renderer.clear_result()

    def set_async(self, is_async):
        if is_async != self._async_renderer.is_async:
            self._async_renderer.set_async(is_async)
            self.clear_result()
            if 'image' in self.result:
                self.result.message = 'Switching rendering process...'
                self.defer_rendering()

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 60, self.content_height / 30))
        if self.font_size != old:
            self.skip_frame() # Layout changed.

    def draw_frame(self):
        self.begin_frame()
        self.args = dnnlib.EasyDict()
        self.pane_w = self.font_size * 45
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)

        # Begin control pane.
        # imgui.set_next_window_position(0, 0)
        # imgui.set_next_window_size(1, 1)
        imgui.begin('##control_pane', closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))

        # Widgets.
        expanded, _visible = imgui_utils.collapsing_header('Latent Data', default=True)
        self.pickle_widget()
        self.latent_widget(expanded)
        # self.capture_widget()

        # Render.
        if self.is_skipping_frames():
            pass
        elif self._defer_rendering > 0:
            self._defer_rendering -= 1
        elif self.args.pkl is not None:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            if result is not None:
                self.result = result

        # Display.
        max_w = self.content_width
        max_h = self.content_height
        pos = np.array([max_w / 2 - self.pane_w /2, max_h / 2])
        if 'image' in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            zoom = np.floor(zoom) if zoom >= 1 else zoom
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)
        if 'error' in self.result:
            self.print_error(self.result.error)
            if 'message' not in self.result:
                self.result.message = str(self.result.error)
        if 'message' in self.result:
            tex = text_utils.get_texture(self.result.message, size=self.font_size, max_width=max_w, max_height=max_h, outline=2)
            tex.draw(pos=pos, align=0.5, rint=True, color=1)

        # End frame.
        self._adjust_font_size()
        imgui.end()
        self.end_frame()

    def run_tracker(self):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        tracker = handTracker(latent_widget=self.latent_widget)  # Pass the LatentWidget instance to handTracker.

        while not self.should_close():
            success, image = cap.read()
            flipped_image = cv2.flip(image, 1)
            flipped_image = tracker.handsFinder(flipped_image)

            lmList = tracker.positionFinder(flipped_image)
            if len(lmList) != 0:
                # print(lmList[0])

                # Detect mouse dragging in the result area.
                # dragging, dx, dy = imgui_utils.drag_hidden_window('##result_area', x=self.pane_w, y=0, width=self.content_width-self.pane_w, height=self.content_height-self.pane_h)

                self.latent_widget.drag(lmList[0][1], lmList[0][2])
                # self.result.message =  f"{lmList[0][1]}, {lmList[0][2]}"

            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Video", 960, 540)

            cv2.imshow("Video", flipped_image)
            cv2.waitKey(1)
#----------------------------------------------------------------------------
class AsyncRenderer:
    def __init__(self):
        self._closed        = False
        self._is_async      = False
        self._cur_args      = None
        self._cur_result    = None
        self._cur_stamp     = 0
        self._renderer_obj  = None
        self._args_queue    = None
        self._result_queue  = None
        self._process       = None

    def close(self):
        self._closed = True
        self._renderer_obj = None
        if self._process is not None:
            self._process.terminate()
        self._process = None
        self._args_queue = None
        self._result_queue = None

    @property
    def is_async(self):
        return self._is_async

    def set_async(self, is_async):
        self._is_async = is_async

    def set_args(self, **args):
        assert not self._closed
        if args != self._cur_args:
            if self._is_async:
                self._set_args_async(**args)
            else:
                self._set_args_sync(**args)
            self._cur_args = args

    def _set_args_async(self, **args):
        if self._process is None:
            self._args_queue = multiprocessing.Queue()
            self._result_queue = multiprocessing.Queue()
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass
            self._process = multiprocessing.Process(target=self._process_fn, args=(self._args_queue, self._result_queue), daemon=True)
            self._process.start()
        self._args_queue.put([args, self._cur_stamp])

    def _set_args_sync(self, **args):
        if self._renderer_obj is None:
            self._renderer_obj = renderer.Renderer()
        self._cur_result = self._renderer_obj.render(**args)

    def get_result(self):
        assert not self._closed
        if self._result_queue is not None:
            while self._result_queue.qsize() > 0:
                result, stamp = self._result_queue.get()
                if stamp == self._cur_stamp:
                    self._cur_result = result
        return self._cur_result

    def clear_result(self):
        assert not self._closed
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp += 1

    @staticmethod
    def _process_fn(args_queue, result_queue):
        renderer_obj = renderer.Renderer()
        cur_args = None
        cur_stamp = None
        while True:
            args, stamp = args_queue.get()
            while args_queue.qsize() > 0:
                args, stamp = args_queue.get()
            if args != cur_args or stamp != cur_stamp:
                result = renderer_obj.render(**args)
                if 'error' in result:
                    result.error = renderer.CapturedException(result.error)
                result_queue.put([result, stamp])
                cur_args = args
                cur_stamp = stamp
#----------------------------------------------------------------------------
class PickleWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.recent_pkls    = []
        self.load(r'C:\Users\project\stylegan3\stylegan3-t-afhqv2-512x512.pkl', ignore_errors=True)

    def load(self, pkl, ignore_errors=False):
        viz = self.viz
        viz.clear_result()
        viz.skip_frame() # The input field will change on next frame.
        try:
            resolved = self.resolve_pkl(pkl)
            name = resolved.replace('\\', '/').split('/')[-1]
            self.cur_pkl = resolved
            self.user_pkl = resolved
            viz.result.message = f'Loading {name}...'
            viz.defer_rendering()
            if resolved in self.recent_pkls:
                self.recent_pkls.remove(resolved)
            self.recent_pkls.insert(0, resolved)
        except:
            self.cur_pkl = None
            self.user_pkl = pkl
            if pkl == '':
                viz.result = dnnlib.EasyDict(message='No network pickle loaded')
            else:
                viz.result = dnnlib.EasyDict(error=renderer.CapturedException())
            if not ignore_errors:
                raise

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        recent_pkls = [pkl for pkl in self.recent_pkls if pkl != self.user_pkl]

        paths = viz.pop_drag_and_drop_paths()
        if paths is not None and len(paths) >= 1:
            self.load(paths[0], ignore_errors=True)

        viz.args.pkl = self.cur_pkl

    def resolve_pkl(self, pattern):
        assert isinstance(pattern, str)
        assert pattern != ''

        # URL => return as is.
        if dnnlib.util.is_url(pattern):
            return pattern

        # Short-hand pattern => locate.
        path = pattern

        # Run dir => pick the last saved snapshot.
        if os.path.isdir(path):
            pkl_files = sorted(glob.glob(os.path.join(path, 'network-snapshot-*.pkl')))
            if len(pkl_files) == 0:
                raise IOError(f'No network pickle found in "{path}"')
            path = pkl_files[-1]

        # Normalize.
        path = os.path.abspath(path)
        return path
#----------------------------------------------------------------------------
class LatentWidget:
    def __init__(self, viz):
        self.viz        = viz
        self.latent     = dnnlib.EasyDict(x=0, y=0, anim=False, speed=0.25)
        self.latent_def = dnnlib.EasyDict(self.latent)
        self.step_y     = 100

    def drag(self, dx, dy):
        # viz = self.viz
        # self.latent.x = dx / viz.font_size * 4e-1
        # self.latent.y = dy / viz.font_size * 4e-1
        viz = self.viz
        self.latent.x = dx / (4e+2)
        self.latent.y = dy / (4e+2)


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Latent')
            imgui.same_line(viz.label_w)
            seed = round(self.latent.x) + round(self.latent.y) * self.step_y
            with imgui_utils.item_width(viz.font_size * 8):
                changed, seed = imgui.input_int('##seed', seed)
                if changed:
                    self.latent.x = seed
                    self.latent.y = 0
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            frac_x = self.latent.x - round(self.latent.x)
            frac_y = self.latent.y - round(self.latent.y)
            with imgui_utils.item_width(viz.font_size * 5):
                changed, (new_frac_x, new_frac_y) = imgui.input_float2('##frac', frac_x, frac_y, format='%+.2f', flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if changed:
                    self.latent.x += new_frac_x - frac_x
                    self.latent.y += new_frac_y - frac_y
            imgui.same_line(viz.label_w + viz.font_size * 13 + viz.spacing * 2)

        if self.latent.anim:
            self.latent.x += viz.frame_delta * self.latent.speed
        viz.args.w0_seeds = [] # [[seed, weight], ...]
        for ofs_x, ofs_y in [[0, 0], [1, 0], [0, 1], [1, 1]]:
            seed_x = np.floor(self.latent.x) + ofs_x
            seed_y = np.floor(self.latent.y) + ofs_y
            seed = (int(seed_x) + int(seed_y) * self.step_y) & ((1 << 32) - 1)
            weight = (1 - abs(self.latent.x - seed_x)) * (1 - abs(self.latent.y - seed_y))
            if weight > 0:
                viz.args.w0_seeds.append([seed, weight])
#----------------------------------------------------------------------------
class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5, latent_widget=None):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.latent_widget = latent_widget # Store the reference to the LatentWidget instance.
        self.last_latent_widget = None  # Store the last LatentWidget reference.
        # 添加以下变量用于计算帧率
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def calculate_fps(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:  # 每秒更新一次
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        self.calculate_fps()
        cv2.putText(image, f'FPS: {int(self.fps)}', (1100, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        thumb_tip = None
        index_tip = None
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
                # if draw and (id == 4 or id == 8):
                #     cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)
                # cv2.putText(image, f'({cx}, {cy})', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                #             cv2.LINE_AA)

                # 检查食指指尖和拇指指尖的坐标
                if id == 4:  # 拇指指尖
                    thumb_tip = (cx, cy)
                    if draw:
                        cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                elif id == 8:  # 食指指尖
                    index_tip = (cx, cy)
                    if draw:
                        cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                        cv2.putText(image, f'({cx}, {cy})', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                                    cv2.LINE_AA)

                if thumb_tip is not None and index_tip is not None:
                    distance = math.sqrt((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2)
                    if distance <= 20:
                        if self.set_latent_widget(self.latent_widget):
                            # self.dump_image = True
                            # self.defer_frames = 2
                            # self.disabled_time = 0.5
                            print(f"pinch detected")
        return lmlist

    def set_latent_widget(self, latent_widget):
        if self.last_latent_widget != latent_widget and latent_widget is not None:
            self.last_latent_widget = latent_widget
            self.latent_widget = latent_widget
            return True
        return False
#----------------------------------------------------------------------------

class CaptureWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.path           = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_screenshots'))
        self.dump_image     = False
        self.dump_gui       = False
        self.defer_frames   = 0
        self.disabled_time  = 0

    def dump_png(self, image):
        viz = self.viz
        try:
            _height, _width, channels = image.shape
            assert channels in [1, 3]
            assert image.dtype == np.uint8
            os.makedirs(self.path, exist_ok=True)
            file_id = 0
            for entry in os.scandir(self.path):
                if entry.is_file():
                    match = re.fullmatch(r'(\d+).*', entry.name)
                    if match:
                        file_id = max(file_id, int(match.group(1)) + 1)
            if channels == 1:
                pil_image = PIL.Image.fromarray(image[:, :, 0], 'L')
            else:
                pil_image = PIL.Image.fromarray(image, 'RGB')
            pil_image.save(os.path.join(self.path, f'{file_id:05d}.png'))
        except:
            viz.result.error = renderer.CapturedException()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        # if show:
        #     with imgui_utils.grayed_out(self.disabled_time != 0):
        #         imgui.text('Capture')
        #         imgui.same_line(viz.label_w)
        #         _changed, self.path = imgui_utils.input_text('##path', self.path, 1024,
        #             flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
        #             width=(-1 - viz.button_w * 2 - viz.spacing * 2),
        #             help_text='PATH')
        #         if imgui.is_item_hovered() and not imgui.is_item_active() and self.path != '':
        #             imgui.set_tooltip(self.path)
        #         imgui.same_line()
        #         if imgui_utils.button('Save image', width=viz.button_w, enabled=(self.disabled_time == 0 and 'image' in viz.result)):
        #             self.dump_image = True
        #             self.defer_frames = 2
        #             self.disabled_time = 0.5
        #         imgui.same_line()
        #         if imgui_utils.button('Save GUI', width=-1, enabled=(self.disabled_time == 0)):
        #             self.dump_gui = True
        #             self.defer_frames = 2
        #             self.disabled_time = 0.5

        self.disabled_time = max(self.disabled_time - viz.frame_delta, 0)
        if self.defer_frames > 0:
            self.defer_frames -= 1
        elif self.dump_image:
            if 'image' in viz.result:
                self.dump_png(viz.result.image)
            self.dump_image = False
        elif self.dump_gui:
            viz.capture_next_frame()
            self.dump_gui = False
        captured_frame = viz.pop_captured_frame()
        if captured_frame is not None:
            self.dump_png(captured_frame)

def main(
):
    viz = Visualizer()

    tracker_thread = threading.Thread(target=viz.run_tracker)
    tracker_thread.start()

    while not viz.should_close():
        viz.draw_frame()

    tracker_thread.join()
    viz.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
