import numpy as np
import cv2
import dtcwt
from tqdm import tqdm

import time
import multiprocessing

from .utils import rebin, randomize_channel, derandomize_channel

class DtcwtImgEncoder:

    def __init__(self, key=0, alpha=1.5, step=5, blk_shape=(35, 30)):
        self.key = key
        self.alpha = alpha
        self.step = step
        self.blk_shape = blk_shape

    def encode(self, img):
        # img is in YUV
        wm_transform = dtcwt.Transform2d()
        wm_coeffs = wm_transform.forward(self.wm, nlevels=1)
        img_transform = dtcwt.Transform2d()
        img_coeffs = img_transform.forward(img[:, :, 1], nlevels=3)
        y_transform = dtcwt.Transform2d()
        y_coeffs = y_transform.forward(img[:, :, 0], nlevels=3)

        # Masks for the level 3 subbands
        masks3 = [0 for i in range(6)]
        shape3 = y_coeffs.highpasses[2][:, :, 0].shape
        for i in range(6):
            masks3[i] = cv2.filter2D(np.abs(y_coeffs.highpasses[1][:,:,i]), -1, np.array([[1/4, 1/4], [1/4, 1/4]]))
            masks3[i] = np.ceil(rebin(masks3[i], shape3) * (1 / self.step))
            masks3[i] *= 1.0 / max(12.0, np.amax(masks3[i]))
        for i in range(6):
            coeff = wm_coeffs.highpasses[0][:, :, i]
            h, w = coeff.shape
            coeffs = np.zeros(masks3[i].shape, dtype='complex_')
            coeffs[:h, :w] = coeff
            coeffs[-h:, :w] = coeff
            coeffs[:h, -w:] = coeff
            coeffs[-h:, -w:] = coeff
            img_coeffs.highpasses[2][:, :, i] += self.alpha * (masks3[i] * coeffs)
        img[:, :, 1] = img_transform.inverse(img_coeffs)
        return img

    def prepare_wm(self, wm_path, img_shape):
        wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
        assert wm is not None, "Watermark not found in {}".format(wm_path)
        wm_shape = self.infer_wm_shape(img_shape)
        wm = cv2.resize(wm, (wm_shape[1], wm_shape[0]))
        self.wm = (wm > 127).astype(np.uint8) * 255
        self.wm = self.wm.astype(np.int32)
        self.wm[self.wm != 255] = -255
        self.wm = randomize_channel(self.wm, self.key, blk_shape=self.blk_shape)

    def embed(self, wm_path, input_path, output_path):
        # Embed watermark into an image
        img = cv2.imread(input_path)
        assert img is not None, "Image not found in {}".format(input_path)
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2YUV)
        self.prepare_wm(wm_path, img.shape)
        wmed_img = self.encode(img)
        wmed_img = cv2.cvtColor(wmed_img, cv2.COLOR_YUV2BGR)
        cv2.imwrite(output_path, wmed_img)
        return wmed_img

    def embed_video(self, wm_path, video_path, output_path, verbose=True):
        # Embed watermark into a video
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_size = (int(width), int(height))
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        fps =  cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, int(fourcc), fps, frame_size)
        self.prepare_wm(wm_path, (frame_size[1], frame_size[0]))
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                count += 1
                frame = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2YUV)
                if verbose:
                    print("Start processing frame {}".format(count))
                wmed_frame = self.encode(frame)
                wmed_frame = cv2.cvtColor(wmed_frame, cv2.COLOR_YUV2BGR)
                wmed_frame = np.clip(wmed_frame, a_min=0, a_max=255)
                wmed_frame = np.around(wmed_frame).astype(np.uint8)
                out.write(wmed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        # cap.release()
        # out.release()

    def embed_video_async(self, wm_path, video_path, output_path, verbose=True):
        # Embed watermark into a video
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_size = (int(width), int(height))
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        fps =  cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(output_path, int(fourcc), fps, frame_size)
        self.prepare_wm(wm_path, (frame_size[1], frame_size[0]))

        pool = multiprocessing.Pool()
        count = 0
        futures = []
        if verbose:
            rbar = tqdm(total=length, position=0)
            wbar = tqdm(total=length, position=1)
            callback = lambda x: DtcwtImgEncoder.callback_verbose(x, out, wbar)
        else:
            callback = lambda x: DtcwtImgEncoder.callback(x, out)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                count += 1
                frame = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2YUV)
                if verbose:
                    # print("Start processing frame {}".format(count))
                    rbar.update(1)
                future = pool.apply_async(DtcwtImgEncoder.encode_async, args=(frame, self.wm, self.alpha, self.step), callback=callback)
                # encode(frame, self.wm, self.alpha, self.step)
                futures.append(future)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        for future in futures:
            future.wait()
        # cap.release()
        # out.release()

    def infer_wm_shape(self, img_shape):
        w = (((img_shape[0] + 1) // 2 + 1) // 2 + 1) // 2
        h = (((img_shape[1] + 1) // 2 + 1) // 2 + 1) // 2
        if w % 2 == 1:
            w += 1
        if h % 2 == 1:
            h += 1
        return (w, h)

    def encode_async(img, wm, alpha, step):
        # img is in YUV
        wm_transform = dtcwt.Transform2d()
        wm_coeffs = wm_transform.forward(wm, nlevels=1)
        img_transform = dtcwt.Transform2d()
        img_coeffs = img_transform.forward(img[:, :, 1], nlevels=3)
        y_transform = dtcwt.Transform2d()
        y_coeffs = y_transform.forward(img[:, :, 0], nlevels=3)

        # Masks for the level 3 subbands
        masks3 = [0 for i in range(6)]
        shape3 = y_coeffs.highpasses[2][:, :, 0].shape
        for i in range(6):
            masks3[i] = cv2.filter2D(np.abs(y_coeffs.highpasses[1][:,:,i]), -1, np.array([[1/4, 1/4], [1/4, 1/4]]))
            masks3[i] = np.ceil(rebin(masks3[i], shape3) * (1 / step))
            masks3[i] *= 1.0 / max(12.0, np.amax(masks3[i]))
        for i in range(6):
            coeff = wm_coeffs.highpasses[0][:, :, i]
            h, w = coeff.shape
            coeffs = np.zeros(masks3[i].shape, dtype='complex_')
            coeffs[:h, :w] = coeff
            coeffs[-h:, :w] = coeff
            coeffs[:h, -w:] = coeff
            coeffs[-h:, -w:] = coeff
            img_coeffs.highpasses[2][:, :, i] += alpha * (masks3[i] * coeffs)
        img[:, :, 1] = img_transform.inverse(img_coeffs)
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        img = np.clip(img, a_min=0, a_max=255)
        img = np.around(img).astype(np.uint8)
        return img

    def callback(x, out):
        out.write(x)

    def callback_verbose(x, out, wbar):
        wbar.update(1)
        out.write(x)

class DtcwtImgDecoder:

    def __init__(self, key=0, alpha=1.5, step=5, blk_shape=(35, 30)):
        self.key = key
        self.alpha = alpha
        self.step = step
        self.blk_shape = blk_shape

    def decode(self, wmed_img):
        # wmed_img is in YUV
        wmed_transform = dtcwt.Transform2d()
        wmed_coeffs = wmed_transform.forward(wmed_img[:, :, 1], nlevels=3)
        y_transform = dtcwt.Transform2d()
        y_coeffs = y_transform.forward(wmed_img[:, :, 0], nlevels=3)

        masks3 = [0 for i in range(6)]
        inv_masks3 = [0 for i in range(6)]
        shape3 = y_coeffs.highpasses[2][:, :, 0].shape
        for i in range(6):
            masks3[i] = cv2.filter2D(np.abs(y_coeffs.highpasses[1][:,:,i]), -1, np.array([[1/4, 1/4], [1/4, 1/4]]))
            masks3[i] = np.ceil(rebin(masks3[i], shape3) * (1.0 / self.step))
            masks3[i][masks3[i] == 0] = 0.01
            masks3[i] *= 1.0 / max(12.0, np.amax(masks3[i]))
            inv_masks3[i] = 1.0 / masks3[i]

        shape = wmed_coeffs.highpasses[2][:,:,i].shape
        h, w = (shape[0] + 1) // 2, (shape[1] + 1) // 2
        coeffs = np.zeros((h, w, 6), dtype='complex_')
        for i in range(6):
            coeff = (wmed_coeffs.highpasses[2][:,:,i]) * inv_masks3[i] * 1 / self.alpha
            coeffs[:,:,i] = coeff[:h, :w] + coeff[:h, -w:] + coeff[-h:, :w] + coeff[-h:, -w:]
        highpasses = tuple([coeffs])
        lowpass = np.zeros((h * 2, w * 2))
        t = dtcwt.Transform2d()
        wm = t.inverse(dtcwt.Pyramid(lowpass, highpasses))
        return wm

    def extract(self, wmed_img_path, output_path):
        wmed_img = cv2.imread(wmed_img_path).astype(np.float32)
        wmed_img = cv2.cvtColor(wmed_img, cv2.COLOR_BGR2YUV)
        wm = self.decode(wmed_img)
        wm = derandomize_channel(wm, self.key, blk_shape=self.blk_shape)
        cv2.imwrite(output_path, wm)
        return wm

    def extract_video(self, wmed_video_path, output_folder, verbose=True):
        wmed_cap = cv2.VideoCapture(wmed_video_path)
        count = 0
        length = int(wmed_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if verbose:
            pbar = tqdm(total=length)
        while wmed_cap.isOpened():
            ret, wmed_frame = wmed_cap.read()
            if ret:
                # wmed_frame = cv2.resize(
                #     wmed_frame, (1920, 1080))
                count += 1
                wmed_frame = cv2.cvtColor(wmed_frame.astype(np.float32), cv2.COLOR_BGR2YUV)
                if verbose:
                    pbar.update(1)
                wm = self.decode(wmed_frame)
                wm = derandomize_channel(wm, self.key, blk_shape=self.blk_shape)
                cv2.imwrite(
                    '{}/frame{}.jpeg'.format(output_folder, count), wm)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
