import numpy as np
import cv2
import dtcwt
from scipy.signal import correlate2d
from tqdm import tqdm

import time
import multiprocessing
import heapq

from .utils import rebin, generate_wm

default_scale = 7.0

class DtcwtKeyEncoder:

    def __init__(self, str=1.0, step=5.0):
        self.alpha = default_scale * str
        self.step = step

    def encode(self, img):
        wm_transform = dtcwt.Transform2d()
        wm_coeffs = wm_transform.forward(self.wm, nlevels=1)
        img_transform = dtcwt.Transform2d()
        img_coeffs = img_transform.forward(img[:, :, 1], nlevels=3)
        y_transform = dtcwt.Transform2d()
        y_coeffs = y_transform.forward(img[:, :, 0], nlevels=3)

        # Masks for level 3 subbands
        masks3 = [0 for i in range(6)]
        shape3 = y_coeffs.highpasses[2][:, :, 0].shape
        for i in range(6):
            masks3[i] = cv2.filter2D(np.abs(y_coeffs.highpasses[1][:,:,i]), -1, np.array([[1/4, 1/4], [1/4, 1/4]]))
            masks3[i] = np.ceil(rebin(masks3[i], shape3) / self.step)
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

    def embed(self, key, input_path, output_path):
        img = cv2.imread(input_path)
        assert img is not None, "Image not found in {}".format(input_path)
        img = img.astype(np.float32)
        wm_shape = self.infer_wm_shape(img.shape)
        self.wm = generate_wm(key, wm_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        wmed_img = self.encode(img)
        wmed_img = cv2.cvtColor(wmed_img, cv2.COLOR_YUV2BGR)
        cv2.imwrite(output_path, wmed_img)
        return wmed_img

    def embed_video(self, keys, seq, frag_length, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_size = (int(width), int(height))
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        fps =  cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepare watermarks
        wm_shape = self.infer_wm_shape((frame_size[1], frame_size[0]))
        wms = [generate_wm(key, wm_shape) for key in keys]

        frag_frames = fps * frag_length
        out = cv2.VideoWriter(output_path, int(fourcc), fps, frame_size)
        count = 0
        pbar = tqdm(total=length)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frag_idx = int((count // frag_frames) % len(seq))
                idx = int(seq[frag_idx])
                self.wm = wms[idx]
                count += 1
                pbar.update(1)
                frame = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2YUV)
                wmed_frame = self.encode(frame)
                wmed_frame = cv2.cvtColor(wmed_frame, cv2.COLOR_YUV2BGR)
                wmed_frame = np.clip(wmed_frame, a_min=0, a_max=255)
                wmed_frame = np.around(wmed_frame).astype(np.uint8)
                out.write(wmed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        out.release()
        cap.release()

    def embed_video_async(self, keys, seq, frag_length, video_path, output_path, threads=None):
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_size = (int(width), int(height))
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        fps =  cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepare watermarks
        wm_shape = self.infer_wm_shape((frame_size[1], frame_size[0]))
        wms = [generate_wm(key, wm_shape) for key in keys]

        frag_frames = fps * frag_length
        out = cv2.VideoWriter(output_path, int(fourcc), fps, frame_size)

        pool = multiprocessing.Pool(threads)
        count = 0
        futures = []
        rbar = tqdm(total=length, position=0, desc="Reading:")
        wbar = tqdm(total=length, position=1, desc="Writing:")
        hp = []
        heapq.heapify(hp)
        out_counter = [0]
        callback = lambda x: DtcwtKeyEncoder.callback(x, out, hp, out_counter, wbar)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frag_idx = int((count // frag_frames) % len(seq))
                idx = int(seq[frag_idx])
                rbar.update(1)
                future = pool.apply_async(DtcwtKeyEncoder.encode_async, args=(frame, wms[idx], self.alpha, self.step, count), callback=callback)
                futures.append(future)
                count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        for future in futures:
            future.wait()
        out.release()
        cap.release()

    def encode_async(img, wm, alpha, step, count):
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2YUV)
        wm_transform = dtcwt.Transform2d()
        wm_coeffs = wm_transform.forward(wm, nlevels=1)
        img_transform = dtcwt.Transform2d()
        img_coeffs = img_transform.forward(img[:, :, 1], nlevels=3)
        y_transform = dtcwt.Transform2d()
        y_coeffs = y_transform.forward(img[:, :, 0], nlevels=3)

        # Masks for level 3 subbands
        masks3 = [0 for i in range(6)]
        shape3 = y_coeffs.highpasses[2][:, :, 0].shape
        for i in range(6):
            masks3[i] = cv2.filter2D(np.abs(y_coeffs.highpasses[1][:,:,i]), -1, np.array([[1/4, 1/4], [1/4, 1/4]]))
            masks3[i] = np.ceil(rebin(masks3[i], shape3) / step)
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
        return count, img

    def callback(x, out, hp, out_counter, wbar):
        # Synchronization
        if x[0] != out_counter[0]:
            heapq.heappush(hp, x)
            return
        else:
            out.write(x[1])
            wbar.update(1)
            out_counter[0] += 1
            while len(hp) != 0 and hp[0][0] == out_counter[0]:
                c, frame = heapq.heappop(hp)
                out.write(frame)
                wbar.update(1)
                out_counter[0] += 1

    def infer_wm_shape(self, img_shape):
        w = (((img_shape[0] + 1) // 2 + 1) // 2 + 1) // 2
        h = (((img_shape[1] + 1) // 2 + 1) // 2 + 1) // 2
        if w % 2 == 1:
            w += 1
        if h % 2 == 1:
            h += 1
        return (w, h)

class DtcwtKeyDecoder:

    def __init__(self, str=1.0, step=5.0):
        self.alpha = default_scale * str
        self.step = step

    def decode(self, wmed_img):
        wmed_transform = dtcwt.Transform2d()
        wmed_coeffs = wmed_transform.forward(wmed_img[:, :, 1], nlevels=3)
        y_transform = dtcwt.Transform2d()
        y_coeffs = y_transform.forward(wmed_img[:, :, 0], nlevels=3)

        masks3 = [0 for i in range(6)]
        inv_masks3 = [0 for i in range(6)]
        shape3 = y_coeffs.highpasses[2][:, :, 0].shape
        for i in range(6):
            masks3[i] = cv2.filter2D(np.abs(y_coeffs.highpasses[1][:,:,i]), -1, np.array([[1/4, 1/4], [1/4, 1/4]]))
            masks3[i] = np.ceil(rebin(masks3[i], shape3) / self.step)
            masks3[i][masks3[i] == 0] = 0.01
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

    def detect(self, keys, wmed_img_path, ori_img_shape=(1080, 1920), mode="fast"):
        # Detect if image is watermarked by any of the keys
        # Return index of key if detected, -1 otherwise
        wmed_img = cv2.imread(wmed_img_path).astype(np.float32)
        wmed_img = cv2.resize(wmed_img, (ori_img_shape[1], ori_img_shape[0]))
        wmed_img = cv2.cvtColor(wmed_img, cv2.COLOR_BGR2YUV)
        wm = self.decode(wmed_img)
        corrs = []
        if mode == "fast":
            shape = wm.shape[0] * wm.shape[1]
            nwm = (wm - np.mean(wm)) / np.std(wm)
            for i in range(len(keys)):
                wmk = generate_wm(keys[i], wm.shape)
                nwmk = (wmk - np.mean(wmk)) / np.std(wmk)
                corr = np.sum(nwm * nwmk) / shape
                corrs.append(corr)
        elif mode == "slow":
            for i in range(len(keys)):
                wmk = generate_wm(keys[i], wm.shape)
                c = correlate2d(wm, wmk) / shape
                idx = np.unravel_index(c.argmax(), c.shape)
                corrs.append(c[idx])
        idx = np.argmax(corrs)
        # print(corrs)
        return idx if corrs[idx] > 0.1 else None

    def detect_video(self, keys, frag_length, wmed_video_path, ori_frame_size=(1080, 1920), mode="fast"):
        wmed_cap = cv2.VideoCapture(wmed_video_path)
        fps = wmed_cap.get(cv2.CAP_PROP_FPS)
        length = int(wmed_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frag_frames = int(frag_length * fps)

        wm_shape = self.infer_wm_shape(ori_frame_size)
        shape = wm_shape[0] * wm_shape[1]
        corrs = np.zeros((len(keys), length))
        wmks = [generate_wm(key, wm_shape) for key in keys]
        nwmks = [(wmk - np.mean(wmk)) / np.std(wmk) for wmk in wmks]
        count = 0
        pbar = tqdm(total=length)
        while wmed_cap.isOpened():
            ret, wmed_frame = wmed_cap.read()
            if ret:
                wmed_frame = cv2.resize(wmed_frame.astype(np.float32), (ori_frame_size[1], ori_frame_size[0]))
                wmed_frame = cv2.cvtColor(wmed_frame, cv2.COLOR_BGR2YUV)
                pbar.update(1)
                wm = self.decode(wmed_frame)
                nwm = (wm - np.mean(wm)) / np.std(wm)
                for i in range(len(keys)):
                    corr = np.sum(nwm * nwmks[i]) / shape
                    corrs[i, count] = corr
                # print(count, corrs[:, count])
                count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        frag_nums = length // frag_frames
        seq = ""
        for i in range(frag_nums):
            s = np.sum(corrs[:, frag_frames * i:frag_frames * (i + 1)], axis=1)
            # print(s)
            idx = np.argmax(s)
            if s[idx] > 1:
                seq += str(idx)
            else:
                seq += "#"
        return seq

    def infer_wm_shape(self, img_shape):
        w = (((img_shape[0] + 1) // 2 + 1) // 2 + 1) // 2
        h = (((img_shape[1] + 1) // 2 + 1) // 2 + 1) // 2
        if w % 2 == 1:
            w += 1
        if h % 2 == 1:
            h += 1
        return (w, h)
