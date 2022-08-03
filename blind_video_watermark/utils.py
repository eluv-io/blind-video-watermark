import cv2
import numpy as np

class ArnoldTransform:

    def __init__(self, img):
        self.img = img
        self.count = 0
        self.len = self.img.shape[0]
        x, y = np.meshgrid(range(self.len), range(self.len))
        self.x_map = (x + y) % self.len
        self.y_map = (2 * x + y) % self.len

    def transform(self, steps=1):
        for i in range(steps):
            self.img = self.img[self.x_map, self.y_map]
            self.count += 1
        return self.img

def generate_wm(key, shape):
    wm = np.random.RandomState(key).randint(0, 2, (shape[0], shape[1]))
    wm[wm == 0] = -1
    return wm

def randomize_channel(channel, key, blk_shape=(8, 8)):
    rows = channel.shape[0] // blk_shape[0] * blk_shape[0]
    cols = channel.shape[1] // blk_shape[1] * blk_shape[1]
    blks = np.array([[
        channel[i:i + blk_shape[0], j:j + blk_shape[1]]
        for j in range(0, cols, blk_shape[1])
    ] for i in range(0, rows, blk_shape[0])])
    shape = blks.shape
    blks = blks.reshape(-1, blk_shape[0], blk_shape[1])
    np.random.RandomState(key).shuffle(blks)
    full_res = np.copy(channel)
    res = np.concatenate(np.concatenate(blks.reshape(shape), 1), 1)
    full_res[:rows, :cols] = res
    return full_res

def randomize_img(img):
    for i in range(3):
        img[:, :, i] = randomize_channel(img[:, :, i])
    return img

def derandomize_channel(channel, key, blk_shape=(8, 8)):
    rows = channel.shape[0] // blk_shape[0] * blk_shape[0]
    cols = channel.shape[1] // blk_shape[1] * blk_shape[1]
    blks = np.array([[
        channel[i:i + blk_shape[0], j:j + blk_shape[1]]
        for j in range(0, cols, blk_shape[1])
    ] for i in range(0, rows, blk_shape[0])])
    shape = blks.shape
    blks = blks.reshape(-1, blk_shape[0], blk_shape[1])
    blk_num = blks.shape[0]
    indices = np.arange(blk_num)
    np.random.RandomState(key).shuffle(indices)
    res = np.zeros(blks.shape)
    res[indices] = blks
    res = np.concatenate(np.concatenate(res.reshape(shape), 1), 1)
    full_res = np.copy(channel)
    full_res[:rows, :cols] = res
    return full_res


def derandomize_img(img):
    for i in range(3):
        img[:, :, i] = derandomize_channel(img[:, :, i])
    return img

def compute_psnr(img1, img2):
    v = 0.
    for i in range(3):
        v += cv2.PSNR(img1[:,:,i], img2[:,:,i])
    return v / 3.0

def rebin(a, shape):
    if a.shape[0] % 2 == 1:
        a = np.vstack((a, np.zeros((1, a.shape[1]))))
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def average_frames(fpaths, output_path):
    ws = 0
    img = cv2.imread(fpaths[0], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    wm = np.zeros(img.shape)
    for path in fpaths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        w = np.mean(img)
        ws += w
        wm += w * img
    wm = wm / ws
    wm = wm / np.amax(wm) * 255.0
    cv2.imwrite(output_path, wm.astype(np.uint8))


def luminance_mask(lum):
    blk_shape = (8, 8)
    rows = lum.shape[0] // blk_shape[0]
    cols = lum.shape[1] // blk_shape[1]
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            blk = lum[i * blk_shape[0]:i * blk_shape[0] + blk_shape[0],
                      j * blk_shape[1]:j * blk_shape[1] + blk_shape[1]]
            coeffs = cv2.dct(blk)
            mask[i][j] = coeffs[0][0]
    l_min, l_max = 90, 255
    f_max = 2
    mask /= 8
    mean = max(l_min, np.mean(mask))
    f_ref = 1 + (mean - l_min) * (f_max - 1) / (l_max - l_min)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] > mean:
                mask[i][j] = 1 + (mask[i][j] - mean) / (l_max - mean) * (f_max - f_ref)
            elif mask[i][j] < 15:
                mask[i][j] = 1.25
            elif mask[i][j] < 25:
                mask[i][j] = 1.125
            else:
                mask[i][j] = 1
    return mask

def texture_mask(lum):
    blk_shape = (8, 8)
    rows = lum.shape[0] // blk_shape[0]
    cols = lum.shape[1] // blk_shape[1]
    mask = np.full((rows, cols), 1.0)
    for i in range(rows):
        for j in range(cols):
            blk = lum[i * blk_shape[0]:i * blk_shape[0] + blk_shape[0],
                      j * blk_shape[1]:j * blk_shape[1] + blk_shape[1]]
            coeffs = cv2.dct(blk)
            coeffs = np.abs(coeffs)
            dcl = coeffs[0][0] + coeffs[0][1] + coeffs[0][2] + coeffs[1][0] + coeffs[1][1] + coeffs[2][0]
            eh = np.sum(coeffs) - dcl
            if eh > 125:
                e = coeffs[3][0] + coeffs[4][0] + coeffs[5][0] + coeffs[6][0] + \
                    coeffs[0][3] + coeffs[0][4] + coeffs[0][5] + coeffs[0][6] + \
                    coeffs[2][1] + coeffs[1][2] + coeffs[2][2] + coeffs[3][3]
                h = eh - e
                l = dcl - coeffs[0][0]
                a1, b1 = 2.3, 1.6
                a2, b2 = 1.4, 1.1
                l_e, le_h  = l / e, (l + e) / h
                if eh > 900:
                    if (l_e  >= a2 and le_h >= b2) or (l_e >= b2 and le_h >= a2) or le_h > 4:
                        mask[i][j] = 1.125 if l + e <= 400 else 1.25
                    else:
                        mask[i][j] = 1 + 1.25 * (eh - 290) / (1800 - 290)
                else:
                    if (l_e  >= a1 and le_h >= b1) or (l_e >= b1 and le_h >= a1) or le_h > 4:
                        mask[i][j] = 1.125 if l + e <= 400 else 1.25
                    elif e + h > 290:
                        mask[i][j] = 1 + 1.25 * (eh - 290) / (1800 - 290)
    return mask

def compute_diff(vid1, vid2, output_path):
    cap1 = cv2.VideoCapture(vid1)
    cap2 = cv2.VideoCapture(vid2)
    count = 0
    width = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps =  cap1.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    print(width, height)
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))
    while cap1.isOpened() and cap2.isOpened():
        ret, frame1 = cap1.read()
        if ret:
            ret, frame2 = cap2.read()
            if ret:
                count += 1
                frame1 = frame1.astype(np.int32)
                frame2 = frame2.astype(np.int32)
                print("Start processing frame {}".format(count))
                diff = frame1 - frame2
                # print(frame1[:,:,0], frame2[:,:,0], diff[:,:,0])
                # print(np.amin(diff), np.amax(diff))
                idx = np.unravel_index(diff.argmax(), diff.shape)
                print(idx, diff[idx])
                diff = np.abs(diff)
                diff = diff / diff[idx] * 255 
                out.write(diff.astype(np.uint8))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    out.release()
    cap1.release()
    cap2.release()

if __name__ == "__main__":
    from skimage.metrics import structural_similarity as ssim
    img1 = cv2.imread("../examples/pics/frame63.jpeg")
    img2 = cv2.imread("../examples/output/watermarked.jpeg")
    v = ssim(img1, img2, data_range=img2.max() - img2.min(), channel_axis=2)
    print(v)
