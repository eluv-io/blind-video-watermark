import time
import blind_video_watermark as bvw

if __name__ == "__main__":
	# Watermark an image
    wm_path = "wms/wm.jpg"
    img_path = "pics/frame63.jpeg"
    output_path = "output/watermarked.jpeg"
    extracted_path = "output/extracted_watermark.jpeg"
    bvw.DtcwtImgEncoder().embed(wm_path, img_path, output_path)
    bvw.DtcwtImgDecoder().extract(output_path, extracted_path)

    # Watermark a video
    wm_path = "wms/wm.jpg"
    video_path = "videos/bbb-short.mp4"
    output_path = "output/output.mp4"
    extracted_path = "output/extracted"
    start = time.time()
    bvw.DtcwtImgEncoder().embed_video_async(wm_path, video_path, output_path, threads=8)
    print(time.time() - start)
    bvw.DtcwtImgDecoder().extract_video(output_path, extracted_path)
