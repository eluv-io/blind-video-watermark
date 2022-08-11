import blind_video_watermark as bvw

if __name__ == "__main__":
	# Watermark an image
    wm_path = "wms/wm.jpg"
    img_path = "pics/frame63.jpeg"
    output_path = "output/output.jpeg"
    extracted_path = "output/extracted_watermark.jpeg"
    bvw.DtcwtImgEncoder().embed(wm_path, img_path, output_path)
    bvw.DtcwtImgDecoder().extract(output_path, extracted_path)

    # Watermark a video
    wm_path = "wms/wm.jpg"
    video_path = "videos/bbb-short.mp4"
    output_path = "output/q.mp4"
    extracted_folder = "output/extracted"

    # # Use this if you want to keep the watermarked frame in memory
    # # Check the source code of `embed` for details
    # bvw.DtcwtImgEncoder().encode(IMG_IN_YUV)

    # # Use this if you don't want multiprocessing
    # bvw.DtcwtImgEncoder().embed_video(wm_path, video_path, output_path)

    # Embedding with 8 processes
    # bvw.DtcwtImgEncoder().embed_video_async(wm_path, video_path, output_path, threads=8)

    # Extract watermark frame by frame and save extractions in a folder
    bvw.DtcwtImgDecoder(key=789).extract_video(output_path, extracted_folder)
