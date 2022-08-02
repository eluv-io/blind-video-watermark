import blind_video_watermark as bvw

if __name__ == "__main__":
    # Watermark image by key
    key = 1
    input_path = "pics/frame63.jpeg"
    output_path = "output/watermarked.jpeg"
    bvw.DtcwtKeyEncoder().embed(1, input_path, output_path)

    # Detect which key the image is watermarked by
    keys = [0, 1, 2, 3, 4]
    idx = bvw.DtcwtKeyDecoder().detect([0, 1, 2, 3, 4], output_path)
    print("Image watermarked by key: ", keys[idx])

    # Watermark video by keys following the sequence
    keys = [10, 11, 12, 13]
    seq = "0231"
    video_path = "videos/bbb-short.mp4"
    output_path = "output/output.mp4"
    bvw.DtcwtKeyEncoder().embed_video_async(keys, seq, 1, video_path, output_path, threads=8)

    # Detect watermarked sequence
    seq = bvw.DtcwtKeyDecoder().detect_video(keys, 1, output_path)
    print("Decoded Sequence:", seq)
