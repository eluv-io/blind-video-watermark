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

    # # Use this if you want to keep the watermarked frame in memory
    # # Check the source code of `embed` for details
    # bvw.DtcwtKeyEncoder().encode(IMG_IN_YUV)

    # The video is divided into segments of frag_length.
    # Each segment is watermarked by a key from the keys list chosen by the number in seq
    # 0231 -> [0, 2, 3, 1] -> [10, 12, 13, 11]
    keys = [10, 11, 12, 13]
    seq = "0231"
    frag_length = 1
    video_path = "videos/bbb-short.mp4"
    output_path = "output/output.mp4"

    # # Use this if you don't want multiprocessing
    # bvw.DtcwtKeyEncoder().embed_video(keys, seq, frag_length, video_path, output_path)

    bvw.DtcwtKeyEncoder().embed_video_async(keys, seq, frag_length, video_path, output_path, threads=8)

    # Detect watermarked sequence
    seq = bvw.DtcwtKeyDecoder().detect_video_async(keys, frag_length, output_path, threads=8)
    print("Decoded Sequence:", seq)
