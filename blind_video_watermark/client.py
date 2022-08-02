import argparse
import multiprocessing

import blind_video_watermark as bvw
# from .dtcwt_img import DtcwtImgEncoder, DtcwtImgDecoder
# from .dtcwt_key import DtcwtKeyEncoder, DtcwtKeyDecoder

usage = 'bvw {embed,extract} -a {img,seq} [options] -i INPUT output'

'bvw embed -a img -i input.mp4 -k 123 -s 1.5 -wm wm.jpg OUTPUT'
'bvw embed -a seq -i input.mp4 -k 123 456 -s 7 -d 1 -wm 0110 OUTPUT'
'bvw extract -a img --key 123 -s 1.5 -i input.mp4 -o'
'bvw extract -a seq --key 123 456 -s 7 -d 1 -i input.mp4 -o'
'../examples/wms/wm.jpg'

def main():
	parser = argparse.ArgumentParser(prog='bvw', usage=usage, description="Blind Video Watermarking in DT CWT Domain")
	parser.add_argument("mode", type=str, choices=['embed', 'extract'], help="Set watermarking mode (embed/extract)")
	parser.add_argument("-i", dest="input", type=str, help="Set input video", required=True)
	parser.add_argument("-a", dest="algo", type=str, choices=['img', 'seq'], help="Set algorithm (img/seq)", required=True)
	parser.add_argument("-wm", dest="wm", type=str, help="Set watermark", required=True)
	parser.add_argument("-k", dest="keys", type=int, nargs='*', help="Set keys", required=True)
	parser.add_argument("-str", dest="str", type=float, default=1.0, help="Set watermarking strength")
	parser.add_argument("-l", dest="len_segment", type=float, default=1.0, help="Set length of each segment")
	parser.add_argument("-j", dest="threads", type=int, default=multiprocessing.cpu_count(), help="Set the number of threads to use")
	parser.add_argument("-size", dest="size", type=str, default="1080:1920", help="Set original frame size for extraction (e.g. 1080:1920)")
	parser.add_argument("output", help="Set output path")

	args = parser.parse_args("embed -a seq -i ../examples/videos/bbb-short.mp4 -k 123 456 -j 8 -wm 0101 output.mp4".split())
	print(args.str)

	if args.mode == "embed":
		if args.algo == "img":
			encoder = bvw.DtcwtImgEncoder(key=args.keys[0], str=args.str)
			encoder.embed_video_async(args.wm, args.input, args.output, threads=args.threads)
		elif args.algo == "seq":
			encoder = bvw.DtcwtKeyEncoder(str=args.str)
			encoder.embed_video_async(args.keys, args.wm, args.len_segment, args.input, args.output, threads=args.threads)
	elif args.mode == "extract":
		size = args.size.split(":")
		size = (int(size[0]), int(size[1]))
		if args.algo == "img":
			decoder = bvw.DtcwtImgDecoder(key=args.keys[0], str=args.str, ori_frame_size=size)
			decoder.extract_video(args.input, args.output, args.size)
		elif args.algo == "seq":
			decoder = bvw.DtcwtKeyDecoder(str=args.str)
			decoder.detect_video(args.keys, args.len_segment, args.input, ori_frame_size=size)
	else:
		print("Invalid Mode")

if __name__ == "__main__":
	main()
