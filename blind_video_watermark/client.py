import argparse
import multiprocessing

from .dtcwt_img import DtcwtImgEncoder, DtcwtImgDecoder
from .dtcwt_key import DtcwtKeyEncoder, DtcwtKeyDecoder

usage = 'bvw {embed,extract} -a {img,seq} [options] -i INPUT -o OUTPUT'

def main():
	parser = argparse.ArgumentParser(prog='bvw', usage=usage, description="Blind Video Watermarking in DT CWT Domain")
	parser.add_argument("mode", type=str, choices=['embed', 'extract'], help="Set watermarking mode (embed/extract)")
	parser.add_argument("-i", dest="input", type=str, help="Set input video", required=True)
	parser.add_argument("-a", dest="algo", type=str, choices=['img', 'seq'], help="Set algorithm (img/seq)", required=True)
	parser.add_argument("-k", dest="keys", type=int, nargs='*', help="Set keys", required=True)
	parser.add_argument("-wm", dest="wm", type=str, help="Set watermark")
	parser.add_argument("-str", dest="str", type=float, default=1.0, help="Set watermarking strength")
	parser.add_argument("-l", dest="len_segment", type=float, default=1.0, help="Set length of each segment")
	parser.add_argument("-j", dest="threads", type=int, default=multiprocessing.cpu_count(), help="Set the number of threads to use")
	parser.add_argument("-size", dest="size", type=str, default="1080:1920", help="Set original frame size for extraction (e.g. 1080:1920)")
	parser.add_argument("-o", dest="output", type=str, help="Set output path", required=True)

	args = parser.parse_args()

	if args.mode == "embed":
		if args.wm == None:
			parser.error("-wm is required for embedding")
		if args.algo == "img":
			encoder = DtcwtImgEncoder(key=args.keys[0], str=args.str)
			encoder.embed_video_async(args.wm, args.input, args.output, threads=args.threads)
		elif args.algo == "seq":
			encoder = DtcwtKeyEncoder(str=args.str)
			encoder.embed_video_async(args.keys, args.wm, args.len_segment, args.input, args.output, threads=args.threads)
	elif args.mode == "extract":
		size = args.size.split(":")
		size = (int(size[0]), int(size[1]))
		if args.algo == "img":
			decoder = DtcwtImgDecoder(key=args.keys[0], str=args.str)
			decoder.extract_video(args.input, args.output, ori_frame_size=size)
		elif args.algo == "seq":
			decoder = DtcwtKeyDecoder(str=args.str)
			seq = decoder.detect_video(args.keys, args.len_segment, args.input, ori_frame_size=size)
			print("Decoded Sequence:", seq)
			with open(args.output, 'w') as fd:
				fd.write(str(seq))
	else:
		print("Invalid Mode")
