#!/usr/bin/env python3
"""Receive webcam frames and person masks, inpaint, display, and record the video."""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = sock.recv(size - len(chunks))
        if not part:
            if not chunks:
                return b""
            raise ConnectionError("Stream ended in the middle of a packet")
        chunks.extend(part)
    return bytes(chunks)


def receive_packet(sock: socket.socket):
    prefix = recv_exact(sock, 4)
    if not prefix:
        return None
    header_size = struct.unpack("!I", prefix)[0]
    if header_size <= 0 or header_size > 64 * 1024:
        raise ValueError(f"Invalid packet header size: {header_size}")
    header = json.loads(recv_exact(sock, header_size).decode("utf-8"))
    frame_size, mask_size = int(header["frame_bytes"]), int(header["mask_bytes"])
    if frame_size <= 0 or mask_size <= 0 or frame_size + mask_size > 32 * 1024 * 1024:
        raise ValueError("Invalid image payload sizes")
    frame_data = recv_exact(sock, frame_size)
    mask_data = recv_exact(sock, mask_size)
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if frame is None or mask is None:
        raise ValueError("Could not decode received frame or mask")
    return header, frame, mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0", help="Listen address")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--output", default="received_inpainted.mp4")
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS")
    parser.add_argument("--radius", type=float, default=5.0, help="OpenCV inpaint radius")
    parser.add_argument("--dilate", type=int, default=5, help="Expand mask by this many pixels")
    parser.add_argument("--no-display", action="store_true", help="Run without a display")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total = 0
    start = time.perf_counter()
    kernel_size = max(1, args.dilate)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.port))
        server.listen(1)
        print(f"Listening on {args.bind}:{args.port}; output: {output}", flush=True)
        conn, peer = server.accept()
        print(f"Connected from {peer[0]}:{peer[1]}", flush=True)
        with conn:
            while True:
                packet = receive_packet(conn)
                if packet is None:
                    break
                header, frame, mask = packet
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                binary = cv2.dilate(binary, kernel, iterations=1)
                result = cv2.inpaint(frame, binary, args.radius, cv2.INPAINT_TELEA)

                if writer is None:
                    height, width = result.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create output video: {output}")
                writer.write(result)
                total += 1
                if not args.no_display:
                    cv2.imshow("Person removed (OpenCV inpaint)", result)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if total % 30 == 0:
                    elapsed = max(time.perf_counter() - start, 1e-6)
                    print(
                        f"Received {total} frames; seq={header['seq']}; "
                        f"receive rate={total / elapsed:.1f} FPS",
                        flush=True,
                    )

    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"Saved {total} frames to {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
