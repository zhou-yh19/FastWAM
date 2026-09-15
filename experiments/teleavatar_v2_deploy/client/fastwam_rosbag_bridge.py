#!/usr/bin/env python3
"""Publish FastWAM non-ROS data as ROS2 topics for rosbag2 recording.

Cameras must match run_client / training keys (SBS stereo), NOT single-eye RTP names:
  head_camera, left_color, right_color

RTP itself is not a ROS topic; only these republished JPEGs enter the bag.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy.time
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension, String

# Keys sent to serve_policy /infer (see ros2_interface._POLICY_TO_RTP).
FASTWAM_CAMERA_NAMES: Tuple[str, ...] = (
    "head_camera",
    "left_color",
    "right_color",
)

# image_source(since) -> {camera_name: (rgb, receive_stamp_sec)} for frames strictly
# newer than since[camera_name]. See RTPH265VideoInterface.get_views_if_newer.
ImageSource = Callable[[Dict[str, float]], Dict[str, Tuple[np.ndarray, float]]]


class FastWamBagPublisher:
    """Bridge FastWAM observations / policy outputs into ROS2 for rosbag2."""

    def __init__(
        self,
        node: Node,
        camera_names: Optional[Iterable[str]] = None,
        *,
        max_width: int = 512,
        max_height: int = 384,
        jpeg_quality: int = 90,
    ) -> None:
        self.node = node
        self.camera_names = list(camera_names or FASTWAM_CAMERA_NAMES)
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))

        # Background image publisher. Recording JPEGs is a bag-only concern, so it
        # must not sit on the control loop -- a synchronous publish_images() call
        # there made the recorded frame cadence follow the loop, which stalls for
        # ~130 ms on every replan (see analyze_deploy_bag.py camera_* artifacts).
        self._image_thread: Optional[threading.Thread] = None
        self._image_stop = threading.Event()
        self._image_source: Optional[ImageSource] = None
        self._image_poll_period = 0.005
        self._last_stamp: Dict[str, float] = {}
        self._stats_lock = threading.Lock()
        self._published_count = 0
        self._encode_ms_total = 0.0
        self._encode_fail_count = 0
        self._log_interval_s = 10.0
        self._dropped_count = 0

        # RELIABLE (not BEST_EFFORT): rosbag2_recorder defaults to RELIABLE and will
        # drop the subscription with "incompatible QoS ... RELIABILITY_QOS_POLICY"
        # if the publisher is BEST_EFFORT — images would never enter the bag.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.image_publishers = {
            name: node.create_publisher(
                CompressedImage,
                f"/fastwam/observation/{name}/compressed",
                image_qos,
            )
            for name in self.camera_names
        }
        self.action_publisher = node.create_publisher(
            Float32MultiArray,
            "/fastwam/policy/action_chunk",
            10,
        )
        self.inference_time_publisher = node.create_publisher(
            Float32,
            "/fastwam/policy/inference_ms",
            10,
        )
        self.event_publisher = node.create_publisher(
            String,
            "/fastwam/episode/event",
            10,
        )
        # Predicted mosaic rollout from infer_joint (optional).
        self.pred_video_publisher = node.create_publisher(
            CompressedImage,
            "/fastwam/policy/pred_video/compressed",
            image_qos,
        )
        self.pred_video_meta_publisher = node.create_publisher(
            String,
            "/fastwam/policy/pred_video/meta",
            10,
        )

    @staticmethod
    def _resize_keep_aspect(
        rgb: np.ndarray, max_width: int, max_height: int
    ) -> np.ndarray:
        """Downscale SBS frames to fit in max_w x max_h (no upscale, keep aspect).

        Training mosaic is 384x512; forcing square 384x384 would distort SBS.
        """
        h, w = rgb.shape[:2]
        if h <= 0 or w <= 0:
            return rgb
        scale = min(max_width / float(w), max_height / float(h), 1.0)
        if scale >= 0.999:
            return rgb
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _stamp_from_epoch(stamp_sec: float) -> Any:
        """Epoch seconds -> builtin_interfaces/Time, split to keep ns precision.

        stamp_sec * 1e9 would exceed float64's exact-integer range, so seconds and
        nanoseconds are carried separately. All views split from one decoded frame
        share a single epoch value, and this mapping is deterministic, so they keep
        landing on one identical stamp -- export_bag_for_offline.py groups the three
        cameras into an observation by exact stamp equality.
        """
        sec = int(stamp_sec)
        nanosec = int(round((stamp_sec - sec) * 1e9))
        if nanosec >= 1_000_000_000:  # rounding can carry into the next second
            sec += 1
            nanosec -= 1_000_000_000
        return rclpy.time.Time(seconds=sec, nanoseconds=nanosec).to_msg()

    def _encode_and_publish(
        self,
        camera_name: str,
        rgb: np.ndarray,
        stamp: Any,
    ) -> bool:
        """Resize + JPEG-encode one RGB frame and publish it. Returns success."""
        publisher = self.image_publishers.get(camera_name)
        if publisher is None:
            return False
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
            self.node.get_logger().warning(
                f"Invalid image for {camera_name}: "
                f"type={type(rgb)}, shape={getattr(rgb, 'shape', None)}"
            )
            return False

        t_encode = time.perf_counter()
        resized = self._resize_keep_aspect(
            np.asarray(rgb), self.max_width, self.max_height
        )
        # RTP path yields RGB; OpenCV JPEG expects BGR.
        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        encode_ms = (time.perf_counter() - t_encode) * 1000.0
        if not ok:
            with self._stats_lock:
                self._encode_fail_count += 1
            self.node.get_logger().warning(f"JPEG encode failed for {camera_name}")
            return False

        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = camera_name
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        publisher.publish(msg)
        with self._stats_lock:
            self._published_count += 1
            self._encode_ms_total += encode_ms
        return True

    def publish_images(self, images: dict) -> None:
        """Publish RGB frames synchronously, stamped now.

        Only for callers with no image_source to poll; the deploy client uses
        start_image_publisher() instead so encoding stays off the control loop.
        """
        stamp = self.node.get_clock().now().to_msg()
        for camera_name in self.image_publishers:
            rgb = images.get(camera_name)
            if rgb is not None:
                self._encode_and_publish(camera_name, rgb, stamp)

    def start_image_publisher(
        self,
        image_source: ImageSource,
        *,
        poll_hz: float = 200.0,
    ) -> None:
        """Republish decoded camera frames from a background thread.

        Polls `image_source` for frames newer than the last one published per
        camera, so each decoded frame is recorded exactly once at the rate the
        decoder actually delivers -- independent of the control loop's cadence.

        Poll rate only bounds the timestamping delay, not the work: encoding runs
        once per new frame. OpenCV releases the GIL during resize/cvtColor/imencode,
        so this genuinely offloads the control loop rather than just deferring it.
        """
        if self._image_thread is not None:
            return
        self._image_source = image_source
        self._image_poll_period = 1.0 / max(float(poll_hz), 1.0)
        self._image_stop.clear()
        self._image_thread = threading.Thread(
            target=self._image_loop,
            name="fastwam-bag-images",
            daemon=True,
        )
        self._image_thread.start()
        self.node.get_logger().info(
            f"bag image publisher started (poll {poll_hz:.0f} Hz, "
            f"{len(self.image_publishers)} cameras)"
        )

    def stop_image_publisher(self, timeout: float = 2.0) -> None:
        thread = self._image_thread
        if thread is None:
            return
        self._image_stop.set()
        thread.join(timeout=timeout)
        self._image_thread = None
        stats = self.image_publisher_stats()
        self.node.get_logger().info(
            f"bag image publisher stopped: published={stats['published']:.0f} "
            f"mean_encode_ms={stats['mean_encode_ms']:.2f} "
            f"dropped~{stats['dropped']:.0f} "
            f"encode_failures={stats['encode_failures']:.0f}"
        )

    def image_publisher_stats(self) -> Dict[str, float]:
        with self._stats_lock:
            count = self._published_count
            total = self._encode_ms_total
            fails = self._encode_fail_count
            dropped = self._dropped_count
        return {
            "published": float(count),
            "mean_encode_ms": (total / count) if count else 0.0,
            "encode_failures": float(fails),
            "dropped": float(dropped),
        }

    def _image_loop(self) -> None:
        source = self._image_source
        if source is None:
            return
        # Reference camera for cadence/drop accounting; all views of one decoded
        # frame carry the same stamp, so one is enough.
        ref_camera = self.camera_names[0] if self.camera_names else None
        intervals: List[float] = []
        last_ref_stamp: Optional[float] = None
        window_start = time.monotonic()
        window_frames = 0
        window_encode_ms = 0.0

        while not self._image_stop.is_set():
            try:
                fresh = source(dict(self._last_stamp))
            except Exception:
                self.node.get_logger().exception("bag image source failed")
                self._image_stop.wait(0.1)
                continue

            for camera_name, (rgb, stamp_sec) in fresh.items():
                # Stamp with the decoder's receive time, not the publish time, so
                # the bag records when each frame was captured.
                stamp = self._stamp_from_epoch(stamp_sec)
                t_encode = time.perf_counter()
                if not self._encode_and_publish(camera_name, rgb, stamp):
                    continue
                self._last_stamp[camera_name] = stamp_sec
                window_encode_ms += (time.perf_counter() - t_encode) * 1000.0

                if camera_name != ref_camera:
                    continue
                window_frames += 1
                if last_ref_stamp is not None:
                    gap = stamp_sec - last_ref_stamp
                    # A gap much wider than the usual one means the encoder could
                    # not keep up and get_views_if_newer skipped to a later frame.
                    if len(intervals) >= 20:
                        typical = float(np.median(intervals))
                        if typical > 0 and gap > 1.5 * typical:
                            with self._stats_lock:
                                self._dropped_count += int(round(gap / typical)) - 1
                    intervals.append(gap)
                    if len(intervals) > 200:
                        del intervals[:-200]
                last_ref_stamp = stamp_sec

            now = time.monotonic()
            if now - window_start >= self._log_interval_s:
                self._log_window(now - window_start, window_frames, window_encode_ms)
                window_start, window_frames, window_encode_ms = now, 0, 0.0

            self._image_stop.wait(self._image_poll_period)

    def _log_window(self, elapsed: float, frames: int, encode_ms: float) -> None:
        """Report achieved republish rate so a deploy run shows coupling at a glance."""
        if elapsed <= 0:
            return
        hz = frames / elapsed
        per_frame = (encode_ms / frames) if frames else 0.0
        with self._stats_lock:
            dropped = self._dropped_count
        msg = (
            f"bag images: {hz:.2f} Hz ({frames} frames/{elapsed:.1f}s), "
            f"encode {per_frame:.1f} ms/frame-set, dropped~{dropped}"
        )
        # Encoding must stay well inside one frame period or frames get skipped.
        if frames and per_frame > 0.6 * (1000.0 / max(hz, 1e-6)):
            self.node.get_logger().warning(
                f"{msg} -- encode is near the frame period; "
                f"lower jpeg_quality or max_width to stop dropping frames"
            )
        else:
            self.node.get_logger().info(msg)

    def publish_action_chunk(self, action_chunk: np.ndarray) -> None:
        array = np.asarray(action_chunk, dtype=np.float32)
        msg = Float32MultiArray()
        msg.data = array.reshape(-1).tolist()
        for index, size in enumerate(array.shape):
            dim = MultiArrayDimension()
            dim.label = f"dim_{index}"
            dim.size = int(size)
            dim.stride = int(np.prod(array.shape[index:]))
            msg.layout.dim.append(dim)
        self.action_publisher.publish(msg)

    def publish_inference_time(self, inference_ms: float) -> None:
        msg = Float32()
        msg.data = float(inference_ms)
        self.inference_time_publisher.publish(msg)

    def publish_event(self, event: str) -> None:
        msg = String()
        msg.data = str(event)
        self.event_publisher.publish(msg)

    def publish_pred_video(
        self,
        frames: Sequence[np.ndarray],
        *,
        infer_id: int = -1,
        meta: Optional[str] = None,
    ) -> None:
        """Publish predicted mosaic frames (RGB HWC uint8) as CompressedImage sequence.

        Topic: /fastwam/policy/pred_video/compressed
        frame_id: pred_{infer_id:06d}_t{t:02d}
        All frames in one rollout share the same ROS stamp (observation time approx).
        """
        if not frames:
            return
        stamp = self.node.get_clock().now().to_msg()
        n = len(frames)
        if meta is None:
            meta = f"infer_id={infer_id};num_frames={n}"
        meta_msg = String()
        meta_msg.data = meta
        self.pred_video_meta_publisher.publish(meta_msg)

        for t, rgb in enumerate(frames):
            if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
                continue
            arr = np.asarray(rgb)
            # Pred video is already ~384x512; still clamp if huge.
            arr = self._resize_keep_aspect(arr, self.max_width, self.max_height)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(
                ".jpg",
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                continue
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.header.frame_id = f"pred_{max(infer_id, 0):06d}_t{t:02d}"
            msg.format = "jpeg"
            msg.data = encoded.tobytes()
            self.pred_video_publisher.publish(msg)

    def topic_list_for_bag(self) -> Sequence[str]:
        cams = [
            f"/fastwam/observation/{name}/compressed" for name in self.camera_names
        ]
        return [
            *cams,
            "/fastwam/policy/action_chunk",
            "/fastwam/policy/inference_ms",
            "/fastwam/policy/pred_video/compressed",
            "/fastwam/policy/pred_video/meta",
            "/fastwam/episode/event",
        ]
