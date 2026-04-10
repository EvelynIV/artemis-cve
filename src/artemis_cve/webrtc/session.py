from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from artemis_cve.inferencers.smoothers import BoxDetectionSmoother
from artemis_cve.inferencers.yolo import BoxDetection, SharedYoloBoxInferencer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingFrame:
    frame_id: int
    pts_ms: int
    image: object


class WebRtcSession:
    def __init__(
        self,
        stream_id: str,
        inferencer: SharedYoloBoxInferencer,
        score_threshold: float,
        max_detections: int | None,
    ) -> None:
        self.stream_id = stream_id
        self.pc = RTCPeerConnection()
        self.inferencer = inferencer
        self.score_threshold = float(score_threshold)
        self.max_detections = max_detections
        self.running = True
        self._video_tasks: set[asyncio.Task] = set()
        self._pending_frame: PendingFrame | None = None
        self._frame_ready = asyncio.Condition()
        # Favor responsiveness over aggressive smoothing in live streams.
        self._smoother = BoxDetectionSmoother(alpha=1.0, match_iou_threshold=0.3)
        self.detection_queues: list[asyncio.Queue] = []

        self.pc.addTransceiver("video", direction="recvonly")
        self.pc.on("track", self._on_track)

    async def create_offer(self) -> RTCSessionDescription:
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)
        return self.pc.localDescription

    async def set_answer(self, sdp: str, sdp_type: str) -> None:
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
        logger.info("Answer set for stream %s", self.stream_id)

    def attach_detection_queue(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=60)
        self.detection_queues.append(queue)
        return queue

    def detach_detection_queue(self, queue: asyncio.Queue) -> None:
        if queue in self.detection_queues:
            self.detection_queues.remove(queue)

    def _on_track(self, track: MediaStreamTrack) -> None:
        if track.kind != "video":
            return
        logger.info("Video track received for stream %s", self.stream_id)
        task = asyncio.create_task(self._process_video(track))
        self._video_tasks.add(task)
        task.add_done_callback(self._on_video_task_done)

    def _on_video_task_done(self, task: asyncio.Task) -> None:
        self._video_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.exception(
                "Video processing task failed for stream %s",
                self.stream_id,
                exc_info=exc,
            )

    async def _process_video(self, track: MediaStreamTrack) -> None:
        receiver_task = asyncio.create_task(self._receive_frames(track))
        processor_task = asyncio.create_task(self._run_inference_loop())
        try:
            await asyncio.gather(receiver_task, processor_task)
        except Exception:
            logger.exception("Unhandled error while processing stream %s", self.stream_id)
            raise
        finally:
            self.running = False
            receiver_task.cancel()
            processor_task.cancel()
            self._smoother.reset()
            async with self._frame_ready:
                self._frame_ready.notify_all()

    async def _receive_frames(self, track: MediaStreamTrack) -> None:
        while self.running:
            try:
                frame = await track.recv()
            except Exception:
                logger.info("Track ended for stream %s", self.stream_id)
                break

            image = frame.to_ndarray(format="bgr24")
            pts_ms = self._frame_pts_ms(frame)
            frame_id = pts_ms

            pending = PendingFrame(frame_id=frame_id, pts_ms=pts_ms, image=image)
            async with self._frame_ready:
                self._pending_frame = pending
                self._frame_ready.notify()

        self.running = False
        async with self._frame_ready:
            self._frame_ready.notify_all()

    async def _run_inference_loop(self) -> None:
        while self.running:
            async with self._frame_ready:
                while self.running and self._pending_frame is None:
                    await self._frame_ready.wait()
                if not self.running and self._pending_frame is None:
                    return
                pending = self._pending_frame
                self._pending_frame = None

            if pending is None:
                continue

            detections = await asyncio.to_thread(
                self.inferencer.infer,
                pending.image,
                self.score_threshold,
                self.max_detections,
            )
            smoothed = self._smoother.smooth(detections)
            self._push_detection(
                frame_id=pending.frame_id,
                pts_ms=pending.pts_ms,
                detections=smoothed,
            )

    @staticmethod
    def _frame_pts_ms(frame) -> int:
        if frame.pts is None or frame.time_base is None:
            return 0
        return int(round(float(frame.pts * frame.time_base) * 1000.0))

    def _push_detection(
        self,
        frame_id: int,
        pts_ms: int,
        detections: list[BoxDetection],
    ) -> None:
        payload = (self.stream_id, str(frame_id), frame_id, pts_ms, detections)
        for queue in self.detection_queues:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    async def close(self) -> None:
        self.running = False
        self._smoother.reset()
        for task in list(self._video_tasks):
            task.cancel()
        await self.pc.close()
