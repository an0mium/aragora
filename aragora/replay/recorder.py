import logging
import os
import queue
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from aragora.config import MAX_REPLAY_QUEUE_SIZE
from .schema import ReplayEvent, ReplayMeta

logger = logging.getLogger(__name__)

class ReplayRecorder:
    """Non-blocking append-only recorder with background writer."""
    
    def __init__(
        self,
        debate_id: str,
        topic: str,
        proposal: str,
        agents: List[Dict[str, str]],
        storage_dir: str = ".nomic/replays"
    ):
        self.debate_id = debate_id
        self.storage_dir = Path(storage_dir)
        self._start_time: Optional[float] = None
        self._event_count = 0
        self._is_active = False
        
        self.session_dir = self.storage_dir / debate_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.meta_path = self.session_dir / "meta.json"
        self.events_path = self.session_dir / "events.jsonl"
        
        self.meta = ReplayMeta(
            debate_id=debate_id,
            topic=topic,
            proposal=proposal,
            agents=agents,
            started_at=datetime.utcnow().isoformat()
        )
        
        self._write_queue: queue.Queue = queue.Queue(maxsize=MAX_REPLAY_QUEUE_SIZE)
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_writer = threading.Event()
        self._event_count_lock = threading.Lock()
    
    def start(self) -> None:
        self._start_time = time.time()
        self._is_active = True
        self._write_meta()
        self._stop_writer.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
    
    def _writer_loop(self) -> None:
        with open(self.events_path, 'a', encoding='utf-8') as f:
            while not self._stop_writer.is_set():
                try:
                    event = self._write_queue.get(timeout=0.1)
                    f.write(event.to_jsonl() + '\n')
                    f.flush()
                    with self._event_count_lock:
                        self._event_count += 1
                except queue.Empty:
                    continue
    
    def _write_meta(self) -> None:
        try:
            with self._event_count_lock:
                self.meta.event_count = self._event_count
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                f.write(self.meta.to_json())
        except (IOError, OSError) as e:
            logger.warning(f"Failed to write replay metadata to {self.meta_path}: {e}")
    
    def _elapsed_ms(self) -> int:
        return int((time.time() - (self._start_time or time.time())) * 1000)
    
    def _record(self, event_type: str, source: str, content: str, metadata: Dict[str, Any] | None = None) -> None:
        if not self._is_active:
            return
        try:
            event = ReplayEvent(
                event_id=str(uuid.uuid4())[:8],
                timestamp=time.time(),
                offset_ms=self._elapsed_ms(),
                event_type=event_type,
                source=source,
                content=content,
                metadata=metadata or {}
            )
            self._write_queue.put_nowait(event)
        except queue.Full:
            logger.warning(f"Replay queue full, dropping {event_type} event")
    
    def record_turn(self, agent_id: str, content: str, round_num: int, loop_id: Optional[str] = None) -> None:
        meta: Dict[str, Any] = {"round": round_num}
        if loop_id:
            meta["loop_id"] = loop_id
        self._record("turn", agent_id, content, meta)
    
    def record_vote(self, agent_id: str, vote: str, reasoning: str) -> None:
        self._record("vote", agent_id, vote, {"reasoning": reasoning})
    
    def record_audience_input(self, user_id: str, message: str, loop_id: Optional[str] = None) -> None:
        meta = {"user_id": user_id}
        if loop_id:
            meta["loop_id"] = loop_id
        self._record("audience_input", user_id, message, meta)
    
    def record_phase_change(self, phase: str) -> None:
        self._record("phase_change", "system", phase)
    
    def record_system(self, message: str) -> None:
        self._record("system", "system", message)
    
    def finalize(self, verdict: str, votes: Dict[str, int]) -> str:
        self._is_active = False
        self._stop_writer.set()
        if self._writer_thread:
            self._writer_thread.join(timeout=10.0)
            if self._writer_thread.is_alive():
                logger.warning(f"Replay writer thread didn't stop in 10s, queue depth: {self._write_queue.qsize()}")
        self.meta.status = "completed"
        self.meta.ended_at = datetime.utcnow().isoformat()
        self.meta.duration_ms = self._elapsed_ms()
        self.meta.final_verdict = verdict
        self.meta.vote_tally = votes
        self._write_meta()
        return str(self.session_dir)
    
    def abort(self) -> None:
        self._is_active = False
        self._stop_writer.set()
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)
            if self._writer_thread.is_alive():
                logger.warning("Replay writer abort: thread didn't stop in 5s")
        self.meta.status = "crashed"
        self._write_meta()