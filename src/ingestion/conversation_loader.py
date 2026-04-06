"""
Conversation data loader for Digital Self RAG.

Loads exported chat conversations (JSON format) and converts them into
standard Chunk objects compatible with the existing RAG pipeline.

Supported input format:
    [
        {"role": "user",      "content": "...", "timestamp": "2024-01-15T10:30:00"},
        {"role": "assistant", "content": "...", "timestamp": "2024-01-15T10:30:05"},
        ...
    ]

Chunking strategies:
    - turn_group: Group N consecutive turns into one chunk (preserves dialogue flow)
    - whole: Entire conversation as a single chunk (for short conversations)
    - decision_point: Split at topic/decision boundaries using role-transition heuristics
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

from .chunker import Chunk

logger = logging.getLogger(__name__)


class ConversationChunkStrategy(Enum):
    """Chunking strategies for conversation data."""

    TURN_GROUP = "turn_group"
    WHOLE = "whole"
    DECISION_POINT = "decision_point"


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str
    content: str
    timestamp: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Conversation:
    """A full conversation with metadata."""

    turns: List[ConversationTurn]
    source: str = "unknown"
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Full conversation as formatted text."""
        lines = []
        for turn in self.turns:
            prefix = f"[{turn.role}]"
            lines.append(f"{prefix} {turn.content}")
        return "\n\n".join(lines)

    @property
    def turn_count(self) -> int:
        return len(self.turns)


class ConversationLoader:
    """
    Loads conversation JSON files and converts them into Chunk objects
    compatible with the standard RAG pipeline.

    The key design choice: we convert conversations into the same Chunk
    dataclass used by TextChunker, so the rest of the pipeline
    (EmbeddingEngine.embed_chunks → FAISSVectorStore.add) works unchanged.

    Conversation-specific metadata (roles, timestamps, topic) is stored
    in Chunk.metadata for downstream use (e.g., behavioral HyDE prompting).

    Usage:
        loader = ConversationLoader(strategy="turn_group", turns_per_chunk=4)
        chunks = loader.load_and_chunk("conversations/chat_export.json")
        # chunks: List[Chunk] — ready for embed_chunks() + store.add()
    """

    def __init__(
        self,
        strategy: str = "turn_group",
        turns_per_chunk: int = 4,
        overlap_turns: int = 1,
        min_chunk_length: int = 50,
    ):
        """
        Args:
            strategy: Chunking strategy — turn_group | whole | decision_point.
            turns_per_chunk: Number of turns per chunk (turn_group strategy).
            overlap_turns: Overlapping turns between chunks (turn_group strategy).
            min_chunk_length: Minimum character length for a chunk to be kept.
        """
        try:
            self.strategy = ConversationChunkStrategy(strategy)
        except ValueError:
            logger.warning(
                f"Unknown conversation chunking strategy: {strategy}. "
                "Falling back to 'turn_group'."
            )
            self.strategy = ConversationChunkStrategy.TURN_GROUP

        self.turns_per_chunk = max(1, turns_per_chunk)
        self.overlap_turns = max(0, min(overlap_turns, turns_per_chunk - 1))
        self.min_chunk_length = min_chunk_length

    def load_file(self, file_path: str) -> Conversation:
        """
        Load a single conversation JSON file.

        Expected format: list of turn objects with "role" and "content" keys.
        Optional keys: "timestamp", plus any additional metadata.

        Args:
            file_path: Path to a JSON file.

        Returns:
            Conversation object.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file isn't valid JSON.
            ValueError: If the data format is invalid.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Conversation file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._parse_conversation(data, source=path.name)

    def load_directory(self, dir_path: str) -> List[Conversation]:
        """
        Load all conversation JSON files from a directory.

        Args:
            dir_path: Path to a directory containing JSON files.

        Returns:
            List of Conversation objects.
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        conversations = []
        for json_file in sorted(path.glob("*.json")):
            try:
                conv = self.load_file(str(json_file))
                conversations.append(conv)
                logger.info(f"Loaded {json_file.name}: {conv.turn_count} turns")
            except Exception as e:
                logger.warning(f"Failed to load {json_file.name}: {e}")

        logger.info(f"Loaded {len(conversations)} conversations from {dir_path}")
        return conversations

    def load_and_chunk(self, file_path: str) -> List[Chunk]:
        """
        Load a conversation file and chunk it in one step.

        Args:
            file_path: Path to a JSON file.

        Returns:
            List of Chunk objects ready for the embedding pipeline.
        """
        conv = self.load_file(file_path)
        return self.chunk_conversation(conv)

    def chunk_conversations(self, conversations: List[Conversation]) -> List[Chunk]:
        """
        Chunk multiple conversations, assigning globally unique chunk IDs.

        Args:
            conversations: List of Conversation objects.

        Returns:
            List of Chunk objects with unique chunk_id values.
        """
        all_chunks = []
        global_id = 0

        for conv in conversations:
            chunks = self.chunk_conversation(conv, start_id=global_id)
            all_chunks.extend(chunks)
            global_id += len(chunks)

        logger.info(
            f"Chunked {len(conversations)} conversations → " f"{len(all_chunks)} chunks"
        )
        return all_chunks

    def chunk_conversation(
        self, conversation: Conversation, start_id: int = 0
    ) -> List[Chunk]:
        """
        Split a conversation into Chunk objects using the configured strategy.

        Each chunk's metadata includes:
            - data_type: "conversation"
            - roles: list of roles in the chunk
            - turn_indices: [start, end) range of turn indices
            - first_timestamp / last_timestamp: time range (if available)
            - source_conversation: original file name

        Args:
            conversation: Conversation to chunk.
            start_id: Starting chunk_id for unique identification.

        Returns:
            List of Chunk objects.
        """
        if self.strategy == ConversationChunkStrategy.WHOLE:
            raw_chunks = self._chunk_whole(conversation)
        elif self.strategy == ConversationChunkStrategy.DECISION_POINT:
            raw_chunks = self._chunk_decision_point(conversation)
        else:
            raw_chunks = self._chunk_turn_group(conversation)

        # Convert to Chunk objects
        chunks = []
        char_offset = 0
        full_text = conversation.text

        for i, (text, meta) in enumerate(raw_chunks):
            if len(text.strip()) < self.min_chunk_length:
                continue

            # Calculate approximate character positions in full conversation
            start_char = full_text.find(text[:50]) if len(text) >= 50 else char_offset
            if start_char < 0:
                start_char = char_offset
            end_char = start_char + len(text)

            chunk_meta = {
                "data_type": "conversation",
                "source_conversation": conversation.source,
                **conversation.metadata,
                **meta,
            }

            chunks.append(
                Chunk(
                    text=text,
                    chunk_id=start_id + len(chunks),
                    source=conversation.source,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=chunk_meta,
                )
            )
            char_offset = end_char

        return chunks

    # ── Chunking strategies ──────────────────────────────────────

    def _chunk_turn_group(self, conversation: Conversation) -> List[tuple]:
        """
        Group consecutive turns into chunks with optional overlap.

        Returns list of (text, metadata) tuples.
        """
        turns = conversation.turns
        if not turns:
            return []

        chunks = []
        step = max(1, self.turns_per_chunk - self.overlap_turns)

        for start in range(0, len(turns), step):
            end = min(start + self.turns_per_chunk, len(turns))
            group = turns[start:end]

            text = self._format_turns(group)
            meta = self._extract_turn_metadata(group, start, end)

            chunks.append((text, meta))

            if end >= len(turns):
                break

        return chunks

    def _chunk_whole(self, conversation: Conversation) -> List[tuple]:
        """
        Entire conversation as a single chunk.

        Returns list of (text, metadata) tuples (always length 1).
        """
        if not conversation.turns:
            return []

        text = self._format_turns(conversation.turns)
        meta = self._extract_turn_metadata(
            conversation.turns, 0, len(conversation.turns)
        )
        return [(text, meta)]

    def _chunk_decision_point(self, conversation: Conversation) -> List[tuple]:
        """
        Split at decision/topic boundaries.

        Heuristic: a new segment starts when:
            1. There's a long gap between timestamps (>= 5 min), OR
            2. The user turn is significantly longer than average
               (likely a new topic/question), OR
            3. More than 6 turns have accumulated in the current segment.

        Falls back to turn_group if insufficient signal.
        """
        turns = conversation.turns
        if len(turns) <= self.turns_per_chunk:
            return self._chunk_whole(conversation)

        # Calculate average user turn length for threshold
        user_lengths = [len(t.content) for t in turns if t.role == "user" and t.content]
        avg_user_len = sum(user_lengths) / len(user_lengths) if user_lengths else 100
        long_threshold = avg_user_len * 2.0

        segments = []
        current_segment: List[ConversationTurn] = []
        seg_start = 0

        for i, turn in enumerate(turns):
            should_split = False

            if len(current_segment) >= 6:
                should_split = True

            if (
                turn.role == "user"
                and len(turn.content) > long_threshold
                and len(current_segment) >= 2
            ):
                should_split = True

            if should_split and current_segment:
                text = self._format_turns(current_segment)
                meta = self._extract_turn_metadata(
                    current_segment, seg_start, seg_start + len(current_segment)
                )
                segments.append((text, meta))
                seg_start = i
                current_segment = []

            current_segment.append(turn)

        # Flush remaining
        if current_segment:
            text = self._format_turns(current_segment)
            meta = self._extract_turn_metadata(
                current_segment, seg_start, seg_start + len(current_segment)
            )
            segments.append((text, meta))

        return segments

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _format_turns(turns: List[ConversationTurn]) -> str:
        """Format a group of turns into readable text."""
        lines = []
        for turn in turns:
            prefix = f"[{turn.role}]"
            lines.append(f"{prefix} {turn.content}")
        return "\n\n".join(lines)

    @staticmethod
    def _extract_turn_metadata(
        turns: List[ConversationTurn],
        start_idx: int,
        end_idx: int,
    ) -> Dict[str, Any]:
        """Extract metadata from a group of turns."""
        roles = list({t.role for t in turns})
        timestamps = [t.timestamp for t in turns if t.timestamp]

        meta: Dict[str, Any] = {
            "roles": roles,
            "turn_indices": [start_idx, end_idx],
            "turn_count": len(turns),
        }

        if timestamps:
            meta["first_timestamp"] = timestamps[0]
            meta["last_timestamp"] = timestamps[-1]

        return meta

    def _parse_conversation(self, data: Any, source: str = "unknown") -> Conversation:
        """
        Parse raw JSON data into a Conversation object.

        Supports two formats:
            1. List of turn dicts: [{"role": "...", "content": "..."}, ...]
            2. Object with "turns" key: {"turns": [...], "metadata": {...}}

        Args:
            data: Parsed JSON data.
            source: Source filename for attribution.

        Returns:
            Conversation object.

        Raises:
            ValueError: If the format is unrecognized.
        """
        conv_metadata: dict = {}

        if isinstance(data, dict):
            if "turns" in data:
                conv_metadata = {k: v for k, v in data.items() if k != "turns"}
                data = data["turns"]
            elif "messages" in data:
                conv_metadata = {k: v for k, v in data.items() if k != "messages"}
                data = data["messages"]
            else:
                raise ValueError("Object format must have 'turns' or 'messages' key.")

        if not isinstance(data, list):
            raise ValueError(f"Expected a list of turns, got {type(data).__name__}.")

        turns = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if "role" not in item or "content" not in item:
                logger.warning(f"Skipping turn missing 'role'/'content': {item}")
                continue

            extra = {
                k: v
                for k, v in item.items()
                if k not in ("role", "content", "timestamp")
            }

            turns.append(
                ConversationTurn(
                    role=str(item["role"]),
                    content=str(item["content"]),
                    timestamp=item.get("timestamp"),
                    metadata=extra,
                )
            )

        if not turns:
            raise ValueError("No valid turns found in conversation data.")

        return Conversation(
            turns=turns,
            source=source,
            metadata=conv_metadata,
        )
