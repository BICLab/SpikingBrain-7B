import time
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "spikingbrain-7b"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 32
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


class NeuromorphicMetrics(BaseModel):
    global_spike_rate: float
    global_sparsity_percent: float
    dense_baseline_flops: int
    synaptic_operations_sops: int
    dense_energy_joules: float
    snn_energy_joules: float
    energy_savings_ratio: float


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    neuromorphic_telemetry: Optional[NeuromorphicMetrics] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-sb-{int(time.time()*1000)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "spikingbrain-7b"
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-sb-{int(time.time()*1000)}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "spikingbrain-7b"
    choices: List[ChunkChoice]
    neuromorphic_step_telemetry: Optional[Dict[str, Any]] = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "spikingbrain"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]
