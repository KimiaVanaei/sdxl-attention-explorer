from __future__ import annotations

from typing import Any

from sdxl_attention.processor import (
    RecordingCrossAttnProcessor,
)
from sdxl_attention.store import AttentionStore


class AttentionRecorder:
    """
    Temporarily install cross-attention recording processors.

    The recorder replaces only processors whose paths contain
    ``.attn2.``. Self-attention processors remain unchanged.

    It can be used as a context manager:

    .. code-block:: python

        with AttentionRecorder(pipe.unet, store):
            result = pipe(prompt="a red cat")

    Original processors are restored when the context exits, including
    when generation raises an exception.
    """

    def __init__(
        self,
        unet: Any,
        store: AttentionStore,
        *,
        reset_on_enter: bool = True,
    ) -> None:
        if not hasattr(unet, "attn_processors"):
            raise TypeError(
                "unet must expose an attn_processors property."
            )

        if not hasattr(unet, "set_attn_processor"):
            raise TypeError(
                "unet must expose set_attn_processor()."
            )

        self.unet = unet
        self.store = store
        self.reset_on_enter = reset_on_enter

        self._original_processors: dict[str, Any] | None = None
        self._recording_names: tuple[str, ...] = ()
        self._active = False

    @property
    def is_active(self) -> bool:
        """Return whether recording processors are installed."""
        return self._active

    @property
    def recording_names(self) -> tuple[str, ...]:
        """Return the processor paths currently being recorded."""
        return self._recording_names

    @property
    def recording_count(self) -> int:
        """Return the number of recording processors installed."""
        return len(self._recording_names)

    def install(self) -> int:
        """
        Install recording processors into all ``attn2`` layers.

        Returns
        -------
        int
            Number of recording processors installed.
        """
        if self._active:
            raise RuntimeError(
                "This AttentionRecorder is already active."
            )

        original_processors = dict(
            self.unet.attn_processors
        )

        if not original_processors:
            raise ValueError(
                "The supplied U-Net contains no attention processors."
            )

        new_processors: dict[str, Any] = {}
        recording_names: list[str] = []

        for name, original_processor in (
            original_processors.items()
        ):
            if ".attn2." in name:
                new_processors[name] = (
                    RecordingCrossAttnProcessor(
                        store=self.store,
                        layer_name=name,
                    )
                )

                recording_names.append(name)
            else:
                new_processors[name] = (
                    original_processor
                )

        if not recording_names:
            raise ValueError(
                "No cross-attention processor paths containing "
                "'.attn2.' were found."
            )

        if self.reset_on_enter:
            self.store.reset()

        try:
            # Diffusers consumes processor dictionaries using pop(),
            # so never pass our preserved dictionary directly.
            self.unet.set_attn_processor(
                dict(new_processors)
            )

        except Exception:
            # Attempt to return the model to its earlier state if
            # installation fails partway through.
            self.unet.set_attn_processor(
                dict(original_processors)
            )
            raise

        self._original_processors = (
            original_processors
        )

        self._recording_names = tuple(
            recording_names
        )

        self._active = True

        return self.recording_count

    def restore(self) -> None:
        """
        Restore the processors that existed before installation.

        Calling this method while inactive is harmless.
        """
        if not self._active:
            return

        if self._original_processors is None:
            raise RuntimeError(
                "Original processor state is unavailable."
            )

        self.unet.set_attn_processor(
            dict(self._original_processors)
        )

        self._original_processors = None
        self._recording_names = ()
        self._active = False

    def __enter__(self) -> AttentionRecorder:
        self.install()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exception_type, exception, traceback

        self.restore()

        # Returning False means exceptions inside the context are not
        # suppressed.
        return False
