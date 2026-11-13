class OrchestratorError(Exception):
    pass


class LLMError(OrchestratorError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMAPIError(LLMError):
    pass


class LLMFormatError(LLMError):
    pass


class RetrievalError(OrchestratorError):
    pass


class RetrievalTimeoutError(RetrievalError):
    pass


class RetrievalAPIError(RetrievalError):
    pass


class RetrievalBusyError(RetrievalError):
    """Предыдущий батч retrieval всё ещё удерживает потоки пула.

    Контролируемая техническая ошибка: новый пул не создаётся, пока живые
    потоки предыдущего батча могут держать ``_LOCAL_INFERENCE_LOCK``.
    """
